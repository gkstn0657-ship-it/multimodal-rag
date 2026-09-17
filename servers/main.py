"""FastAPI 엔드포인트: POST /ask.

Phase 5 요구사항 반영:
  - VLM 서버 다운 시 503 + 명확한 에러 메시지
  - 검색 결과 0건 시 "문서에서 찾을 수 없습니다" 정상 응답 (예외 아님)
  - 단계별(검색/리랭킹/생성) 소요시간 로깅
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from servers import metrics
from servers.answer import generate_from_chunks
from servers.retrieve import rerank, search
from servers.vlm_client import VLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multimodal_rag")

app = FastAPI(title="multimodal-rag")
_vlm_client = VLMClient()

# 원본 페이지 이미지를 브라우저에서 바로 볼 수 있게 정적 서빙 (테스트 화면용, D-25)
settings.rendered_pages_dir.mkdir(parents=True, exist_ok=True)
app.mount("/pages", StaticFiles(directory=str(settings.rendered_pages_dir)), name="pages")

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


class AskRequest(BaseModel):
    question: str


class ChunkOut(BaseModel):
    page_id: str
    route: str
    score: float
    text: str
    image_url: str | None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[ChunkOut]
    timing: dict


@app.get("/health")
def health() -> dict:
    vlm_ok = _vlm_client.ping()
    return {"status": "ok" if vlm_ok else "degraded", "vlm_available": vlm_ok}


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.summary()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question이 비어 있습니다.")

    timing = metrics.RequestTiming()

    with metrics.stopwatch() as elapsed_retrieve:
        try:
            candidates = search(req.question)
        except Exception as exc:  # noqa: BLE001
            logger.exception("검색 단계 실패")
            raise HTTPException(status_code=503, detail=f"검색 서비스를 사용할 수 없습니다: {exc}") from exc
    timing.retrieve_sec = elapsed_retrieve()

    with metrics.stopwatch() as elapsed_rerank:
        chunks = rerank(req.question, candidates)
    timing.rerank_sec = elapsed_rerank()

    def _to_chunk_out(c) -> ChunkOut:
        image_url = None
        if c.image_path:
            name = Path(c.image_path).name
            if (settings.rendered_pages_dir / name).exists():
                image_url = f"/pages/{name}"
        return ChunkOut(page_id=c.page_id, route=c.route, score=c.score, text=c.text, image_url=image_url)

    if not chunks:
        timing.total_sec = timing.retrieve_sec + timing.rerank_sec
        metrics.record(timing)
        return AskResponse(answer="문서에서 찾을 수 없습니다.", sources=[], chunks=[], timing=timing.as_dict())

    if not _vlm_client.ping():
        raise HTTPException(status_code=503, detail="VLM 서버가 응답하지 않습니다. Ollama 서버 상태를 확인하세요.")

    with metrics.stopwatch() as elapsed_generate:
        try:
            result = generate_from_chunks(req.question, chunks, client=_vlm_client)
        except Exception as exc:  # noqa: BLE001
            logger.exception("답변 생성 단계 실패")
            raise HTTPException(status_code=503, detail=f"답변 생성에 실패했습니다: {exc}") from exc
    timing.generate_sec = elapsed_generate()
    timing.total_sec = timing.retrieve_sec + timing.rerank_sec + timing.generate_sec
    metrics.record(timing)

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        chunks=[_to_chunk_out(c) for c in chunks],
        timing=timing.as_dict(),
    )
