"""BGE-M3 임베딩 + ChromaDB 저장.

D-02에 따라 device는 호출자가 명시한다:
  - 인덱싱 스크립트(run_index.py)는 VLM을 내린 뒤 embed_device_indexing(cuda)로 호출
  - 서빙 코드(retrieve.py)는 embed_device_serving(cpu)로 호출
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import settings
from indexing.ingest import Route, RoutedPage

# 답변 단계에서 원본 페이지 이미지를 다시 VLM에 넣는 경로. 원본 이미지를 디스크에 남긴다.
IMAGE_BACKED_ROUTES = {Route.IMAGE.value, Route.OCR.value}

# 저장 해상도. VLM 입력이 1280px로 축소되므로 그 이상은 디스크만 차지한다.
SAVED_IMAGE_MAX_SIDE_PX = 1280


@dataclass
class IndexedChunk:
    page_id: str
    route: str
    text_for_embedding: str  # TEXT: 본문 / OCR: OCR 텍스트 / IMAGE: 캡션
    source_file: str
    image_path: str | None  # IMAGE·OCR 경로만 채워짐


_embedder_cache: dict[str, SentenceTransformer] = {}


def get_embedder(device: str) -> SentenceTransformer:
    if device not in _embedder_cache:
        kwargs = {}
        if device == "cuda":
            # D-05: fp16으로 배치당 GPU 연산량 절반, 품질 손실은 검색 용도에서 무시할 수준
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        model = SentenceTransformer(settings.embed_model_name, device=device, **kwargs)
        model.max_seq_length = settings.embed_max_seq_length
        _embedder_cache[device] = model
    return _embedder_cache[device]


def source_file_from_page_id(page_id: str) -> str:
    """SDS KoPub의 id는 '경로/파일명_페이지' 형태다. 정확한 문서 매핑은 annotations.parquet 조인으로."""
    return page_id


def _safe_name(page_id: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in page_id)[:150]


def image_path_for(page_id: str) -> str:
    return str(settings.rendered_pages_dir / f"{_safe_name(page_id)}.png")


def save_page_image(page_id: str, image: Image.Image) -> str:
    """원본 페이지 이미지를 긴 변 1280px로 줄여 디스크에 저장하고 경로를 반환한다. 이미 있으면 건너뛴다."""
    settings.rendered_pages_dir.mkdir(parents=True, exist_ok=True)
    path = settings.rendered_pages_dir / f"{_safe_name(page_id)}.png"
    if path.exists():
        return str(path)
    w, h = image.size
    longest = max(w, h)
    if longest > SAVED_IMAGE_MAX_SIDE_PX:
        scale = SAVED_IMAGE_MAX_SIDE_PX / longest
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(path, format="PNG")
    return str(path)


def build_chunks(routed_pages: list[RoutedPage], captions: dict[str, str]) -> list[IndexedChunk]:
    """라우팅 결과 + 캡션 결과를 임베딩 대상 청크로 변환한다. BLANK는 제외."""
    chunks: list[IndexedChunk] = []
    for rp in routed_pages:
        page = rp.page
        route = rp.route

        if route == Route.BLANK:
            continue
        if route == Route.TEXT:
            text_for_embedding = page.text
        elif route == Route.OCR:
            text_for_embedding = page.ocr
        elif route == Route.IMAGE:
            text_for_embedding = captions.get(page.id, "")
            if not text_for_embedding.strip():
                # 캡션 실패 페이지는 OCR/텍스트로 대체해 누락시키지 않는다
                text_for_embedding = page.ocr or page.text
        else:  # IMAGE_OVERFLOW
            text_for_embedding = page.ocr or page.text

        if not text_for_embedding.strip():
            continue

        chunks.append(
            IndexedChunk(
                page_id=page.id,
                route=route.value,
                text_for_embedding=text_for_embedding,
                source_file=source_file_from_page_id(page.id),
                image_path=image_path_for(page.id) if route.value in IMAGE_BACKED_ROUTES else None,
            )
        )
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(settings.collection_name)


def embed_and_store(chunks: list[IndexedChunk], device: str, batch_size: int | None = None, progress: bool = True) -> int:
    """청크를 임베딩하여 ChromaDB에 저장한다. 저장된 청크 수를 반환한다."""
    if not chunks:
        return 0

    batch_size = batch_size or settings.embed_batch_size
    embedder = get_embedder(device)
    collection = get_collection()

    rng = range(0, len(chunks), batch_size)
    if progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="embedding", unit="batch")

    stored = 0
    for i in rng:
        batch = chunks[i : i + batch_size]
        texts = [c.text_for_embedding for c in batch]
        vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        collection.upsert(
            ids=[c.page_id for c in batch],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[
                {
                    "route": c.route,
                    "source_file": c.source_file,
                    "image_path": c.image_path or "",
                }
                for c in batch
            ],
        )
        stored += len(batch)

    return stored
