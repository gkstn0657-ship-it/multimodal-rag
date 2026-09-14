"""답변 생성.

- 이미지 경로 출신 청크: 캡션이 아니라 원본 페이지 이미지를 VLM에 직접 입력한다.
- 텍스트 경로 출신 청크: 텍스트를 컨텍스트로 입력한다.
- D-02: 답변 단계에 투입하는 원본 이미지는 최대 settings.answer_image_cap장으로 제한하고,
  넘치는 이미지 출신 청크는 캡션 텍스트를 컨텍스트에 추가한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import settings
from servers.retrieve import RetrievedChunk, retrieve
from servers.vlm_client import VLMClient

SYSTEM_PROMPT = (
    "제공된 문서에 근거해서만 답하라. 근거가 없으면 "
    "'문서에서 찾을 수 없습니다'라고 답하라. 답변 끝에 출처(파일명·페이지)를 표기하라."
)


@dataclass
class AnswerResult:
    answer: str
    sources: list[str]
    used_chunks: list[RetrievedChunk]


def _build_context_text(chunks: list[RetrievedChunk], images_used: list[str]) -> str:
    """텍스트 경로 청크 + 이미지 상한 초과분(캡션)을 하나의 컨텍스트 문자열로 합친다."""
    parts = []
    for c in chunks:
        if c.image_path and c.image_path in images_used:
            continue  # 원본 이미지로 직접 투입되므로 텍스트 컨텍스트에서 중복 제외
        parts.append(f"[출처: {c.source_file}]\n{c.text}")
    return "\n\n".join(parts)


def generate_from_chunks(query: str, chunks: list[RetrievedChunk], client: VLMClient | None = None) -> AnswerResult:
    """검색된 청크로부터 답변을 생성한다 (검색과 생성 단계를 분리해 계측 가능하게 함)."""
    client = client or VLMClient()

    if not chunks:
        return AnswerResult(answer="문서에서 찾을 수 없습니다.", sources=[], used_chunks=[])

    # 원본 이미지가 있는 청크(IMAGE·OCR 경로) 중 상한만큼만 이미지로 투입 (D-02, D-04)
    image_chunks = [c for c in chunks if c.image_path]
    images_to_send = [c.image_path for c in image_chunks[: settings.answer_image_cap]]

    context_text = _build_context_text(chunks, images_to_send)
    user_text = f"질문: {query}\n\n참고 문서:\n{context_text}"

    response = client.answer_with_images(SYSTEM_PROMPT, user_text, images_to_send)

    sources = [c.source_file for c in chunks]
    return AnswerResult(answer=response, sources=sources, used_chunks=chunks)


def answer(query: str, client: VLMClient | None = None) -> AnswerResult:
    chunks = retrieve(query)
    return generate_from_chunks(query, chunks, client=client)
