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


def _text_char_budget(num_images: int) -> int:
    """이미지 수에 따라 텍스트 컨텍스트에 쓸 수 있는 글자 수를 계산한다 (D-11)."""
    tokens = (
        settings.vlm_num_ctx
        - settings.answer_max_tokens
        - settings.answer_context_reserved_tokens
        - settings.answer_context_tokens_per_image * num_images
    )
    return max(int(tokens * settings.answer_context_chars_per_token), 300)


def _build_context_text(chunks: list[RetrievedChunk], images_used: list[str]) -> str:
    """텍스트 경로 청크 + 이미지 상한 초과분(캡션)을 하나의 컨텍스트 문자열로 합친다.

    D-09: 청크 5개(원문은 페이지당 최대 13,000자) + 이미지 2장을 그대로 합치면
    4k 컨텍스트를 넘는다(실측 5,519토큰).
    D-11: 500자 고정 절단은 정답 표(민원 통계, 면세 한도 등)를 정확히 잘라냈다.
    이미지 수에 따라 전체 글자 예산을 잡고, 재랭킹 순서대로 채운다. 이미지가 없는
    대부분의 질의에서는 예산이 훨씬 넉넉해진다.
    """
    budget = _text_char_budget(len(images_used))
    per_chunk_cap = settings.answer_context_max_chars_per_chunk
    parts = []
    for c in chunks:
        if c.image_path and c.image_path in images_used:
            continue  # 원본 이미지로 직접 투입되므로 텍스트 컨텍스트에서 중복 제외
        if budget <= 0:
            break
        allowed = min(per_chunk_cap, budget)
        text = c.text if len(c.text) <= allowed else c.text[:allowed] + " …(생략)"
        budget -= len(text)
        parts.append(f"[출처: {c.source_file}]\n{text}")
    return "\n\n".join(parts)


def generate_from_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    client: VLMClient | None = None,
    use_images: bool = True,
) -> AnswerResult:
    """검색된 청크로부터 답변을 생성한다 (검색과 생성 단계를 분리해 계측 가능하게 함).

    use_images=False: 원본 이미지를 VLM에 넣지 않고 텍스트(캡션 포함)만 사용한다.
    Phase 3 생성 품질 평가에서 "캡션 텍스트는 있지만 원본 이미지는 안 보여주는" 비교
    조건을 만들 때 쓴다 — VLM에 원본 이미지를 직접 보여주는 것 자체의 기여를 분리해서 본다.
    """
    client = client or VLMClient()

    if not chunks:
        return AnswerResult(answer="문서에서 찾을 수 없습니다.", sources=[], used_chunks=[])

    # 원본 이미지가 있는 청크(IMAGE·OCR 경로) 중 상한만큼만 이미지로 투입 (D-02, D-04)
    # D-11: 재랭킹 상위 answer_image_max_rank 안의 청크만. 5위 무관 이미지가 답을 망친 실측 때문.
    top_for_images = chunks[: settings.answer_image_max_rank]
    image_chunks = [c for c in top_for_images if c.image_path] if use_images else []
    images_to_send = [c.image_path for c in image_chunks[: settings.answer_image_cap]]

    context_text = _build_context_text(chunks, images_to_send)
    user_text = f"질문: {query}\n\n참고 문서:\n{context_text}"

    response = client.answer_with_images(
        SYSTEM_PROMPT, user_text, images_to_send, max_tokens=settings.answer_max_tokens
    )

    sources = [c.source_file for c in chunks]
    return AnswerResult(answer=response, sources=sources, used_chunks=chunks)


def answer(query: str, client: VLMClient | None = None) -> AnswerResult:
    chunks = retrieve(query)
    return generate_from_chunks(query, chunks, client=client)
