"""이미지 경로 페이지에 대한 VLM 캡셔닝.

캡션 실패(품질 저하)는 파이프라인이 아니라 프롬프트만 조정한다는
docs/plan.md Phase 1의 원칙을 지키기 위해, 프롬프트를 이 파일 하나에 고정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from servers.vlm_client import VLMClient

CAPTION_PROMPT = (
    "이 문서 페이지의 내용을 한국어로 설명하라. 표는 행·열 구조를 유지해 "
    "마크다운으로, 차트는 축·수치·추세를 구체적으로 기술하라. "
    "영문 표 헤더나 약어는 임의로 번역하지 말고 원문 그대로 표기하라. "
    "숫자는 이미지에 보이는 그대로 정확히 옮겨 적어라."
)


@dataclass
class CaptionResult:
    page_id: str
    caption: str
    elapsed_sec: float


def caption_page(client: VLMClient, page_id: str, image) -> CaptionResult:
    import time

    t0 = time.time()
    caption = client.caption(image, CAPTION_PROMPT)
    elapsed = time.time() - t0
    return CaptionResult(page_id=page_id, caption=caption, elapsed_sec=elapsed)
