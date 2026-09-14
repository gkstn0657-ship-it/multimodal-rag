"""이미지 경로 페이지에 대한 VLM 캡셔닝.

캡션 실패(품질 저하)는 파이프라인이 아니라 프롬프트만 조정한다는
docs/plan.md Phase 1의 원칙을 지키기 위해, 프롬프트를 이 파일 하나에 고정한다.

캡션 결과는 JSONL 캐시(settings.caption_cache_path)에 한 줄씩 바로 기록한다.
수 시간짜리 작업이 중간에 죽어도 다시 실행하면 캐시된 페이지는 건너뛴다.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from config import settings
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
    error: str | None = None


def caption_page(client: VLMClient, page_id: str, image, retries: int = 1) -> CaptionResult:
    """한 페이지를 캡셔닝한다. 실패 시 retries만큼 재시도하고, 끝내 실패하면 error를 채워 반환한다."""
    t0 = time.time()
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            caption = client.caption(image, CAPTION_PROMPT, max_tokens=settings.caption_max_tokens)
            return CaptionResult(page_id=page_id, caption=caption, elapsed_sec=time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2)
    return CaptionResult(page_id=page_id, caption="", elapsed_sec=time.time() - t0, error=last_err)


class CaptionCache:
    """append-only JSONL 캐시. 스레드 안전."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.caption_cache_path
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("caption"):
                        self._data[rec["page_id"]] = rec["caption"]

    def __contains__(self, page_id: str) -> bool:
        return page_id in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, page_id: str) -> str | None:
        return self._data.get(page_id)

    def put(self, result: CaptionResult) -> None:
        if not result.caption:
            return
        rec = {"page_id": result.page_id, "caption": result.caption, "elapsed_sec": round(result.elapsed_sec, 2)}
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._data[result.page_id] = result.caption
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def as_dict(self) -> dict[str, str]:
        return dict(self._data)
