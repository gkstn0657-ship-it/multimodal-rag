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
    "이 문서 페이지의 내용을 한국어로 사실대로 설명하라.\n"
    "- 이미지에 실제로 표(행·열 구조)가 있을 때만 마크다운 표로 옮겨라. "
    "표가 없으면 표를 만들지 마라. 없는 칸이나 숫자를 채워 넣지 마라.\n"
    "- 차트나 그래프가 있으면 보이는 축·수치·범례만 그대로 적어라. "
    "값이 안 보이면 '수치 불명'이라고 쓰고 추세나 원인을 추측해서 서술하지 마라.\n"
    "- 사진, 도장, 손글씨, 캡션(사진 설명) 문구가 있으면 반드시 그대로 옮겨 적어라. "
    "이런 페이지에서는 사진에 찍힌 캡션 문구가 가장 중요한 정보다.\n"
    "- 제목·표지 페이지처럼 내용이 적으면 보이는 텍스트만 그대로 적고, "
    "본문 내용을 상상해서 만들어내지 마라.\n"
    "- 장식용 색깔 막대, 도형, 배경 무늬처럼 정보가 없는 디자인 요소는 설명하지 말고 무시하라.\n"
    "- 영문 표 헤더나 약어는 임의로 번역하지 말고 원문 그대로 표기하라.\n"
    "- 숫자·글자는 이미지에 보이는 그대로 정확히 옮겨 적고, 흐릿하거나 안 보이면 "
    "'판독 불가'라고 써라. 추측해서 채우지 마라."
)


# D-15: 검색용 전사(轉寫) 우선 프롬프트. v2(위)는 창작을 막았지만 캡션이 짧아져 검색 신호가 약해졌다
# (합성 질의 194건에서 OCR 단독보다 낮음). 보이는 글자를 그대로 옮기게 해 검색 가능한 문자열을 확보하고,
# 해석·추측은 마지막 한 문장으로만 제한한다. 창작 금지 원칙(D-07)은 그대로다.
CAPTION_PROMPT_V3 = (
    "이 문서 페이지에 보이는 글자를 있는 그대로 모두 옮겨 적어라.\n"
    "- 제목, 소제목, 표의 머리글과 각 칸의 값, 차트의 제목·축 이름·범례·수치, 사진 설명 문구, 도장·손글씨를 포함한다.\n"
    "- 표는 한 행씩 '항목: 값, 값, …' 형식으로 옮겨라. 차트는 '범례: 수치' 형식으로 읽히는 값만 적어라.\n"
    "- 보이지 않거나 흐릿한 글자는 '(판독 불가)'라고 쓰고 추측해서 채우지 마라. 없는 표·숫자·이름을 만들지 마라.\n"
    "- 장식용 도형·색 막대·배경 무늬는 적지 마라.\n"
    "- 글자를 다 옮긴 뒤 마지막 줄에 '요약:'으로 시작하는 한 문장으로 이 페이지가 무엇인지 적어라."
)

@dataclass
class CaptionResult:
    page_id: str
    caption: str
    elapsed_sec: float
    error: str | None = None


def caption_page(client: VLMClient, page_id: str, image, retries: int = 1, prompt: str = CAPTION_PROMPT) -> CaptionResult:
    """한 페이지를 캡셔닝한다. 실패 시 retries만큼 재시도하고, 끝내 실패하면 error를 채워 반환한다."""
    t0 = time.time()
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            caption = client.caption(image, prompt, max_tokens=settings.caption_max_tokens)
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
