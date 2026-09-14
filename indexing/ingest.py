"""문서 인제스트: 페이지를 네 갈래로 라우팅한다 (D-04).

  TEXT   : 추출 텍스트 >= route_min_text_chars           -> 텍스트 그대로 인덱싱
  OCR    : 텍스트 빈약, OCR >= route_min_ocr_chars       -> OCR 텍스트로 인덱싱 (VLM 생략)
  BLANK  : 텍스트·OCR 모두 빈약, 잉크 비율 < threshold    -> 인덱싱 제외 (백지)
  IMAGE  : 그 외 (표·차트·그림 등 진짜 시각 페이지)       -> VLM 캡셔닝
  IMAGE_OVERFLOW: IMAGE 대상이지만 상한 초과              -> OCR/텍스트로 대체

전체 코퍼스 실측(2026-09-14): 텍스트<100자 3,707장 중 OCR>=100자 1,387장,
둘 다 거의 빈 1,478장(표본 84장 중 80장이 잉크 0.2% 미만 백지). VLM 대상은 약 900장.

메모리: TEXT/OCR 경로 페이지는 image_bytes를 버린다. 4만 페이지 원본 이미지를
전부 들고 있으면 21GB가 되므로, 답변 단계에서 원본 이미지가 필요한 경로만 유지한다.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import Enum

from PIL import Image

from config import settings
from indexing.corpus import Page, iter_pages


class Route(str, Enum):
    TEXT = "text"
    OCR = "ocr"
    IMAGE = "image"
    BLANK = "blank"
    IMAGE_OVERFLOW = "image_overflow"


@dataclass
class RoutedPage:
    page: Page
    route: Route


@dataclass
class RoutingStats:
    total: int = 0
    text: int = 0
    ocr: int = 0
    image: int = 0
    blank: int = 0
    image_overflow: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "text_route": self.text,
            "ocr_route": self.ocr,
            "image_route": self.image,
            "blank_skipped": self.blank,
            "image_overflow_route": self.image_overflow,
        }


def ink_ratio(image_bytes: bytes, thumb: int = 400, dark_below: int = 200) -> float:
    """썸네일 그레이스케일에서 어두운 픽셀 비율. 백지 판정용."""
    im = Image.open(io.BytesIO(image_bytes)).convert("L")
    im.thumbnail((thumb, thumb))
    hist = im.histogram()  # 256 bins
    dark = sum(hist[:dark_below])
    total = sum(hist)
    return dark / total if total else 0.0


def decide_route(page: Page) -> Route:
    """상한(cap)을 고려하지 않은 순수 라우팅 결정."""
    text_len = len(page.text.strip())
    ocr_len = len(page.ocr.strip())

    if text_len >= settings.route_min_text_chars:
        return Route.TEXT
    if ocr_len >= settings.route_min_ocr_chars:
        return Route.OCR
    if page.image_bytes is None:
        # 이미지가 없고 텍스트도 없으면 인덱싱할 내용이 없다
        return Route.BLANK
    if ink_ratio(page.image_bytes) < settings.blank_ink_threshold:
        return Route.BLANK
    return Route.IMAGE


def route_pages(limit: int | None = None) -> tuple[list[RoutedPage], RoutingStats]:
    """전체(또는 limit개) 페이지를 라우팅하고 통계를 함께 반환한다."""
    stats = RoutingStats()
    routed: list[RoutedPage] = []
    image_route_count = 0

    for page in iter_pages(limit=limit):
        stats.total += 1
        route = decide_route(page)

        if route == Route.IMAGE:
            if image_route_count >= settings.image_route_page_cap:
                route = Route.IMAGE_OVERFLOW
            else:
                image_route_count += 1

        if route == Route.TEXT:
            stats.text += 1
            page.image_bytes = None
        elif route == Route.OCR:
            stats.ocr += 1
            # 답변 단계에서 원본 스캔 이미지를 쓰므로 유지
        elif route == Route.IMAGE:
            stats.image += 1
        elif route == Route.BLANK:
            stats.blank += 1
            page.image_bytes = None
        else:  # IMAGE_OVERFLOW
            stats.image_overflow += 1
            page.image_bytes = None

        routed.append(RoutedPage(page=page, route=route))

    return routed, stats


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _, stats = route_pages(limit=limit)
    print(f"라우팅 완료: {stats.as_dict()}")
