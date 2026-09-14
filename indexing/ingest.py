"""문서 인제스트: 페이지를 텍스트 경로 / 이미지 경로로 라우팅한다.

라우팅 기준 (docs/plan.md Phase 1):
  - 추출 텍스트 100자 미만 → 이미지 경로 (스캔/이미지 페이지로 간주)
  - 그 외 → 텍스트 경로

D-03에 따라 이미지 경로는 상한(settings.image_route_page_cap)을 둔다.
상한을 넘는 이미지 후보는 캡셔닝 없이 텍스트(OCR)로 대체 인덱싱한다 —
"조용히 누락"시키지 않고 라우팅 통계에 별도로 잡는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from config import settings
from indexing.corpus import Page, iter_pages


class Route(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    IMAGE_OVERFLOW = "image_overflow"  # 이미지 경로 대상이지만 상한 초과 -> OCR로 대체


@dataclass
class RoutedPage:
    page: Page
    route: Route


@dataclass
class RoutingStats:
    total: int = 0
    text: int = 0
    image: int = 0
    image_overflow: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "text_route": self.text,
            "image_route": self.image,
            "image_overflow_route": self.image_overflow,
        }


def route_pages(limit: int | None = None) -> tuple[list[RoutedPage], RoutingStats]:
    """전체(또는 limit개) 페이지를 라우팅하고 통계를 함께 반환한다."""
    stats = RoutingStats()
    routed: list[RoutedPage] = []
    image_route_count = 0

    for page in iter_pages(limit=limit):
        stats.total += 1
        text_len = len(page.text.strip())

        if text_len >= settings.route_min_text_chars:
            stats.text += 1
            routed.append(RoutedPage(page=page, route=Route.TEXT))
            continue

        # 이미지 경로 후보
        if image_route_count < settings.image_route_page_cap and page.image_bytes is not None:
            image_route_count += 1
            stats.image += 1
            routed.append(RoutedPage(page=page, route=Route.IMAGE))
        else:
            stats.image_overflow += 1
            routed.append(RoutedPage(page=page, route=Route.IMAGE_OVERFLOW))

    return routed, stats


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    routed, stats = route_pages(limit=limit)
    print(f"라우팅 완료: {stats.as_dict()}")
