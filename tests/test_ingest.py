"""라우팅 로직 단위 테스트 (실제 코퍼스/모델 다운로드 없이 검증 가능)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch

from indexing.ingest import Route, route_pages
from indexing.corpus import Page


def _pages():
    return [
        Page(id="p1", text="충분히 긴 본문 텍스트입니다. " * 10, ocr="", image_bytes=None),
        Page(id="p2", text="짧음", ocr="OCR 텍스트", image_bytes=b"fake-png-bytes"),
        Page(id="p3", text="", ocr="", image_bytes=b"fake-png-bytes-2"),
    ]


def test_text_route_for_long_text():
    with patch("indexing.ingest.iter_pages", return_value=iter(_pages())):
        routed, stats = route_pages()
    assert routed[0].route == Route.TEXT
    assert stats.text == 1


def test_image_route_for_short_text_with_image():
    with patch("indexing.ingest.iter_pages", return_value=iter(_pages())):
        routed, stats = route_pages()
    assert routed[1].route == Route.IMAGE
    assert routed[2].route == Route.IMAGE
    assert stats.image == 2


def test_overflow_when_cap_exceeded():
    with patch("indexing.ingest.settings") as mock_settings:
        mock_settings.route_min_text_chars = 100
        mock_settings.image_route_page_cap = 1
        with patch("indexing.ingest.iter_pages", return_value=iter(_pages())):
            routed, stats = route_pages()
    assert stats.image == 1
    assert stats.image_overflow == 1


def test_stats_total_matches_page_count():
    with patch("indexing.ingest.iter_pages", return_value=iter(_pages())):
        routed, stats = route_pages()
    assert stats.total == len(_pages())
    assert len(routed) == len(_pages())
