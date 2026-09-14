"""라우팅 로직 단위 테스트 (실제 코퍼스/모델 다운로드 없이 검증 가능)."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.corpus import Page  # noqa: E402
from indexing.ingest import Route, decide_route, ink_ratio, route_pages  # noqa: E402


def _png(draw_content: bool) -> bytes:
    im = Image.new("RGB", (200, 280), "white")
    if draw_content:
        d = ImageDraw.Draw(im)
        d.rectangle([20, 20, 180, 120], fill="black")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


BLANK_PNG = _png(False)
CHART_PNG = _png(True)


def _pages():
    return [
        Page(id="text", text="충분히 긴 본문 텍스트입니다. " * 10, ocr="", image_bytes=CHART_PNG),
        Page(id="ocr", text="짧음", ocr="스캔된 본문 OCR 결과입니다. " * 10, image_bytes=CHART_PNG),
        Page(id="chart", text="", ocr="", image_bytes=CHART_PNG),
        Page(id="blank", text="", ocr="", image_bytes=BLANK_PNG),
        Page(id="noimg", text="", ocr="", image_bytes=None),
    ]


def test_ink_ratio_separates_blank_from_content():
    assert ink_ratio(BLANK_PNG) < 0.002
    assert ink_ratio(CHART_PNG) > 0.1


def test_decide_route_four_ways():
    p = {pg.id: pg for pg in _pages()}
    assert decide_route(p["text"]) == Route.TEXT
    assert decide_route(p["ocr"]) == Route.OCR
    assert decide_route(p["chart"]) == Route.IMAGE
    assert decide_route(p["blank"]) == Route.BLANK
    assert decide_route(p["noimg"]) == Route.BLANK


def test_route_pages_stats_and_memory_release():
    with patch("indexing.ingest.iter_pages", return_value=iter(_pages())):
        routed, stats = route_pages()
    assert stats.as_dict() == {
        "total": 5,
        "text_route": 1,
        "ocr_route": 1,
        "image_route": 1,
        "blank_skipped": 2,
        "image_overflow_route": 0,
    }
    by_id = {rp.page.id: rp for rp in routed}
    # TEXT/BLANK는 이미지 바이트를 버리고, IMAGE/OCR은 유지한다
    assert by_id["text"].page.image_bytes is None
    assert by_id["blank"].page.image_bytes is None
    assert by_id["chart"].page.image_bytes is not None
    assert by_id["ocr"].page.image_bytes is not None


def test_overflow_when_cap_exceeded():
    pages = [Page(id=f"c{i}", text="", ocr="", image_bytes=CHART_PNG) for i in range(3)]
    with patch("indexing.ingest.settings") as s:
        s.route_min_text_chars = 100
        s.route_min_ocr_chars = 100
        s.blank_ink_threshold = 0.002
        s.image_route_page_cap = 1
        with patch("indexing.ingest.iter_pages", return_value=iter(pages)):
            routed, stats = route_pages()
    assert stats.image == 1
    assert stats.image_overflow == 2
    assert [rp.route for rp in routed] == [Route.IMAGE, Route.IMAGE_OVERFLOW, Route.IMAGE_OVERFLOW]
