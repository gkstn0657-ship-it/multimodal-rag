"""build_chunks 의 경로별 임베딩 텍스트 규칙 (D-15) 단위 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.corpus import Page  # noqa: E402
from indexing.embed_store import build_chunks  # noqa: E402
from indexing.ingest import Route, RoutedPage  # noqa: E402


def _rp(pid, route, text="", ocr=""):
    return RoutedPage(page=Page(id=pid, text=text, ocr=ocr, image_bytes=None), route=route)


def test_image_route_embeds_caption_and_ocr_union():
    chunks = build_chunks([_rp("p", Route.IMAGE, text="", ocr="2022 동구 성인지 통계")], {"p": "요약: 건강 지표 4.6"})
    assert len(chunks) == 1
    assert "요약: 건강 지표 4.6" in chunks[0].text_for_embedding
    assert "2022 동구 성인지 통계" in chunks[0].text_for_embedding


def test_image_route_without_caption_falls_back_to_ocr():
    chunks = build_chunks([_rp("p", Route.IMAGE, ocr="OCR 조각")], {})
    assert chunks[0].text_for_embedding == "OCR 조각"


def test_image_route_without_ocr_keeps_caption():
    chunks = build_chunks([_rp("p", Route.IMAGE)], {"p": "사진: 광복군 포대"})
    assert chunks[0].text_for_embedding == "사진: 광복군 포대"


def test_blank_route_is_skipped_and_text_route_uses_text():
    chunks = build_chunks([_rp("b", Route.BLANK), _rp("t", Route.TEXT, text="본문 " * 30)], {})
    assert [c.page_id for c in chunks] == ["t"]
    assert chunks[0].text_for_embedding.startswith("본문")
