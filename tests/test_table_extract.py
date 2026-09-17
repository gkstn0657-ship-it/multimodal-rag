"""표 마크다운 추출 단위 테스트 (D-31)."""

from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.table_extract import merge_text_and_tables, tables_markdown  # noqa: E402


def test_merge_appends_table_markdown_after_original_text():
    out = merge_text_and_tables("원문 텍스트", "| a | b |\n|---|---|\n| 1 | 2 |")
    assert out.startswith("원문 텍스트")
    assert "[표 구조]" in out
    assert "| a | b |" in out


def test_merge_returns_original_when_no_table():
    assert merge_text_and_tables("원문 텍스트", "") == "원문 텍스트"


def test_tables_markdown_empty_on_plain_text_page():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "표가 없는 평범한 문단입니다.")
    assert tables_markdown(page) == ""
    doc.close()


def test_tables_markdown_extracts_grid_as_markdown():
    doc = fitz.open()
    page = doc.new_page()
    # 2x2 표를 실제 선으로 그려 PyMuPDF가 표로 인식하게 한다.
    rows = [72, 100, 128]
    cols = [72, 150, 228]
    for y in rows:
        page.draw_line((cols[0], y), (cols[-1], y))
    for x in cols:
        page.draw_line((x, rows[0]), (x, rows[-1]))
    page.insert_text((80, 92), "이름")
    page.insert_text((158, 92), "값")
    page.insert_text((80, 120), "정원")
    page.insert_text((158, 120), "133")

    md = tables_markdown(page)
    doc.close()
    assert "|" in md
    assert "133" in md
