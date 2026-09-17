"""절 경계·no_content 판정 단위 테스트 (D-27, D-28)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.section_bounds import (  # noqa: E402
    build_section_map,
    detect_section_title,
    is_no_content,
    parse_doc_and_page,
)


def test_parse_doc_and_page_splits_trailing_page_number():
    assert parse_doc_and_page("dart/00258801_2023_1_[카카오]사업보고서_211") == (
        "dart/00258801_2023_1_[카카오]사업보고서",
        211,
    )


def test_parse_doc_and_page_none_without_trailing_number():
    assert parse_doc_and_page("no_page_suffix") is None


def test_cover_page_is_no_content():
    text = "2018년 러시아 주요 경제 지표\n2019. 3.\n러시아주재관\n요약: 표지입니다."
    assert is_no_content(text, "image") is True


def test_chart_page_with_data_is_content():
    # 실측: 표지 신호("제3장")가 있어도 <그림>·데이터가 있으면 내용 있음으로 본다.
    text = "제3장 산불 위험성 평가 및 진단\n<그림 39> 마을 주도로 (1)\n<그림 40> 마을 주도로 (2)"
    assert is_no_content(text, "image") is False


def test_markdown_table_is_content():
    text = "5. 이전 직종코드\n| 기존 직종코드 | 새로운 직종코드 |\n| --- | --- |\n| 1.CAD설계사 | 1.CAD설계사 |"
    assert is_no_content(text, "image") is False


def test_number_with_particle_is_not_false_positive():
    # D-27 버그 수정: "30명을"처럼 조사가 바로 붙어도 데이터 신호로 인식해야 한다.
    text = "제4장 치매환자 가족의 생활실태\n요양보호사 자격증이 있다고 응답한 30명을 대상으로 함"
    assert is_no_content(text, "image") is False


def test_section_intro_list_without_data_is_no_content():
    # D-27에서 찾은 실제 사례: 소제목 목록만 있고 표·수치가 없는 절 소개 페이지.
    text = "# VI 재정운영지표 분석\n\n1. 재정수지비율\n2. 재정지표\n3. 결산지표"
    assert is_no_content(text, "image") is True


def test_text_route_page_is_always_content_unless_toc():
    # 텍스트 경로는 route_min_text_chars(100자) 이상이 보장되므로 목차가 아니면 내용 있음으로 본다.
    text = "일반 본문 페이지입니다. " * 20
    assert is_no_content(text, "text") is False


def test_toc_dotleader_page_is_no_content_regardless_of_route():
    text = "목 차\n" + "\n".join(f"{i}장 ....................... {i}" for i in range(1, 6))
    assert is_no_content(text, "text") is True


def test_detect_section_title_finds_chapter_pattern():
    text = "제2장 정책계획의 개요\n\n2.1 정책계획의 배경 및 목적"
    assert detect_section_title(text) == "제2장 정책계획의 개요"


def test_detect_section_title_none_when_no_pattern():
    assert detect_section_title("평범한 본문 첫 줄입니다.\n다음 줄.") is None


def test_build_section_map_propagates_title_and_links_neighbors():
    records = [
        {"id": "doc_1", "document": "제1장 개요\n내용", "metadata": {"route": "text"}},
        {"id": "doc_2", "document": "본문 계속", "metadata": {"route": "text"}},
        {"id": "doc_3", "document": "제2장 분석\n내용", "metadata": {"route": "text"}},
    ]
    section_map = build_section_map(records)
    assert section_map["doc_1"]["section_title"] == "제1장 개요"
    assert section_map["doc_2"]["section_title"] == "제1장 개요"  # 이전 절 제목 유지
    assert section_map["doc_3"]["section_title"] == "제2장 분석"
    assert section_map["doc_1"]["prev_id"] is None
    assert section_map["doc_2"]["prev_id"] == "doc_1"
    assert section_map["doc_2"]["next_id"] == "doc_3"
    assert section_map["doc_3"]["next_id"] is None


def test_build_section_map_skips_ids_without_page_suffix():
    records = [{"id": "no_page_number", "document": "text", "metadata": {"route": "text"}}]
    assert build_section_map(records) == {}
