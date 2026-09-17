"""PDF 표를 마크다운으로 뽑는다 (D-31).

D-30 직후 발견한 "5성급" 오독(충남 공공기관 경영평가 662쪽, D-31 기록)은 2단 표가 PDF 텍스트
추출 과정에서 한 줄로 평평하게 풀리며 셀 간 행·열 대응이 사라진 것이 원인이었다. 원본 PDF가
있는 DART 코퍼스는 PyMuPDF의 표 검출기(`page.find_tables()`)로 셀 구조를 그대로 뽑아 마크다운
표로 되돌릴 수 있다 — 모델이 필요 없다.

SDS KoPub은 이미 추출된 텍스트만 파켓으로 받아 원본 PDF가 없어 이 방식을 쓸 수 없다(D-31).
"""

from __future__ import annotations

import fitz  # PyMuPDF


def tables_markdown(page: "fitz.Page") -> str:
    """페이지에서 검출된 표를 마크다운 문자열로 합쳐 돌려준다. 표가 없으면 빈 문자열."""
    try:
        tabs = page.find_tables()
    except Exception:  # noqa: BLE001 — 표 검출 실패는 원문 텍스트만으로 계속 진행
        return ""
    if not tabs.tables:
        return ""
    parts = []
    for t in tabs.tables:
        try:
            md = t.to_markdown()
        except Exception:  # noqa: BLE001
            continue
        if md and md.strip():
            parts.append(md.strip())
    return "\n\n".join(parts)


def merge_text_and_tables(raw_text: str, table_md: str) -> str:
    """원문 텍스트 뒤에 표 마크다운을 이어 붙인다. 원문은 그대로 보존한다(프로즈가 섞여 있을 수 있어서).

    표가 없으면 원문 그대로 돌려준다.
    """
    if not table_md:
        return raw_text
    return f"{raw_text}\n\n[표 구조]\n{table_md}"
