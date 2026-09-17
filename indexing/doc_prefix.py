"""페이지 ID로부터 문서명(+DART는 회사·사업연도) 접두어를 만든다 (D-21).

BM25 색인 시에만 청크 텍스트 앞에 붙여 넣는다(임베딩 텍스트·답변 컨텍스트는 안 바꾼다, 0단계).
목적: "LG화학 배당금" 질의에 NAVER 페이지가 걸리는 것처럼, 밀집·BM25 둘 다 페이지 본문에
회사명·문서명이 없으면 못 잡는 경우를 접두어로 보완한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_TRAILING_PAGE_NO = re.compile(r"_\d+$")
_RCEPT_FROM_STEM = re.compile(r"^dart/.+?_\d{4}_(\d{14})_")


def sds_title(page_id: str) -> str:
    """SDS KoPub: 'public_pdf/prism/<파일명>_<쪽>' 등에서 파일명만 뽑는다."""
    without_page = _TRAILING_PAGE_NO.sub("", page_id)
    title = without_page.rsplit("/", 1)[-1]
    return title[:80]


def load_dart_manifest(manifest_path: Path) -> dict[str, str]:
    """rcept_no -> report_nm (예: '[기재정정]사업보고서 (2022.12)')."""
    out: dict[str, str] = {}
    if not manifest_path.exists():
        return out
    for line in manifest_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[rec["rcept_no"]] = rec["report_nm"]
    return out


def dart_prefix(page_id: str, manifest: dict[str, str]) -> str:
    """'[회사] 보고서명(사업연도)' 형태. 회사명은 파일명 첫 대괄호, 보고서명은 manifest의 report_nm."""
    m = _RCEPT_FROM_STEM.match(page_id)
    report_nm = manifest.get(m.group(1), "") if m else ""
    bracket = re.search(r"\[([^\]]+)\]", page_id)
    company = bracket.group(1) if bracket else ""
    parts = [p for p in (f"[{company}]" if company else "", report_nm) if p]
    return " ".join(parts)


def prefix_for(page_id: str, dart_manifest: dict[str, str]) -> str:
    if page_id.startswith("dart/"):
        p = dart_prefix(page_id, dart_manifest)
        return p or sds_title(page_id)
    return sds_title(page_id)
