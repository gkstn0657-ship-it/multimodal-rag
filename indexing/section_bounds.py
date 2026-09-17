"""절 경계·빈 페이지(no_content) 판정 (D-27, D-28).

D-26·D-27에서 정리한 대로, 시맨틱 청킹(텍스트 유사도 기반 경계)은 텍스트가 거의 없는
IMAGE 경로 페이지를 경계 판정에서 빠뜨리므로 이 코퍼스에는 쓰지 않는다. 대신 **문서
구조 신호**로 경계를 잡는다:

  - 반복 머리글의 절 제목 변화 ("제N장 …", "PART N", 로마숫자 장 표기, "N. 제목")
  - DART처럼 머리글이 없는 문서는 절 시작 페이지 첫 줄의 "N. 제목" 패턴
  - 목차 페이지(점선 리더 "….....42" 3회 이상)

같은 신호로 "내용 없는 페이지"(표지·구분지·목차)도 함께 판정한다. IMAGE 경로 페이지 중
- 표·차트·수치 등 실제 데이터 신호(<그림>, <표>, 퍼센트·금액·건수 등)가 없고
- 장 표지 신호(PART N, 제N장, 로마숫자 장 표기)가 있거나 페이지 텍스트 자체가 아주 짧으면
`no_content=True`로 표시한다. 이 페이지는 검색 후보에서 제외되지만(D-27 항목 1), 절
제목 추출 재료로는 계속 쓰인다.

이미 만들어진 벡터 저장소의 meta.jsonl(문서·페이지 텍스트, route)만으로 동작하고,
원본 PDF를 다시 읽지 않는다 — 인덱싱을 다시 하지 않고 저장소 위에 얹는 후처리 단계다.

결과는 <store_dir>/section_meta.jsonl 에 페이지 ID별로 한 줄씩 저장한다:
  {"id", "no_content", "section_title", "prev_id", "next_id"}
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

_PAGE_SUFFIX_RE = re.compile(r"^(.*)_(\d+)$")

_TOC_DOTLEADER_RE = re.compile(r"\.{5,}\s*\d+")

# 표·차트·수치 등 실제 내용이 있다는 신호. 이게 있으면 표지 신호와 무관하게 내용 있음으로 본다.
# 주의: 단위 뒤에 \b를 넣으면 "30명을"처럼 조사가 바로 붙는 한국어 문장에서 명사·조사가 둘 다
# \w로 인식돼 경계가 안 잡혀 매칭이 실패한다(실측: 치매실태조사 페이지가 오탐). 단위 뒤 \b는 쓰지 않는다.
_DATA_SIGNAL_RE = re.compile(
    r"<그림|<표|표\s*:|\d{1,3}(?:[,.]\d+)*\s*(?:%|원|건|명|㎥|억|만|점|개|톤|kg|km|억원|만원)"
)

# 마크다운 표(파이프 열이 2개 이상)가 있으면 데이터로 본다.
_TABLE_ROW_RE = re.compile(r"\|.*\|.*\|")

# 장 표지·구분지·목차형 페이지 신호.
_DIVIDER_SIGNAL_RE = re.compile(
    r"PART\s*\d+|^\s*제\s*\d+\s*[장부편]|^\s*[IVXLCM]{1,4}\s*[.\)]|"
    r"다른\s*내용이\s*없습니다|표지(?:입니다|로 보입니다|를 나타)|cover page",
    re.MULTILINE | re.IGNORECASE,
)

# "N. 소제목" 목록만 나열하는 절 소개 페이지 판정용(예: "1. 재정수지비율 | 2. 재정지표 | 3. 결산지표").
_NUMBERED_ITEM_RE = re.compile(r"^(?:#+\s*)?-?\s*\d+\.\s+\S")

# 절 경계로 볼 첫 줄 패턴 (머리글의 절 제목, DART의 번호 제목 등).
_SECTION_MARK_RE = re.compile(
    r"^(?:제\s*\d+\s*[장절편부]|[IVXLCM]{1,4}\s*[.\)]|\d{1,2}\.\s*[가-힣A-Za-z]|PART\s*\d+)"
)


def parse_doc_and_page(page_id: str) -> tuple[str, int] | None:
    """'dart/…_211' -> ('dart/…', 211). 끝이 페이지 번호가 아니면 None."""
    m = _PAGE_SUFFIX_RE.match(page_id)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def is_no_content(text: str, route: str) -> bool:
    """표지·구분지·목차처럼 답변 근거가 되지 못하는 페이지인지 판정한다.

    - 목차(점선 리더 3회 이상)는 경로와 무관하게 no_content.
    - 그 외에는 IMAGE 경로에서만 판정한다. TEXT/OCR 경로는 원문 분량이 이미
      route_min_text_chars(100자) 이상이라 대개 실제 내용이 있다고 본다.
    - 데이터 신호(표·그림·수치)가 있으면 표지 신호가 있어도 내용 있음으로 본다
      (실측: 지하수 조사 보고서의 <그림 …> 페이지들은 divider 문구 없이도 실제 수치를 담음).
    """
    stripped = text.strip()
    if not stripped:
        return True
    if len(_TOC_DOTLEADER_RE.findall(text)) >= 3:
        return True
    if route != "image":
        return False
    if _DATA_SIGNAL_RE.search(text):
        return False
    if _TABLE_ROW_RE.search(text):
        return False
    if _DIVIDER_SIGNAL_RE.search(text):
        return True
    # 절 소개 페이지: "N. 소제목" 목록만 있고(표·수치 없음) 줄 수가 적으면 표지·구분지로 본다.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    numbered = sum(1 for l in lines if _NUMBERED_ITEM_RE.match(l))
    if numbered >= 2 and len(lines) <= 20:
        return True
    return len(stripped) < 220


def detect_section_title(text: str) -> str | None:
    """페이지 첫 몇 줄에서 절 제목 패턴을 찾는다. 없으면 None(이전 절 제목 유지)."""
    for line in text.splitlines()[:4]:
        line = line.strip().lstrip("#").strip()
        if not line:
            continue
        if _SECTION_MARK_RE.match(line):
            return line[:60]
    return None


def build_section_map(records: list[dict]) -> dict[str, dict]:
    """meta.jsonl 레코드(id/document/metadata) 목록에서 페이지별 절 정보를 만든다.

    문서(doc_key) 안에서 페이지 번호 순으로 훑으며, 절 제목을 찾으면 그 뒤 페이지들에
    이어 붙이고(다음 절 제목이 나올 때까지), no_content와 앞뒤 페이지 ID를 함께 기록한다.
    """
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    skipped = 0
    for rec in records:
        parsed = parse_doc_and_page(rec["id"])
        if not parsed:
            skipped += 1
            continue
        doc_key, page_no = parsed
        groups[doc_key].append((page_no, rec))

    out: dict[str, dict] = {}
    for doc_key, pages in groups.items():
        pages.sort(key=lambda x: x[0])
        ids_in_order = [rec["id"] for _, rec in pages]
        current_title = ""
        for i, (_page_no, rec) in enumerate(pages):
            text = rec["document"]
            route = rec["metadata"].get("route", "")
            nc = is_no_content(text, route)
            title = detect_section_title(text)
            if title:
                current_title = title
            out[rec["id"]] = {
                "id": rec["id"],
                "no_content": nc,
                "section_title": current_title,
                "prev_id": ids_in_order[i - 1] if i > 0 else None,
                "next_id": ids_in_order[i + 1] if i + 1 < len(ids_in_order) else None,
            }
    if skipped:
        print(f"경고: 페이지 ID 패턴이 안 맞아 건너뛴 레코드 {skipped}건")
    return out


def build_for_store(store_dir: Path) -> dict:
    """store_dir/meta.jsonl을 읽어 section_meta.jsonl을 같은 디렉터리에 저장하고 통계를 돌려준다."""
    records = []
    with (store_dir / "meta.jsonl").open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    section_map = build_section_map(records)

    out_path = store_dir / "section_meta.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            info = section_map.get(rec["id"])
            if info is None:
                continue
            f.write(json.dumps(info, ensure_ascii=False) + "\n")

    n = len(records)
    n_no_content = sum(1 for v in section_map.values() if v["no_content"])
    n_image = sum(1 for r in records if r["metadata"].get("route") == "image")
    n_image_no_content = sum(
        1
        for r in records
        if r["metadata"].get("route") == "image" and section_map.get(r["id"], {}).get("no_content")
    )
    n_with_title = sum(1 for v in section_map.values() if v["section_title"])
    return {
        "store_dir": str(store_dir),
        "total_pages": n,
        "no_content_pages": n_no_content,
        "no_content_ratio": round(n_no_content / n, 4) if n else 0.0,
        "image_route_pages": n_image,
        "image_route_no_content": n_image_no_content,
        "image_route_no_content_ratio": round(n_image_no_content / n_image, 4) if n_image else 0.0,
        "pages_with_section_title": n_with_title,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("store_dirs", nargs="+")
    args = parser.parse_args()
    for d in args.store_dirs:
        stats = build_for_store(Path(d))
        print(json.dumps(stats, ensure_ascii=False, indent=2))
