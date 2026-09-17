"""PDF 파일 묶음을 페이지 단위 Page 스트림으로 바꾼다 (DART 2차 코퍼스용, D-16).

SDS KoPub은 페이지 이미지·텍스트·OCR이 파켓에 미리 들어 있었지만, DART 공시 첨부 PDF는 원본
파일이다. PyMuPDF로 페이지마다 텍스트 레이어를 뽑고 이미지를 렌더링해 `indexing.corpus.Page`와
같은 모양으로 내보낸다. 라우팅(`indexing.ingest.decide_route`)은 그대로 재사용한다.

- text: PDF 텍스트 레이어. 스캔 PDF는 비어 있거나 매우 짧다 → IMAGE 경로로 간다.
  D-31: 표가 검출되면(PyMuPDF find_tables) 원문 뒤에 마크다운 표를 이어 붙인다 — 2단 표 레이아웃이
  한 줄로 평평하게 풀리며 행·열 대응이 사라지는 문제(D-31의 "5성급" 오독과 같은 종류)를 막기 위함.
- ocr: 로컬 OCR 엔진이 없어 빈 문자열로 둔다. 따라서 텍스트가 빈 페이지는 OCR 경로 없이
  잉크 비율로 BLANK/IMAGE만 갈린다.
- image_bytes: 150 DPI 렌더링 PNG. VLM 입력은 1,280px로 다시 줄이므로 그 이상은 낭비다.
- id: "dart/<파일 stem>_<페이지번호>" — SDS와 같은 '파일_페이지' 꼴을 유지해 출처 표기·평가 코드가 그대로 돈다.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from indexing.corpus import Page
from indexing.table_extract import merge_text_and_tables, tables_markdown

RENDER_DPI = 150


def iter_pdf_pages(pdf_paths: list[Path], id_prefix: str = "dart", limit: int | None = None) -> Iterator[Page]:
    """여러 PDF의 페이지를 순서대로 Page로 내보낸다."""
    count = 0
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"PDF 열기 실패, 건너뜀: {pdf_path} ({exc})")
            continue
        stem = pdf_path.stem
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            text = page.get_text("text") or ""
            text = merge_text_and_tables(text, tables_markdown(page))
            pix = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
            image_bytes = pix.tobytes("png")
            yield Page(id=f"{id_prefix}/{stem}_{pno + 1}", text=text, ocr="", image_bytes=image_bytes)
            count += 1
            if limit is not None and count >= limit:
                doc.close()
                return
        doc.close()


def list_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def pdf_stats(pdf_paths: list[Path]) -> dict:
    """페이지 수와 텍스트 레이어 유무 분포. 코퍼스 규모와 스캔 비율을 먼저 보기 위한 용도."""
    total = 0
    sparse = 0
    per_file = []
    for p in pdf_paths:
        try:
            doc = fitz.open(p)
        except Exception:  # noqa: BLE001
            continue
        n = doc.page_count
        s = sum(1 for i in range(n) if len((doc.load_page(i).get_text("text") or "").strip()) < 100)
        per_file.append({"file": p.name, "pages": n, "sparse_text_pages": s})
        total += n
        sparse += s
        doc.close()
    return {"files": len(per_file), "pages": total, "sparse_text_pages": sparse, "per_file": per_file}
