"""OpenDART에서 상장사 정기보고서 첨부 PDF를 내려받아 2차 코퍼스를 만든다 (D-16).

목적: 실제 기업 문서(표 밀도가 높고, 같은 회사의 여러 연도 보고서가 서로 방해 문서가 되며,
오래된 첨부에는 스캔본이 섞인)로 라우팅·검색을 시험한다. `docs/dataset.md` 2차 코퍼스 항목 참고.

필요: OpenDART 인증키 — `.env`에 `DART_API_KEY=...` (git 제외). https://opendart.fss.or.kr 에서 발급.
일일 호출 한도가 있으므로(2026-05 기준 20,000건으로 알려짐) 회사·연도 수를 작게 시작한다.

재현: 내려받은 PDF는 git에 넣지 않고 `data/dart/manifest.jsonl`(회사·접수번호·파일명·URL·sha256·수집일)만
남긴다. 이 파일과 같은 스크립트로 누구든 같은 코퍼스를 다시 만들 수 있다.

사용법:
    python -m indexing.dart_fetch --corps 삼성전자 현대자동차 --years 2022 2023 2024
    python -m indexing.dart_fetch --corps 삼성전자 --years 2024 --dry-run   # 목록만
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv

DATA_DIR = Path("data/dart")
PDF_DIR = DATA_DIR / "pdfs"
MANIFEST = DATA_DIR / "manifest.jsonl"

# D-16 추가: OpenDartReader.download()는 pdf.do를 Referer 없이 바로 호출해 0바이트를 받는다.
# 실제 사이트는 팝업(main.do)을 먼저 열어 PDFJSESSIONID 쿠키를 받고, 그 팝업 페이지를 Referer로
# 달아 pdf.do를 호출한다. Referer 검사가 핵심이며 쿠키만으로는 통과하지 않는다(직접 확인).
_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0"})


def fetch_pdf_bytes(rcept_no: str, dcm_no: str) -> bytes:
    main_url = f"https://dart.fss.or.kr/pdf/download/main.do?rcp_no={rcept_no}&dcm_no={dcm_no}"
    _session.get(main_url, timeout=15)
    pdf_url = f"https://dart.fss.or.kr/pdf/download/pdf.do?rcp_no={rcept_no}&dcm_no={dcm_no}"
    resp = _session.get(pdf_url, headers={"Referer": main_url}, timeout=60)
    resp.raise_for_status()
    if resp.content[:4] != b"%PDF":
        raise ValueError(f"PDF 아님 (앞 8바이트: {resp.content[:8]!r})")
    return resp.content

# 정기공시 중 사업보고서만. 분기·반기보고서는 표가 겹쳐 방해 문서로는 좋지만 양이 급증하므로 첫 판에서는 제외.
REPORT_KEYWORD = "사업보고서"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    return {json.loads(l)["url"]: json.loads(l) for l in MANIFEST.open(encoding="utf-8") if l.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corps", nargs="+", required=True, help="회사명 (예: 삼성전자)")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="사업연도 (보고서 제출은 다음 해 3월)")
    parser.add_argument("--dry-run", action="store_true", help="다운로드 없이 첨부 목록만 출력")
    parser.add_argument("--max-files-per-report", type=int, default=3)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        raise SystemExit("DART_API_KEY 가 없습니다. .env 에 넣어 주세요 (https://opendart.fss.or.kr 에서 발급).")

    import OpenDartReader  # noqa: WPS433 — 키가 있을 때만 임포트 (패키지 임포트 자체가 클래스를 반환한다)

    dart = OpenDartReader(api_key)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    added = 0

    for corp in args.corps:
        for year in args.years:
            # 사업보고서는 다음 해 3~4월에 제출된다.
            start, end = f"{year + 1}-01-01", f"{year + 1}-06-30"
            try:
                reports = dart.list(corp, start=start, end=end, kind="A")  # A: 정기공시
            except Exception as exc:  # noqa: BLE001
                print(f"[{corp} {year}] 목록 조회 실패: {exc}")
                continue
            if reports is None or len(reports) == 0:
                print(f"[{corp} {year}] 정기공시 없음")
                continue
            reports = reports[reports["report_nm"].str.contains(REPORT_KEYWORD, na=False)]
            for _, row in reports.iterrows():
                rcept_no = row["rcept_no"]
                try:
                    files = dart.attach_files(rcept_no)  # {파일명: URL}
                except Exception as exc:  # noqa: BLE001
                    print(f"[{corp} {year}] 첨부 조회 실패 {rcept_no}: {exc}")
                    continue
                pdfs = [(n, u) for n, u in files.items() if n.lower().endswith(".pdf")][: args.max_files_per_report]
                print(f"[{corp} {year}] {row['report_nm']} ({rcept_no}) PDF 첨부 {len(pdfs)}건")
                for name, url in pdfs:
                    if url in manifest:
                        continue
                    safe = f"{corp}_{year}_{rcept_no}_{name}".replace("/", "_").replace(" ", "_")
                    dest = PDF_DIR / safe
                    if args.dry_run:
                        print("   -", name)
                        continue
                    parsed = parse_qs(urlparse(url).query)
                    dcm_no = parsed.get("dcm_no", [None])[0]
                    try:
                        data = fetch_pdf_bytes(rcept_no, dcm_no)
                        dest.write_bytes(data)
                    except Exception as exc:  # noqa: BLE001
                        print(f"   다운로드 실패 {name}: {exc}")
                        continue
                    rec = {
                        "corp": corp, "year": year, "rcept_no": rcept_no, "report_nm": row["report_nm"],
                        "file": dest.name, "url": url, "sha256": sha256_of(dest), "bytes": dest.stat().st_size,
                        "fetched": date.today().isoformat(),
                    }
                    with MANIFEST.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    manifest[url] = rec
                    added += 1
                    time.sleep(0.5)  # 호출 한도 배려
    print(f"완료: 새 파일 {added}건, manifest 총 {len(manifest)}건 → {MANIFEST}")


if __name__ == "__main__":
    main()
