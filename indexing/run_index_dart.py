"""DART 2차 코퍼스 인덱싱 오케스트레이터 (D-16).

run_index.py와 같은 단계(라우팅 -> 원본 이미지 저장 -> 캡셔닝 -> 언로드 -> 임베딩)를 따르되,
페이지 소스가 SDS KoPub 파켓이 아니라 indexing.pdf_corpus.iter_pdf_pages(원본 PDF)이고,
저장소를 SDS 인덱스와 분리한다 (D-16: 다른 코퍼스이므로 같은 벡터 스토어에 섞지 않는다).

사용법:
    python -m indexing.run_index_dart --limit 200   # 스모크 테스트
    python -m indexing.run_index_dart                # 전체
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from config import settings
from indexing.caption import CaptionCache, caption_page
from indexing.embed_store import build_chunks, embed_and_store, save_page_image
from indexing.ingest import Route, route_pages
from indexing.pdf_corpus import iter_pdf_pages, list_pdfs
from servers.vlm_client import VLMClient

DART_PDF_DIR = Path("data/dart/pdfs")
DART_VECTOR_STORE_DIR = Path("data/vector_store_dart")
DART_CAPTION_CACHE = Path("data/captions_dart.jsonl")
DART_REPORT_PATH = Path("data/dart/index_report.json")


def unload_vlm() -> None:
    try:
        subprocess.run(["ollama", "stop", settings.vlm_model], check=False, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def run(limit: int | None) -> None:
    t_start = time.time()

    pdfs = list_pdfs(DART_PDF_DIR)
    print(f"[0/3] PDF {len(pdfs)}개 발견 ({DART_PDF_DIR})", flush=True)

    print("[1/3] 페이지 라우팅 중...", flush=True)
    routed, stats = route_pages(limit=limit, pages=iter_pdf_pages(pdfs, limit=limit))
    print(f"  라우팅 통계: {stats.as_dict()}", flush=True)

    image_backed = [rp for rp in routed if rp.route in (Route.IMAGE, Route.OCR)]
    for rp in tqdm(image_backed, desc="saving page images", unit="page"):
        save_page_image(rp.page.id, rp.page.to_pil())

    cache = CaptionCache(path=DART_CAPTION_CACHE)
    image_pages = [rp for rp in routed if rp.route == Route.IMAGE]
    todo = [rp for rp in image_pages if rp.page.id not in cache]
    print(
        f"[2/3] 캡셔닝: 대상 {len(image_pages)}장, 캐시 {len(image_pages) - len(todo)}장, "
        f"남은 {len(todo)}장, 병렬 {settings.caption_parallel}",
        flush=True,
    )

    caption_times: list[float] = []
    failures: list[dict] = []
    if todo:
        client = VLMClient()
        if not client.ping():
            raise RuntimeError("VLM 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")

        from indexing.caption import CAPTION_PROMPT_V3

        t_cap = time.time()
        with ThreadPoolExecutor(max_workers=settings.caption_parallel) as ex:
            futures = {
                ex.submit(caption_page, client, rp.page.id, rp.page.to_pil(), 1, CAPTION_PROMPT_V3): rp
                for rp in todo
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="captioning", unit="page"):
                result = fut.result()
                if result.error:
                    failures.append({"page_id": result.page_id, "error": result.error})
                else:
                    cache.put(result)
                    caption_times.append(result.elapsed_sec)
        wall = time.time() - t_cap
        print(
            f"  캡셔닝 완료: 성공 {len(caption_times)}장, 실패 {len(failures)}장, "
            f"벽시계 {wall / 60:.1f}분",
            flush=True,
        )
        for rp in image_pages:
            rp.page.image_bytes = None

    unload_vlm()

    if failures:
        fail_path = Path("data/dart/caption_failures.json")
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  실패 목록 저장: {fail_path}", flush=True)

    print("[3/3] 청크 생성 및 임베딩(GPU) 중...", flush=True)
    for rp in routed:
        if rp.route == Route.OCR:
            rp.page.image_bytes = None
    chunks = build_chunks(routed, cache.as_dict())

    # D-16: SDS 인덱스(settings.vector_store_dir)와 분리된 디렉터리에 저장한다.
    # embed_and_store는 settings.vector_store_dir을 직접 참조하므로 호출 구간만 임시로 바꾼다.
    original_dir = settings.vector_store_dir
    settings.vector_store_dir = DART_VECTOR_STORE_DIR
    try:
        stored = embed_and_store(chunks, device=settings.embed_device_indexing)
    finally:
        settings.vector_store_dir = original_dir
    print(f"  저장 완료: {stored}건 -> {DART_VECTOR_STORE_DIR}", flush=True)

    report = {
        "limit": limit,
        "pdf_files": len(pdfs),
        "routing": stats.as_dict(),
        "captioned_total_in_cache": len(cache),
        "captioned_this_run": len(caption_times),
        "caption_failures": len(failures),
        "stored_chunks": stored,
        "total_wall_sec": round(time.time() - t_start, 1),
    }
    DART_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"리포트 저장: {DART_REPORT_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 페이지 수 제한 (스모크 테스트용)")
    args = parser.parse_args()
    run(limit=args.limit)
