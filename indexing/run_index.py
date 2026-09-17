"""인덱싱 오케스트레이터.

D-02 순서 준수: VLM 캡셔닝을 먼저 끝내고 VLM을 언로드한 뒤 GPU 임베딩을 수행한다.
D-04: 캡셔닝은 병렬(settings.caption_parallel)로, 결과는 JSONL 캐시에 즉시 기록해
      중단 후 재실행 시 이어서 진행한다. 개별 페이지 실패는 전체를 멈추지 않는다.

사용법:
    python -m indexing.run_index --limit 500     # 스모크 테스트
    python -m indexing.run_index                  # 전체 인덱싱 (재실행하면 캐시된 캡션은 건너뜀)
    python -m indexing.run_index --skip-embed     # 캡셔닝만
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from config import settings
from indexing.caption import CaptionCache, caption_page
from indexing.embed_store import build_chunks, embed_and_store, save_page_image
from indexing.ingest import Route, route_pages
from servers.vlm_client import VLMClient


def unload_vlm() -> None:
    """Ollama에 VLM 언로드를 요청한다 (GPU 메모리 회수)."""
    try:
        subprocess.run(["ollama", "stop", settings.vlm_model], check=False, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


def run(limit: int | None, skip_embed: bool = False) -> None:
    t_start = time.time()

    print("[1/3] 페이지 라우팅 중...", flush=True)
    routed, stats = route_pages(limit=limit)
    print(f"  라우팅 통계: {stats.as_dict()}", flush=True)

    # --- 원본 이미지 저장 (IMAGE·OCR 경로) ---
    image_backed = [rp for rp in routed if rp.route in (Route.IMAGE, Route.OCR)]
    for rp in tqdm(image_backed, desc="saving page images", unit="page"):
        save_page_image(rp.page.id, rp.page.to_pil())

    # --- 캡셔닝 ---
    cache = CaptionCache()
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

        from indexing.caption import CAPTION_PROMPT_V3  # D-15/D-19: 전사 우선 프롬프트가 운영 기본

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
            f"벽시계 {wall / 60:.1f}분 (장당 {wall / max(len(todo), 1):.2f}초)",
            flush=True,
        )
        if caption_times:
            print(
                f"  장당 지연: 평균 {sum(caption_times) / len(caption_times):.1f}초 / "
                f"p95 {_percentile(caption_times, 0.95):.1f}초",
                flush=True,
            )
        # 이미지 페이지 객체의 원본 바이트는 이제 필요 없다
        for rp in image_pages:
            rp.page.image_bytes = None

    unload_vlm()

    if failures:
        fail_path = settings.data_dir / "caption_failures.json"
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  실패 목록 저장: {fail_path} (이 페이지들은 OCR/텍스트로 대체 인덱싱됨)", flush=True)

    # --- 임베딩 ---
    stored = 0
    if not skip_embed:
        print("[3/3] 청크 생성 및 임베딩(GPU) 중...", flush=True)
        for rp in routed:
            if rp.route == Route.OCR:
                rp.page.image_bytes = None
        chunks = build_chunks(routed, cache.as_dict())
        # D-06: 벡터는 메모리에 누적됐다가 embed_and_store 끝에서 한 번만 디스크에 쓰인다.
        stored = embed_and_store(chunks, device=settings.embed_device_indexing)
        print(f"  저장 완료: {stored}건", flush=True)
    else:
        print("[3/3] 임베딩 생략 (--skip-embed)", flush=True)

    report = {
        "limit": limit,
        "routing": stats.as_dict(),
        "captioned_total_in_cache": len(cache),
        "captioned_this_run": len(caption_times),
        "caption_failures": len(failures),
        "caption_time_avg_sec": (sum(caption_times) / len(caption_times)) if caption_times else None,
        "caption_time_p95_sec": _percentile(caption_times, 0.95),
        "caption_parallel": settings.caption_parallel,
        "stored_chunks": stored,
        "total_wall_sec": round(time.time() - t_start, 1),
    }
    report_path = settings.data_dir / "index_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"리포트 저장: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 페이지 수 제한 (스모크 테스트용)")
    parser.add_argument("--skip-embed", action="store_true", help="캡셔닝만 수행하고 임베딩은 생략")
    args = parser.parse_args()
    run(limit=args.limit, skip_embed=args.skip_embed)
