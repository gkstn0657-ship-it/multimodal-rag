"""인덱싱 오케스트레이터.

D-02 순서 준수: VLM 캡셔닝을 먼저 끝내고 VLM을 언로드한 뒤,
GPU 임베딩을 수행한다. 두 단계가 동시에 GPU를 점유하지 않는다.

사용법:
    python -m indexing.run_index --limit 500          # 스모크 테스트
    python -m indexing.run_index                       # 전체 인덱싱
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from tqdm import tqdm

from config import settings
from indexing.caption import caption_page
from indexing.embed_store import build_chunks, embed_and_store, save_page_image
from indexing.ingest import Route, route_pages
from servers.vlm_client import VLMClient


def unload_vlm() -> None:
    """Ollama에 VLM 언로드를 요청한다 (GPU 메모리 회수)."""
    try:
        subprocess.run(
            ["ollama", "stop", settings.vlm_model],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def run(limit: int | None) -> None:
    print("[1/3] 페이지 라우팅 중...")
    routed, stats = route_pages(limit=limit)
    print(f"  라우팅 통계: {stats.as_dict()}")

    print("[2/3] 이미지 경로 페이지 캡셔닝 중 (VLM GPU 사용)...")
    client = VLMClient()
    if not client.ping():
        raise RuntimeError("VLM 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")

    captions: dict[str, str] = {}
    caption_times: list[float] = []
    image_pages = [rp for rp in routed if rp.route == Route.IMAGE]

    for rp in tqdm(image_pages, desc="captioning"):
        page = rp.page
        image = page.to_pil()
        save_page_image(page.id, image)
        result = caption_page(client, page.id, image)
        captions[page.id] = result.caption
        caption_times.append(result.elapsed_sec)

    unload_vlm()
    print(f"  캡셔닝 완료: {len(captions)}건, VLM 언로드 요청함.")

    if caption_times:
        avg = sum(caption_times) / len(caption_times)
        p95 = sorted(caption_times)[int(len(caption_times) * 0.95)]
        print(f"  캡셔닝 소요시간: 평균 {avg:.1f}초 / p95 {p95:.1f}초")

    print("[3/3] 청크 생성 및 임베딩(GPU) 중...")
    chunks = build_chunks(routed, captions)
    stored = embed_and_store(chunks, device=settings.embed_device_indexing)
    print(f"  저장 완료: {stored}건")

    report = {
        "routing": stats.as_dict(),
        "captioned": len(captions),
        "stored_chunks": stored,
        "caption_time_avg_sec": (sum(caption_times) / len(caption_times)) if caption_times else None,
        "caption_time_p95_sec": (sorted(caption_times)[int(len(caption_times) * 0.95)] if caption_times else None),
    }
    report_path = settings.data_dir / "index_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"리포트 저장: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 페이지 수 제한 (스모크 테스트용)")
    args = parser.parse_args()
    run(limit=args.limit)
