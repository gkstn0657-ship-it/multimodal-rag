"""검색 판정 평가용 풀 생성 (D-18, eval/retrieval_judge_rubric.md).

고정 표본(eval/judge_sample_retrieval.json)의 질의마다 비교 대상 시스템별로 실제 파이프라인
(임베딩 top-10 → 리랭커 → top-5)을 돌리고, 나온 페이지를 합쳐 중복을 없앤 풀을 만든다.
판정자는 풀의 페이지를 시스템·순위를 모른 채 한 번씩만 등급 매긴다(pooling).

출력:
  eval/judge_retrieval_pool.json   — 질의별 시스템 순위 + 풀(페이지 텍스트 앞부분·이미지 경로)
  eval/judge_retrieval_grades.json — 등급 기입용 빈 틀 {sample_id: {page_id: null}} (이미 있으면 유지)

사용법:
    python -m eval.judge_retrieval_pool
"""

from __future__ import annotations

import json
from pathlib import Path

from config import settings
import indexing.embed_store as es
from servers.retrieve import retrieve

SAMPLE = Path("eval/judge_sample_retrieval.json")
POOL = Path("eval/judge_retrieval_pool.json")
GRADES = Path("eval/judge_retrieval_grades.json")
SNIPPET_CHARS = 1500

SYSTEMS = {
    "sds": {
        "v3_caption_ocr": Path("data/vector_store"),  # D-19: 운영 저장소
        "v2_caption": Path("data/vector_store_variant_v2_caption"),
        "baseline_ocr": Path("data/vector_store_baseline_ocr"),
    },
    # D-20: dart_dense(하이브리드 끔) vs dart_hybrid(BM25 RRF 병합)로 회사·연도 혼동 개선을 확인한다.
    "dart": {"dart_dense": Path("data/vector_store_dart"), "dart_hybrid": Path("data/vector_store_dart")},
}
HYBRID_OVERRIDE = {"dart_dense": False, "dart_hybrid": True}


def run_system(store: Path, query: str) -> list:
    settings.vector_store_dir = store
    es.get_store(force_reload=True)
    return retrieve(query)


def main() -> None:
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]
    grades = json.loads(GRADES.read_text(encoding="utf-8")) if GRADES.exists() else {}
    out_items = []

    # 저장소 로드를 줄이기 위해 시스템 순서를 바깥 루프로 둔다.
    per_query_ranked: dict[str, dict[str, list]] = {s["sample_id"]: {} for s in sample}
    for corpus, systems in SYSTEMS.items():
        items = [s for s in sample if s["corpus"] == corpus]
        for name, store in systems.items():
            if not (store / "vectors.npy").exists():
                print(f"{name}: 저장소 없음, 건너뜀 ({store})")
                continue
            settings.vector_store_dir = store
            settings.hybrid_enabled = HYBRID_OVERRIDE.get(name, settings.hybrid_enabled)
            es.get_store(force_reload=True)
            for s in items:
                per_query_ranked[s["sample_id"]][name] = retrieve(s["query"])
            print(f"{corpus}/{name}: {len(items)}문항 완료 (hybrid={settings.hybrid_enabled})", flush=True)

    for s in sample:
        ranked = per_query_ranked[s["sample_id"]]
        pool: dict[str, dict] = {}
        for name, chunks in ranked.items():
            for rank, c in enumerate(chunks, 1):
                entry = pool.setdefault(
                    c.page_id,
                    {"page_id": c.page_id, "route": c.route, "image_path": c.image_path,
                     "text": c.text[:SNIPPET_CHARS], "is_gold": c.page_id == s.get("gold_page_id"), "ranks": {}},
                )
                entry["ranks"][name] = rank
        out_items.append({
            **{k: s[k] for k in ("sample_id", "corpus", "type", "query", "gold_page_id", "expect_answer")},
            "systems": {name: [c.page_id for c in chunks] for name, chunks in ranked.items()},
            "pool": list(pool.values()),
        })
        grades.setdefault(s["sample_id"], {})
        for pid in pool:
            grades[s["sample_id"]].setdefault(pid, None)

    POOL.write_text(json.dumps({"items": out_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    GRADES.write_text(json.dumps(grades, ensure_ascii=False, indent=2), encoding="utf-8")
    n_pages = sum(len(v) for v in grades.values())
    print(f"풀 저장: {POOL} ({len(out_items)}문항, 판정 대상 페이지 {n_pages}건) / 등급 틀: {GRADES}")


if __name__ == "__main__":
    main()
