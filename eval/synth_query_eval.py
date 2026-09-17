"""IMAGE 경로 페이지용 합성 질의로 캡션 저장소 vs OCR 베이스라인 검색 성능 비교 (D-08 후속).

SDS KoPub 라벨 600문항은 정답이 IMAGE 경로인 문항이 1개뿐이라 캡셔닝 효과를 검증할 수 없었다.
그래서 IMAGE 경로 997장 중 200장을 뽑아 VLM이 페이지 이미지를 보고 질문 하나씩 쓰게 했다
(`eval/synth_queries_image_pages.jsonl`, 생성 프롬프트는 생성 스크립트 참고). 정답은 그 페이지다.

편향 주의: 질문을 쓴 모델(Qwen2.5-VL)이 캡션을 쓴 모델과 같다. 이미지에서 직접 생성했고 캡션
텍스트를 보여주지 않았지만, 같은 모델이 같은 페이지에서 뽑는 어휘가 겹칠 수 있어 캡션 저장소에
유리한 쪽으로 편향될 수 있다. 결과는 "상한 추정"으로 해석한다.

사용법:
    python -m eval.synth_query_eval
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from config import settings
import indexing.embed_store as es
from eval.retrieval_eval import K_VALUES, metrics_from_ranks, run_search_topk

QUERIES = Path("eval/synth_queries_image_pages.jsonl")
# 비교 대상. 변형 저장소는 eval/build_variant_store.py로 만든다.
STORES = {
    "v3_caption_ocr": Path("data/vector_store"),  # D-19: 운영 저장소 (v3 캡션+OCR)
    "v2_caption": Path("data/vector_store_variant_v2_caption"),  # D-07~D-18 운영본(캡션 단독)
    "v2_caption_ocr": Path("data/vector_store_variant_v2_caption_ocr"),
    "v1_caption": Path("data/vector_store_variant_v1_caption"),
    "v1_caption_ocr": Path("data/vector_store_variant_v1_caption_ocr"),
    "baseline_ocr": Path("data/vector_store_baseline_ocr"),
}
OURS = "v3_caption_ocr"
BASELINE = "baseline_ocr"
REPORT = Path("eval/synth_query_report_v3.json")


def load_queries() -> list[dict]:
    rows = [json.loads(l) for l in QUERIES.open(encoding="utf-8")]
    # 질문이 비었거나 페이지를 가리키는 지시어가 남은 것은 제외(검색 질의로 성립하지 않음).
    # 2026-09-17 보강: 판정 표본 추출 시 "이 도서/이 그림/N페이지의" 같은 변형이 걸러지지 않은 것을 발견.
    bad = (
        "이 페이지", "이 문서", "위 표", "위 그림", "이 도서", "이 그림", "이 주제", "이 표", "페이지의",
        "이 자료", "이 사진", "본 자료", "이 연구진", "이 보고서", "본 보고서",
    )
    return [r for r in rows if r["query"].strip() and not any(b in r["query"] for b in bad)]


def evaluate_store(name: str, path: Path, queries: list[dict], k: int) -> dict:
    settings.vector_store_dir = path
    es.get_store(force_reload=True)
    ranks: list[int | None] = []
    for q in queries:
        ranked = run_search_topk(q["query"], k)
        try:
            ranks.append(ranked.index(q["page_id"]) + 1)
        except ValueError:
            ranks.append(None)
    return metrics_from_ranks(ranks), ranks


def main() -> None:
    queries = load_queries()
    print(f"합성 질의 {len(queries)}개 (필터 후)")
    k = max(K_VALUES)
    report: dict = {"n_queries": len(queries), "stores": {}}
    all_ranks: dict[str, list] = {}
    for name, path in STORES.items():
        if not (path / "vectors.npy").exists():
            print(f"{name}: 저장소 없음, 건너뜀 ({path})")
            continue
        metrics, ranks = evaluate_store(name, path, queries, k)
        report["stores"][name] = metrics
        all_ranks[name] = ranks
        print(name, json.dumps(metrics, ensure_ascii=False))

    # 문항별 승패: OURS가 더 높은 순위(작은 수)면 ours 승
    wins = defaultdict(int)
    for a, b in zip(all_ranks[OURS], all_ranks[BASELINE]):
        ra = a or 10**6
        rb = b or 10**6
        if ra < rb:
            wins["ours_better"] += 1
        elif rb < ra:
            wins["baseline_better"] += 1
        else:
            wins["tie"] += 1
    report["compared"] = {"ours": OURS, "baseline": BASELINE}
    report["per_query_wins"] = dict(wins)
    report["per_query_ranks"] = [
        {"page_id": q["page_id"], "query": q["query"], **{name: r[i] for name, r in all_ranks.items()}}
        for i, q in enumerate(queries)
    ]
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("승패:", dict(wins))
    print(f"저장: {REPORT}")


if __name__ == "__main__":
    main()
