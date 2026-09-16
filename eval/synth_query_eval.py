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
from eval.retrieval_eval import K_VALUES, run_search_topk

QUERIES = Path("eval/synth_queries_image_pages.jsonl")
STORES = {
    "ours_caption": Path("data/vector_store"),
    "baseline_ocr": Path("data/vector_store_baseline_ocr"),
}


def load_queries() -> list[dict]:
    rows = [json.loads(l) for l in QUERIES.open(encoding="utf-8")]
    # 질문이 비었거나 페이지를 가리키는 표현이 남은 것은 제외
    bad = ("이 페이지", "이 문서", "위 표", "위 그림")
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
    n = len(ranks)
    out = {f"recall@{kk}": sum(1 for r in ranks if r and r <= kk) / n for kk in K_VALUES}
    out["mrr"] = sum(1.0 / r for r in ranks if r) / n
    out["n"] = n
    return out, ranks


def main() -> None:
    queries = load_queries()
    print(f"합성 질의 {len(queries)}개 (필터 후)")
    k = max(K_VALUES)
    report: dict = {"n_queries": len(queries), "stores": {}}
    all_ranks: dict[str, list] = {}
    for name, path in STORES.items():
        metrics, ranks = evaluate_store(name, path, queries, k)
        report["stores"][name] = metrics
        all_ranks[name] = ranks
        print(name, json.dumps(metrics, ensure_ascii=False))

    # 문항별 승패: 캡션이 더 높은 순위(작은 수)면 ours 승
    wins = defaultdict(int)
    for a, b in zip(all_ranks["ours_caption"], all_ranks["baseline_ocr"]):
        ra = a or 10**6
        rb = b or 10**6
        if ra < rb:
            wins["ours_better"] += 1
        elif rb < ra:
            wins["baseline_better"] += 1
        else:
            wins["tie"] += 1
    report["per_query_wins"] = dict(wins)
    report["per_query_ranks"] = [
        {"page_id": q["page_id"], "query": q["query"], "ours": a, "baseline": b}
        for q, a, b in zip(queries, all_ranks["ours_caption"], all_ranks["baseline_ocr"])
    ]
    Path("eval/synth_query_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("승패:", dict(wins))
    print("저장: eval/synth_query_report.json")


if __name__ == "__main__":
    main()
