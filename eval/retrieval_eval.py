"""검색 성능 평가: SDS KoPub VDR 라벨 600쌍 기준.

judge가 필요 없는 자동 평가다. QA 각 행의 `id` 필드가 곧 정답 페이지 ID다
(row['ground_truth'][0]이 코퍼스 파켓의 몇 번째 행인지를 가리키는 인덱스이고,
그 행의 id 컬럼이 row['id']와 정확히 일치함을 확인했다 — 즉 `id` 자체를
정답으로 바로 쓸 수 있다).

지표:
  - Recall@k (k=1,3,5,10): top-k 안에 정답 페이지가 있으면 1, 아니면 0
  - MRR: 정답이 나온 순위의 역수 평균 (없으면 0)
  - 도메인/유형(text·visual·cross)별로 나눠서도 집계한다 — 이미지 캡셔닝이
    실제로 visual/cross 유형에서 얼마나 도움이 되는지 보기 위함이다.

사전등록 원칙: 이 스크립트를 실행하기 전에 지표와 k값을 정한다.
실행 후 결과를 보고 지표를 바꾸지 않는다.

사용법:
    python -m eval.retrieval_eval                 # 전체 600문항
    python -m eval.retrieval_eval --limit 50       # 스모크 테스트
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import pyarrow.parquet as pq

from config import settings
from servers.retrieve import search

K_VALUES = (1, 3, 5, 10)


@dataclass
class QueryResult:
    qa_id: str
    query: str
    type: str
    domain: str
    ground_truth: str
    ranked_ids: list[str]

    def rank_of_answer(self) -> int | None:
        """정답이 몇 번째(1-based)에 있는지. 없으면 None."""
        try:
            return self.ranked_ids.index(self.ground_truth) + 1
        except ValueError:
            return None

    def recall_at(self, k: int) -> int:
        rank = self.rank_of_answer()
        return int(rank is not None and rank <= k)

    def reciprocal_rank(self) -> float:
        rank = self.rank_of_answer()
        return 1.0 / rank if rank else 0.0


def load_qa(limit: int | None = None) -> list[dict]:
    rows = pq.read_table(settings.qa_parquet).to_pylist()
    return rows[:limit] if limit else rows


def run_search_topk(query: str, k: int) -> list[str]:
    """재랭킹 이전 단계(순수 임베딩 검색)에서 top-k 페이지 ID를 뽑는다.

    settings.top_k_candidates(기본 10)가 평가에 필요한 k보다 작으면 임시로 늘린다.
    """
    orig_k = settings.top_k_candidates
    try:
        settings.top_k_candidates = max(k, orig_k)
        candidates = search(query)
        return [c.page_id for c in candidates][:k]
    finally:
        settings.top_k_candidates = orig_k


def evaluate(limit: int | None = None, k: int = max(K_VALUES)) -> tuple[list[QueryResult], dict]:
    qa_rows = load_qa(limit=limit)
    results: list[QueryResult] = []

    t0 = time.time()
    for i, row in enumerate(qa_rows):
        ranked = run_search_topk(row["query"], k)
        results.append(
            QueryResult(
                qa_id=row["id"],
                query=row["query"],
                type=row["type"],
                domain=row["domain"],
                ground_truth=row["id"],  # 검증됨: id 필드가 곧 정답 페이지
                ranked_ids=ranked,
            )
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(qa_rows)} 처리, 경과 {time.time() - t0:.0f}초", flush=True)

    summary = summarize(results)
    return results, summary


def ndcg_single(rank: int | None, k: int) -> float:
    """정답이 1개일 때의 nDCG@k. IDCG=1이므로 정답이 k위 안이면 1/log2(rank+1), 아니면 0."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def metrics_from_ranks(ranks: list[int | None]) -> dict:
    """정답 순위 목록(없으면 None)에서 Recall@k·nDCG@k·MRR을 계산한다. 두 평가 스크립트가 공유한다."""
    n = len(ranks)
    if n == 0:
        return {}
    metrics = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in K_VALUES}
    metrics.update({f"ndcg@{k}": sum(ndcg_single(r, k) for r in ranks) / n for k in K_VALUES})
    metrics["mrr"] = sum(1.0 / r for r in ranks if r) / n
    metrics["n"] = n
    return metrics


def _macro_metrics(results: list[QueryResult]) -> dict:
    return metrics_from_ranks([r.rank_of_answer() for r in results])


def summarize(results: list[QueryResult]) -> dict:
    by_type: dict[str, list[QueryResult]] = defaultdict(list)
    by_domain: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_type[r.type].append(r)
        by_domain[r.domain].append(r)

    return {
        "overall": _macro_metrics(results),
        "by_type": {t: _macro_metrics(rs) for t, rs in sorted(by_type.items())},
        "by_domain": {d: _macro_metrics(rs) for d, rs in sorted(by_domain.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="평가 문항 수 제한 (스모크 테스트용)")
    parser.add_argument("--store", default=None, help="평가할 벡터 저장소 디렉터리 (기본: settings.vector_store_dir)")
    parser.add_argument("--tag", default="", help="출력 파일 접미어 (예: v3 → report_v3.json)")
    args = parser.parse_args()

    if args.store:
        from pathlib import Path

        import indexing.embed_store as es

        settings.vector_store_dir = Path(args.store)
        es.get_store(force_reload=True)

    print(f"검색 평가 시작 (limit={args.limit}, store={settings.vector_store_dir})...")
    results, summary = evaluate(limit=args.limit)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = settings.project_root / "eval" / f"report{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = settings.project_root / "eval" / f"report_detail{suffix}.json"

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path.write_text(
        json.dumps(
            [
                {
                    "id": r.qa_id,
                    "type": r.type,
                    "domain": r.domain,
                    "rank": r.rank_of_answer(),
                    "ranked_ids": r.ranked_ids,
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n저장: {out_path}, {detail_path}")
