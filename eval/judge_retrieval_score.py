"""판정 등급 → 시스템별 지표 (D-18, eval/retrieval_judge_rubric.md).

입력: eval/judge_retrieval_pool.json(시스템별 top-5), eval/judge_retrieval_grades.json(페이지 등급 0/1/2).
지표(문항 단위 평균):
  - ndcg@5: 이득 = 등급, IDCG는 그 문항 풀의 등급을 내림차순으로 정렬한 이상적 top-5.
            풀에 등급>0이 하나도 없는 문항(답 없는 문항 등)은 nDCG 평균에서 제외한다.
  - recall@5: top-5에 등급 2가 하나라도 있으면 1.
  - irrelevant@5: top-5 중 등급 0의 비율.
  - gold_in_top5: 자동 정답(gold_page_id)이 top-5에 있는 비율 — 판정과 자동 지표의 교차 확인용.
답 없는 문항(expect_answer=False)은 따로 집계한다: top-5에 등급>0이 하나라도 있으면 "오탐".

사용법:
    python -m eval.judge_retrieval_score
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

POOL = Path("eval/judge_retrieval_pool.json")
GRADES = Path("eval/judge_retrieval_grades.json")
REPORT = Path("eval/judge_retrieval_report.json")
K = 5


def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def main() -> None:
    items = json.loads(POOL.read_text(encoding="utf-8"))["items"]
    grades = json.loads(GRADES.read_text(encoding="utf-8"))

    per_system: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    no_answer: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "false_positive": 0})
    ungraded = 0

    for it in items:
        g = grades.get(it["sample_id"], {})
        pool_gains = []
        for pid in {p["page_id"] for p in it["pool"]}:
            if g.get(pid) is None:
                ungraded += 1
            pool_gains.append(g.get(pid) or 0)
        ideal = sorted(pool_gains, reverse=True)[:K]
        idcg = dcg(ideal)

        for name, top in it["systems"].items():
            gains = [g.get(pid) or 0 for pid in top[:K]]
            if not it["expect_answer"]:
                no_answer[name]["n"] += 1
                no_answer[name]["false_positive"] += int(any(x > 0 for x in gains))
                continue
            key = f"{it['corpus']}/{it['type']}"
            for bucket in ("all", key):
                m = per_system[name][bucket]
                m.append({
                    "ndcg": (dcg(gains) / idcg) if idcg > 0 else None,
                    "recall": int(any(x == 2 for x in gains)),
                    "irrelevant": sum(1 for x in gains if x == 0) / max(len(gains), 1),
                    "gold": int(it.get("gold_page_id") in top[:K]) if it.get("gold_page_id") else None,
                })

    def agg(rows: list[dict]) -> dict:
        nd = [r["ndcg"] for r in rows if r["ndcg"] is not None]
        gold = [r["gold"] for r in rows if r["gold"] is not None]
        return {
            "n": len(rows),
            "ndcg@5": sum(nd) / len(nd) if nd else None,
            "recall@5": sum(r["recall"] for r in rows) / len(rows),
            "irrelevant@5": sum(r["irrelevant"] for r in rows) / len(rows),
            "gold_in_top5": sum(gold) / len(gold) if gold else None,
        }

    report = {
        "ungraded_pages": ungraded,
        "systems": {name: {b: agg(rows) for b, rows in buckets.items()} for name, buckets in per_system.items()},
        "no_answer_queries": dict(no_answer),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if ungraded:
        print(f"경고: 등급 미기입 페이지 {ungraded}건 (0으로 계산됨)")


if __name__ == "__main__":
    main()
