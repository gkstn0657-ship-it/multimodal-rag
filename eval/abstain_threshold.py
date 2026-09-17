"""기권 게이트 임계값 분석 (D-22).

판정 표본 42문항을 현재 운영 설정(v3 캡션+OCR, BM25 하이브리드+접두어, D-19~D-21)으로 다시 검색해
리랭크 점수를 얻고, 이미 채점된 등급(judge_retrieval_grades.json)과 묶어 점수 분포를 본다.

기권 게이트는 "리랭크 1위 점수 < 임계값이면 VLM을 부르지 않고 기권한다"이므로, 볼 것은 1위 점수와
1위 페이지의 등급의 관계다:
  - 1위가 등급 2(정답)인데 기권하면: 과잉기권(false abstain) — 원래 맞힐 질의를 놓친다.
  - 1위가 등급 0/1이거나 애초에 정답이 없는 질의(expect_answer=False)인데 응답하면: 원래 있어야 할
    기권을 놓친 것 — 무관한 근거로 그럴듯한 오답을 만드는 D-19/D-20/D-21의 실패 유형이다.

사용법:
    python -m eval.abstain_threshold
"""

from __future__ import annotations

import json
from pathlib import Path

from config import settings
import indexing.embed_store as es
from servers.retrieve import retrieve

SAMPLE = Path("eval/judge_sample_retrieval.json")
GRADES = Path("eval/judge_retrieval_grades.json")
OUT = Path("eval/abstain_threshold_report.json")

STORE_BY_CORPUS = {"sds": Path("data/vector_store"), "dart": Path("data/vector_store_dart")}


def main() -> None:
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]
    grades = json.loads(GRADES.read_text(encoding="utf-8"))
    settings.hybrid_enabled = True

    rows = []
    ungraded_top1 = []
    for corpus, store in STORE_BY_CORPUS.items():
        settings.vector_store_dir = store
        es.get_store(force_reload=True)
        items = [s for s in sample if s["corpus"] == corpus]
        for s in items:
            chunks = retrieve(s["query"])
            if not chunks:
                rows.append({"sample_id": s["sample_id"], "corpus": corpus, "expect_answer": s["expect_answer"],
                             "top1_score": None, "top1_grade": None, "top1_page": None})
                continue
            top1 = chunks[0]
            g = grades.get(s["sample_id"], {}).get(top1.page_id)
            if g is None:
                ungraded_top1.append((s["sample_id"], top1.page_id))
            rows.append({
                "sample_id": s["sample_id"], "corpus": corpus, "query": s["query"],
                "expect_answer": s["expect_answer"], "top1_page": top1.page_id,
                "top1_score": round(float(top1.score), 4), "top1_grade": g,
            })
        print(f"{corpus}: {len(items)}문항 검색 완료", flush=True)

    rows.sort(key=lambda r: (r["top1_score"] is None, r["top1_score"]))
    OUT.write_text(json.dumps({"rows": rows, "ungraded_top1": ungraded_top1}, ensure_ascii=False, indent=2), encoding="utf-8")

    if ungraded_top1:
        print(f"경고: 채점 안 된 1위 페이지 {len(ungraded_top1)}건, eval/abstain_threshold_report.json 참고", flush=True)

    print("\n정렬(점수 오름차순): sample_id | score | grade | expect_answer | corpus")
    for r in rows:
        print(f"  {r['sample_id']:4s} | {r['top1_score']} | grade={r['top1_grade']} | "
              f"expect={r['expect_answer']} | {r['corpus']}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
