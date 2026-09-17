"""검색 판정 평가 표본 고정 (D-18, eval/retrieval_judge_rubric.md).

채점 전에 문항을 뽑아 파일로 고정하고 커밋한다. 결과를 본 뒤 문항을 바꾸지 않는다.

- SDS KoPub 라벨셋 15: type(text/visual/cross)별 5개, seed 고정
- 합성 IMAGE 경로 질의 15: seed 고정
- DART 10 + 답 없는 질의 2: 아래에 손으로 썼다. 회사·연도가 명시된 실사용형 질의 위주.

사용법:
    python -m eval.make_judge_sample
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from config import settings
from eval.synth_query_eval import load_queries

SEED = 20260917
OUT = Path("eval/judge_sample_retrieval.json")

# DART 코퍼스: 삼성전자·현대자동차·LG화학·NAVER·카카오, 사업연도 2022~2024 사업보고서(정정 포함).
# expect_answer=False 는 코퍼스에 답이 없도록 쓴 문항(기권 평가용).
DART_QUERIES = [
    {"query": "삼성전자 2023년 연결기준 매출액", "expect_answer": True},
    {"query": "카카오 2024년 사업보고서 임직원 수", "expect_answer": True},
    {"query": "현대자동차 2023년과 2024년 영업이익 비교", "expect_answer": True},
    {"query": "NAVER 2022 최대주주 지분율", "expect_answer": True},
    {"query": "LG화학 2023년 주당 배당금", "expect_answer": True},
    {"query": "카카오 사업보고서 정정 사유", "expect_answer": True},
    {"query": "삼성전자 2024 연구개발비 매출 대비 비율", "expect_answer": True},
    {"query": "네이버 2023년 등기임원 보수 총액", "expect_answer": True},
    {"query": "현대자동차 2024 사업보고서 주요 제품 매출 비중", "expect_answer": True},
    {"query": "LG화학 2022 주주총회 안건", "expect_answer": True},
    {"query": "카카오 2025년 상반기 매출", "expect_answer": False},
    {"query": "삼성전자 2021년 배당 결정", "expect_answer": False},
]


def main() -> None:
    rng = random.Random(SEED)

    qa = pq.read_table(settings.qa_parquet).to_pylist()
    sds: list[dict] = []
    for t in ("text", "visual", "cross"):
        rows = [r for r in qa if r["type"] == t]
        for r in rng.sample(rows, 5):
            sds.append({"corpus": "sds", "source": "sds_kopub_qa", "type": t, "qa_id": r["id"],
                        "query": r["query"], "gold_page_id": r["id"], "expect_answer": True})

    synth = [{"corpus": "sds", "source": "synth_image_pages", "type": "image", "query": q["query"],
              "gold_page_id": q["page_id"], "expect_answer": True}
             for q in rng.sample(load_queries(), 15)]

    dart = [{"corpus": "dart", "source": "manual", "type": "dart", **q, "gold_page_id": None} for q in DART_QUERIES]

    sample = sds + synth + dart
    for i, s in enumerate(sample, 1):
        s["sample_id"] = f"J{i:02d}"
    OUT.write_text(json.dumps({"seed": SEED, "n": len(sample), "items": sample}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"표본 {len(sample)}문항 저장: {OUT} (sds {len(sds)}, synth {len(synth)}, dart {len(dart)})")


if __name__ == "__main__":
    main()
