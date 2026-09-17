"""생성 품질 재평가용 답변 재생성 (D-24).

D-11~D-13(동적 컨텍스트 예산, 이미지 상위 랭크 제한, temperature 0, 자기모순 기권 수정)과
D-19~D-21(v3 캡션+OCR 운영 저장소, BM25 하이브리드+접두어)이 전부 반영된 현재 파이프라인으로
`eval/gen_eval_sample.json` 15문항을 ours/baseline 두 조건으로 다시 생성한다.

사용법:
    python -m eval.regen_gen_answers
"""

from __future__ import annotations

import json
from pathlib import Path

from config import settings
import indexing.embed_store as es
from servers.answer import generate_from_chunks
from servers.retrieve import retrieve
from servers.vlm_client import VLMClient

SAMPLE = Path("eval/gen_eval_sample.json")
OUT = Path("eval/gen_eval_answers_v2.json")


def main() -> None:
    settings.vector_store_dir = Path("data/vector_store")
    settings.hybrid_enabled = True
    es.get_store(force_reload=True)

    items = json.loads(SAMPLE.read_text(encoding="utf-8"))
    client = VLMClient()
    out = []
    for i, it in enumerate(items, 1):
        chunks = retrieve(it["query"])
        ours = generate_from_chunks(it["query"], chunks, client=client, use_images=True)
        baseline = generate_from_chunks(it["query"], chunks, client=client, use_images=False)
        out.append({
            "idx": i, "id": it["id"], "type": it["type"], "query": it["query"],
            "gold_answer": it["answer"],
            "ours_answer": ours.answer, "ours_sources": ours.sources[:3],
            "baseline_answer": baseline.answer, "baseline_sources": baseline.sources[:3],
        })
        print(f"[{i}/15] {it['type']} done", flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
