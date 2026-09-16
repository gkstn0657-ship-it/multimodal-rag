"""IMAGE 경로 페이지용 합성 질의 생성 (D-15).

IMAGE 경로 997장 중 200장을 뽑아, VLM이 페이지 이미지를 보고 "이 페이지로만 답할 수 있는 질문"
하나를 쓰게 한다. 정답은 그 페이지다. 결과는 `eval/synth_queries_image_pages.jsonl`에 한 줄씩
즉시 기록하고, 재실행하면 이미 만든 페이지는 건너뛴다.

편향: 질문을 쓴 모델이 캡션을 쓴 모델과 같다(Qwen2.5-VL). 캡션 텍스트는 보여주지 않고 이미지에서
직접 생성했지만, 같은 모델이 같은 페이지에서 고르는 어휘가 겹칠 수 있어 캡션 저장소에 유리할 수 있다.

사용법:
    python -m eval.synth_query_gen            # 200장, 병렬 2
"""

from __future__ import annotations

import json
import pathlib
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from indexing.embed_store import image_path_for
from servers.vlm_client import VLMClient

SAMPLE_SIZE = 200
SEED = 2026
PARALLEL = 2
OUT = pathlib.Path("eval/synth_queries_image_pages.jsonl")

PROMPT = (
    "당신은 이 문서 페이지를 읽은 공공기관 직원입니다. 이 페이지의 내용으로만 답할 수 있는, "
    "실제 업무에서 던질 법한 구체적인 질문을 한국어로 한 문장 쓰세요. "
    "규칙: 질문만 출력하고 다른 말은 쓰지 마세요. '이 페이지', '이 문서', '위 표'처럼 페이지를 가리키는 표현은 쓰지 마세요. "
    "페이지에 있는 구체적인 명칭·항목·연도·수치를 질문 안에 자연스럽게 넣으세요. 페이지가 표지나 목차뿐이면 그 제목에 대한 질문을 쓰세요."
)


def main() -> None:
    ids = [json.loads(l)["page_id"] for l in open("data/captions.jsonl", encoding="utf-8")]
    random.seed(SEED)
    sample = random.sample(ids, SAMPLE_SIZE)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["page_id"] for l in OUT.open(encoding="utf-8")}
    todo = [p for p in sample if p not in done]
    print(f"IMAGE 경로 {len(ids)}장 | 표본 {len(sample)} | 이미 생성 {len(done)} | 남은 {len(todo)}", flush=True)

    client = VLMClient()

    def gen(pid: str) -> tuple[str, str]:
        img = Image.open(image_path_for(pid))
        q = client.caption(img, PROMPT, max_tokens=120).strip().splitlines()[0].strip()
        return pid, q

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(PARALLEL) as ex, OUT.open("a", encoding="utf-8") as f:
        for fut in as_completed([ex.submit(gen, p) for p in todo]):
            try:
                pid, q = fut.result()
            except Exception as exc:  # noqa: BLE001
                print("ERR", repr(exc)[:120], flush=True)
                continue
            f.write(json.dumps({"page_id": pid, "query": q}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"완료 {n} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
