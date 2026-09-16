"""IMAGE 경로 페이지 재캡셔닝 (D-15, 전사 우선 프롬프트 v3).

data/captions.jsonl(v2)에 있는 페이지 ID를 그대로 대상으로 삼고, 렌더링된 페이지 이미지
(data/rendered_pages)를 읽어 캡셔닝한다. 코퍼스 파켓을 다시 훑지 않는다.
결과는 data/captions_v3.jsonl에 즉시 기록하며 재실행하면 이어서 간다.

사용법:
    python -m indexing.recaption            # 병렬 4
"""

from __future__ import annotations

import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from config import settings
from indexing.caption import CAPTION_PROMPT_V3, CaptionCache, caption_page
from indexing.embed_store import image_path_for
from servers.vlm_client import VLMClient

SRC = pathlib.Path("data/captions.jsonl")
OUT = pathlib.Path("data/captions_v3.jsonl")


def main() -> None:
    ids = [json.loads(l)["page_id"] for l in SRC.open(encoding="utf-8")]
    cache = CaptionCache(OUT)
    todo = [p for p in ids if p not in cache]
    print(f"대상 {len(ids)}장 | 캐시 {len(cache)} | 남은 {len(todo)} | 병렬 {settings.caption_parallel}", flush=True)
    client = VLMClient()
    if not client.ping():
        raise RuntimeError("VLM 서버 연결 실패")
    t0 = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(settings.caption_parallel) as ex:
        futs = {ex.submit(caption_page, client, pid, Image.open(image_path_for(pid)), 1, CAPTION_PROMPT_V3): pid for pid in todo}
        for fut in as_completed(futs):
            r = fut.result()
            if r.error:
                fail += 1
                print("ERR", futs[fut][-40:], r.error[:80], flush=True)
            else:
                cache.put(r)
                ok += 1
            if (ok + fail) % 50 == 0:
                print(f"{ok + fail}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"완료: 성공 {ok} 실패 {fail} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
