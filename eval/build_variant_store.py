"""IMAGE 경로 청크의 임베딩 텍스트만 바꾼 변형 벡터 저장소를 만든다 (D-15·D-17 비교용).

기본 저장소(data/vector_store)를 그대로 복사하되, route == "image" 인 행만 지정한 캡션 파일
(+ 선택적으로 OCR/텍스트 합집합)으로 텍스트를 바꾸고 그 행들만 다시 임베딩한다.
TEXT/OCR 경로 39,000여 행은 재임베딩하지 않으므로 GPU에서 1~2분이면 끝난다.

D-15의 변형 저장소(v1_caption, v1_caption_ocr, v2_caption_ocr)도 같은 방식으로 만들었다.
D-02 순서: VLM이 GPU를 쓰는 동안 돌리지 않는다(캡셔닝이 끝난 뒤 실행).

사용법:
    python -m eval.build_variant_store --captions data/captions_v3.jsonl --with-ocr \
        --out data/vector_store_variant_v3_caption_ocr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from config import settings
from indexing.caption import CaptionCache
from indexing.embed_store import get_embedder
from indexing.vector_store import VectorStore


def load_ocr_for(page_ids: set[str]) -> dict[str, str]:
    """코퍼스 파켓에서 지정 페이지의 OCR(없으면 본문 텍스트)만 읽는다. 이미지 컬럼은 읽지 않는다."""
    table = pq.read_table(settings.corpus_parquet, columns=["id", "text", "ocr"])
    out: dict[str, str] = {}
    for pid, text, ocr in zip(table["id"].to_pylist(), table["text"].to_pylist(), table["ocr"].to_pylist()):
        if pid in page_ids:
            out[pid] = ((ocr or "") or (text or "")).strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True, help="캡션 JSONL (page_id, caption)")
    parser.add_argument("--with-ocr", action="store_true", help="캡션 뒤에 OCR/텍스트를 이어 붙인다")
    parser.add_argument("--out", required=True, help="출력 저장소 디렉터리")
    parser.add_argument("--base", default=None, help="기본 저장소 (기본: settings.vector_store_dir)")
    args = parser.parse_args()

    base = VectorStore.load(Path(args.base) if args.base else settings.vector_store_dir)
    captions = CaptionCache(Path(args.captions)).as_dict()
    image_idx = [i for i, m in enumerate(base.metadatas) if m.get("route") == "image"]
    image_ids = {base.ids[i] for i in image_idx}
    ocr_map = load_ocr_for(image_ids) if args.with_ocr else {}

    new_texts: list[str] = []
    missing = 0
    for i in image_idx:
        pid = base.ids[i]
        cap = captions.get(pid, "").strip()
        if not cap:
            missing += 1
        parts = [cap, ocr_map.get(pid, "")] if args.with_ocr else [cap]
        text = "\n".join(p for p in parts if p)
        new_texts.append(text or base.documents[i])
    print(f"IMAGE 경로 {len(image_idx)}행 교체, 캡션 없는 페이지 {missing}장 (OCR/기존 텍스트 유지)")

    embedder = get_embedder(settings.embed_device_indexing)
    vectors = embedder.encode(
        new_texts, normalize_embeddings=True, batch_size=settings.embed_batch_size, show_progress_bar=True
    )

    matrix = base._matrix.copy()
    for row, i in enumerate(image_idx):
        matrix[i] = vectors[row]
        base.documents[i] = new_texts[row]
    base._matrix = matrix.astype(np.float32)
    base.save(Path(args.out))
    print(f"저장: {args.out} ({len(base)}행)")


if __name__ == "__main__":
    main()
