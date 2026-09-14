"""SDS KoPub VDR 코퍼스 파켓 리더.

코퍼스는 이미 페이지 단위로 파싱되어 있어(image/text/ocr 컬럼) PDF 렌더링이
필요 없다. 이 프로젝트에서 "인덱싱"은 각 페이지를 텍스트/이미지 경로로
라우팅하고, 이미지 경로는 VLM 캡셔닝을 거치는 과정이다.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from PIL import Image

from config import settings


@dataclass
class Page:
    id: str
    text: str
    ocr: str
    image_bytes: bytes | None

    def to_pil(self) -> Image.Image:
        if self.image_bytes is None:
            raise ValueError(f"페이지 {self.id}에 이미지가 없습니다.")
        return Image.open(io.BytesIO(self.image_bytes))


def iter_pages(parquet_path: Path | None = None, limit: int | None = None) -> Iterator[Page]:
    """코퍼스 파켓을 스트리밍으로 읽어 Page를 하나씩 반환한다.

    40,781페이지 전체를 한 번에 메모리에 올리지 않도록 row-group 단위로 읽는다.
    """
    path = parquet_path or settings.corpus_parquet
    pf = pq.ParquetFile(path)

    count = 0
    for batch in pf.iter_batches(batch_size=256, columns=["id", "text", "ocr", "image"]):
        d = batch.to_pydict()
        for i in range(len(d["id"])):
            image_field = d["image"][i]
            # HF datasets Image 컬럼은 보통 {"bytes": ..., "path": ...} 구조의 struct.
            image_bytes = None
            if image_field is not None:
                if isinstance(image_field, dict):
                    image_bytes = image_field.get("bytes")
                elif isinstance(image_field, (bytes, bytearray)):
                    image_bytes = bytes(image_field)

            yield Page(
                id=d["id"][i],
                text=d["text"][i] or "",
                ocr=d["ocr"][i] or "",
                image_bytes=image_bytes,
            )
            count += 1
            if limit is not None and count >= limit:
                return


def count_pages(parquet_path: Path | None = None) -> int:
    path = parquet_path or settings.corpus_parquet
    return pq.ParquetFile(path).metadata.num_rows
