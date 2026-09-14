"""BGE-M3 임베딩 + ChromaDB 저장.

D-02에 따라 device는 호출자가 명시한다:
  - 인덱싱 스크립트(run_index.py)는 VLM을 내린 뒤 embed_device_indexing(cuda)로 호출
  - 서빙 코드(retrieve.py)는 embed_device_serving(cpu)로 호출
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings
from indexing.ingest import Route, RoutedPage


@dataclass
class IndexedChunk:
    page_id: str
    route: str
    text_for_embedding: str  # 텍스트 경로: 본문 / 이미지 경로: 캡션
    source_file: str
    image_path: str | None  # 이미지 경로 페이지만 채워짐 (답변 단계에서 원본 이미지 재사용)


_embedder_cache: dict[str, SentenceTransformer] = {}


def get_embedder(device: str) -> SentenceTransformer:
    if device not in _embedder_cache:
        _embedder_cache[device] = SentenceTransformer(settings.embed_model_name, device=device)
    return _embedder_cache[device]


def source_file_from_page_id(page_id: str) -> str:
    """'public_pdf/prism/파일명_페이지번호' 형태의 id에서 파일명 부분을 추출한다."""
    # 마지막 '_' 뒤가 페이지 인덱스인 경우가 많으나, 파일명 자체에 '_'가 많아
    # 안전하게 전체 id를 source_file로 쓰고, 별도 page_no는 메타데이터로 남기지 않는다
    # (SDS KoPub는 annotations.parquet의 page_indices로 페이지-문서 매핑을 제공하므로
    #  향후 정확한 매핑이 필요하면 annotations를 조인한다).
    return page_id


def build_chunks(routed_pages: list[RoutedPage], captions: dict[str, str]) -> list[IndexedChunk]:
    """라우팅 결과 + 캡션 결과를 임베딩 대상 청크로 변환한다."""
    chunks: list[IndexedChunk] = []
    for rp in routed_pages:
        page = rp.page
        if rp.route == Route.TEXT:
            text_for_embedding = page.text
            image_path = None
        elif rp.route == Route.IMAGE:
            text_for_embedding = captions.get(page.id, "")
            image_path = str(settings.rendered_pages_dir / f"{_safe_name(page.id)}.png")
        else:  # IMAGE_OVERFLOW: 상한 초과 -> OCR로 대체, 원본 이미지는 저장하지 않음
            text_for_embedding = page.ocr or page.text
            image_path = None

        if not text_for_embedding.strip():
            continue

        chunks.append(
            IndexedChunk(
                page_id=page.id,
                route=rp.route.value,
                text_for_embedding=text_for_embedding,
                source_file=source_file_from_page_id(page.id),
                image_path=image_path,
            )
        )
    return chunks


def _safe_name(page_id: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in page_id)[:150]


def save_page_image(page_id: str, image) -> str:
    """이미지 경로 페이지의 원본 이미지를 디스크에 저장하고 경로를 반환한다."""
    settings.rendered_pages_dir.mkdir(parents=True, exist_ok=True)
    path = settings.rendered_pages_dir / f"{_safe_name(page_id)}.png"
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(path, format="PNG")
    return str(path)


def get_collection():
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(settings.collection_name)


def embed_and_store(chunks: list[IndexedChunk], device: str, batch_size: int = 64) -> int:
    """청크를 임베딩하여 ChromaDB에 저장한다. 저장된 청크 수를 반환한다."""
    if not chunks:
        return 0

    embedder = get_embedder(device)
    collection = get_collection()

    stored = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text_for_embedding for c in batch]
        vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        collection.upsert(
            ids=[c.page_id for c in batch],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[
                {
                    "route": c.route,
                    "source_file": c.source_file,
                    "image_path": c.image_path or "",
                }
                for c in batch
            ],
        )
        stored += len(batch)

    return stored
