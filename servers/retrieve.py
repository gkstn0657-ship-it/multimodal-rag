"""검색: BGE-M3 임베딩 → 로컬 벡터 스토어 top-k(브루트포스) → bge-reranker-v2-m3 재랭킹 → top-n.

D-02: 서빙 시점에는 VLM이 GPU를 점유하므로 임베더/리랭커는 CPU에서 돈다.
D-06: ChromaDB 대신 indexing.vector_store.VectorStore를 쓴다(배경은 그 모듈 참고).
search()/rerank()로 단계를 분리해 metrics.py에서 단계별 시간을 잴 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from config import settings
from indexing.embed_store import get_embedder, get_store


@dataclass
class Candidate:
    page_id: str
    text: str
    route: str
    source_file: str
    image_path: str | None


@dataclass
class RetrievedChunk:
    page_id: str
    text: str
    route: str
    source_file: str
    image_path: str | None
    score: float


_reranker_cache: dict[str, CrossEncoder] = {}


def get_reranker(device: str) -> CrossEncoder:
    if device not in _reranker_cache:
        _reranker_cache[device] = CrossEncoder(settings.rerank_model_name, device=device)
    return _reranker_cache[device]


def search(query: str, device: str | None = None) -> list[Candidate]:
    """임베딩 + 로컬 벡터 스토어 top-k 검색 (재랭킹 이전 단계)."""
    dev = device or settings.embed_device_serving
    embedder = get_embedder(dev)
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]

    store = get_store()
    result = store.query(query_vec, n_results=settings.top_k_candidates)

    if not result.ids:
        return []

    return [
        Candidate(
            page_id=result.ids[i],
            text=result.documents[i],
            route=result.metadatas[i].get("route", ""),
            source_file=result.metadatas[i].get("source_file", ""),
            image_path=result.metadatas[i].get("image_path") or None,
        )
        for i in range(len(result.ids))
    ]


def rerank(query: str, candidates: list[Candidate], device: str | None = None) -> list[RetrievedChunk]:
    """재랭킹 후 top-n만 반환."""
    if not candidates:
        return []

    dev = device or settings.embed_device_serving
    reranker = get_reranker(dev)
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)

    ranked = [
        RetrievedChunk(
            page_id=c.page_id,
            text=c.text,
            route=c.route,
            source_file=c.source_file,
            image_path=c.image_path,
            score=float(scores[i]),
        )
        for i, c in enumerate(candidates)
    ]
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked[: settings.top_n_after_rerank]


def retrieve(query: str, device: str | None = None) -> list[RetrievedChunk]:
    """search() + rerank()를 한 번에 수행하는 편의 함수."""
    candidates = search(query, device=device)
    return rerank(query, candidates, device=device)
