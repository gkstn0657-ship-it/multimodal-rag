"""검색: BGE-M3 임베딩 → 로컬 벡터 스토어 top-k(브루트포스) [+ D-20: BM25 RRF 병합] → bge-reranker-v2-m3 재랭킹 → top-n.

D-02: 서빙 시점에는 VLM이 GPU를 점유하므로 임베더/리랭커는 CPU에서 돈다.
D-06: ChromaDB 대신 indexing.vector_store.VectorStore를 쓴다(배경은 그 모듈 참고).
D-20: DART 실사용형 질의에서 밀집 검색이 회사명·연도 같은 고유명사 일치를 보장하지 못해
(D-19) BM25를 옆에 붙여 RRF로 합친다. 저장소에 bm25.npz가 없으면 밀집 검색만 한다(하위 호환).
search()/rerank()로 단계를 분리해 metrics.py에서 단계별 시간을 잴 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from config import settings
from indexing.bm25_index import BM25Index
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


_bm25_cache: dict[str, BM25Index] = {}


def get_bm25() -> BM25Index | None:
    """운영 저장소 디렉터리의 BM25 인덱스. 없으면 None(밀집 검색만 수행)."""
    key = str(settings.vector_store_dir)
    if key not in _bm25_cache:
        _bm25_cache[key] = BM25Index.load(settings.vector_store_dir) if BM25Index.exists(settings.vector_store_dir) else None
    return _bm25_cache[key]


def _candidate(store, i: int) -> Candidate:
    m = store.metadatas[i]
    return Candidate(
        page_id=store.ids[i],
        text=store.documents[i],
        route=m.get("route", ""),
        source_file=m.get("source_file", ""),
        image_path=m.get("image_path") or None,
    )


def search(query: str, device: str | None = None) -> list[Candidate]:
    """벡터 top-k (+ D-20: BM25 top-k를 RRF로 병합) 검색. 재랭킹 이전 단계.

    hybrid_enabled가 켜져 있고 저장소에 bm25.npz가 있으면 두 순위를 RRF(1/(rrf_k+rank))로 합쳐
    상위 top_k_candidates개를 돌려준다. 그렇지 않으면 밀집 검색만 한다.
    """
    dev = device or settings.embed_device_serving
    embedder = get_embedder(dev)
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]

    store = get_store()
    k = settings.top_k_candidates
    dense = store.query(query_vec, n_results=k)
    if not dense.ids:
        return []

    bm25 = get_bm25() if settings.hybrid_enabled else None
    if bm25 is None:
        return [
            Candidate(
                page_id=dense.ids[i], text=dense.documents[i], route=dense.metadatas[i].get("route", ""),
                source_file=dense.metadatas[i].get("source_file", ""), image_path=dense.metadatas[i].get("image_path") or None,
            )
            for i in range(len(dense.ids))
        ]

    # RRF 병합. 벡터 결과는 id로, BM25 결과는 행 번호로 오므로 행 번호 기준으로 맞춘다.
    id_to_row = {pid: i for i, pid in enumerate(store.ids)}
    fused: dict[int, float] = {}
    for rank, pid in enumerate(dense.ids, 1):
        fused[id_to_row[pid]] = fused.get(id_to_row[pid], 0.0) + 1.0 / (settings.rrf_k + rank)
    rows, _ = bm25.query(query, n_results=k)
    for rank, row in enumerate(rows.tolist(), 1):
        fused[row] = fused.get(row, 0.0) + 1.0 / (settings.rrf_k + rank)
    top_rows = sorted(fused, key=fused.get, reverse=True)[:k]
    return [_candidate(store, row) for row in top_rows]


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
