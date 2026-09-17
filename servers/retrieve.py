"""검색: BGE-M3 임베딩 → 로컬 벡터 스토어 top-k(브루트포스) [+ D-20: BM25 RRF 병합] → bge-reranker-v2-m3 재랭킹 → top-n.

D-02: 서빙 시점에는 VLM이 GPU를 점유하므로 임베더/리랭커는 CPU에서 돈다.
D-06: ChromaDB 대신 indexing.vector_store.VectorStore를 쓴다(배경은 그 모듈 참고).
D-20: DART 실사용형 질의에서 밀집 검색이 회사명·연도 같은 고유명사 일치를 보장하지 못해
(D-19) BM25를 옆에 붙여 RRF로 합친다. 저장소에 bm25.npz가 없으면 밀집 검색만 한다(하위 호환).
search()/rerank()로 단계를 분리해 metrics.py에서 단계별 시간을 잴 수 있게 한다.

D-27·D-28: 표지·구분지·목차(no_content) 페이지는 제목만으로 질의와 잘 맞아 검색 상위에
오르지만 답변 근거가 되지 못한다. section_meta.jsonl(indexing.section_bounds)이 저장소에
있으면 이 페이지를 후보에서 제외한다(filter_no_content). 제외분을 메우기 위해 dense/BM25
모두 k보다 넉넉히(no_content_overfetch_factor배) 가져온 뒤 필터링한다.
"""

from __future__ import annotations

import json
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


_section_meta_cache: dict[str, dict] = {}


def get_section_meta() -> dict:
    """운영 저장소 디렉터리의 section_meta.jsonl(indexing.section_bounds 산출물). 없으면 {}."""
    key = str(settings.vector_store_dir)
    if key not in _section_meta_cache:
        path = settings.vector_store_dir / "section_meta.jsonl"
        meta: dict = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    meta[rec["id"]] = rec
        _section_meta_cache[key] = meta
    return _section_meta_cache[key]


_id_row_cache: dict[str, dict[str, int]] = {}


def _id_row_index(store) -> dict[str, int]:
    key = str(settings.vector_store_dir)
    if key not in _id_row_cache:
        _id_row_cache[key] = {pid: i for i, pid in enumerate(store.ids)}
    return _id_row_cache[key]


def get_page_by_id(page_id: str) -> Candidate | None:
    """페이지 ID로 저장소에서 직접 한 건을 찾는다 (D-28: 부모-자식 이웃 페이지 조회용)."""
    store = get_store()
    row = _id_row_index(store).get(page_id)
    return _candidate(store, row) if row is not None else None


def get_neighbor_candidate(page_id: str) -> Candidate | None:
    """page_id와 같은 절의 다음 페이지. 없거나 no_content 페이지면 None (D-28).

    답이 표지·구분지 바로 다음 페이지에 이어지는 경우를 보완한다(텍스트/이미지 모두).
    실측: "제10절에서 재정계획의 목표는?" 질의는 정답이 이미지가 아니라 다음 페이지의 텍스트에
    있었다 — 이미지만 보완하면 놓치므로 텍스트 청크째로 돌려준다.
    """
    section_meta = get_section_meta()
    rec = section_meta.get(page_id)
    if not rec or not rec.get("next_id"):
        return None
    next_id = rec["next_id"]
    neighbor_meta = section_meta.get(next_id)
    if neighbor_meta and neighbor_meta.get("no_content"):
        return None
    return get_page_by_id(next_id)


def _candidate(store, i: int) -> Candidate:
    m = store.metadatas[i]
    return Candidate(
        page_id=store.ids[i],
        text=store.documents[i],
        route=m.get("route", ""),
        source_file=m.get("source_file", ""),
        image_path=m.get("image_path") or None,
    )


def _search_pool(query: str, query_vec, store, bm25: BM25Index | None, fetch_k: int, is_ok) -> list[Candidate]:
    """dense(+BM25) 후보를 fetch_k개씩 가져와 RRF로 합치고 is_ok로 거른 뒤 점수 내림차순으로 돌려준다."""
    dense = store.query(query_vec, n_results=fetch_k)
    if not dense.ids:
        return []

    if bm25 is None:
        return [
            Candidate(
                page_id=dense.ids[i], text=dense.documents[i], route=dense.metadatas[i].get("route", ""),
                source_file=dense.metadatas[i].get("source_file", ""), image_path=dense.metadatas[i].get("image_path") or None,
            )
            for i in range(len(dense.ids)) if is_ok(dense.ids[i])
        ]

    # RRF 병합. 벡터 결과는 id로, BM25 결과는 행 번호로 오므로 행 번호 기준으로 맞춘다.
    id_to_row = {pid: i for i, pid in enumerate(store.ids)}
    fused: dict[int, float] = {}
    for rank, pid in enumerate(dense.ids, 1):
        if not is_ok(pid):
            continue
        fused[id_to_row[pid]] = fused.get(id_to_row[pid], 0.0) + 1.0 / (settings.rrf_k + rank)
    rows, _ = bm25.query(query, n_results=fetch_k)
    for rank, row in enumerate(rows.tolist(), 1):
        if not is_ok(store.ids[row]):
            continue
        fused[row] = fused.get(row, 0.0) + 1.0 / (settings.rrf_k + rank)
    top_rows = sorted(fused, key=fused.get, reverse=True)
    return [_candidate(store, row) for row in top_rows]


def search(query: str, device: str | None = None) -> list[Candidate]:
    """벡터 top-k (+ D-20: BM25 top-k를 RRF로 병합) 검색. 재랭킹 이전 단계.

    hybrid_enabled가 켜져 있고 저장소에 bm25.npz가 있으면 두 순위를 RRF(1/(rrf_k+rank))로 합쳐
    상위 top_k_candidates개를 돌려준다. 그렇지 않으면 밀집 검색만 한다.

    D-27·D-28: filter_no_content가 켜져 있고 section_meta.jsonl이 있으면 no_content 페이지
    (표지·구분지·목차)를 후보에서 뺀다. 처음엔 원래와 같은 k로만 가져온다 — no_content 페이지가
    실제로 그 k 안에 없는 대다수 질의에서는 필터링 전과 순위가 완전히 같아야 하기 때문이다.
    (실측: dense/BM25 모두 k*3으로 넉넉히 가져와 한 번에 거르면, no_content와 무관한 질의에서도
    RRF 융합 풀이 넓어져 순위가 흔들려 SDS 600 R@5가 0.762→0.743으로 오히려 떨어졌다. 원인은
    필터링 자체가 아니라 이 "일괄 확대"였다.) 필터링으로 k개를 못 채울 때만 fetch_k를 넓혀 한 번 더
    가져온다.
    """
    dev = device or settings.embed_device_serving
    embedder = get_embedder(dev)
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]

    store = get_store()
    k = settings.top_k_candidates
    section_meta = get_section_meta() if settings.filter_no_content else {}

    def is_ok(pid: str) -> bool:
        rec = section_meta.get(pid)
        return not (rec and rec.get("no_content"))

    bm25 = get_bm25() if settings.hybrid_enabled else None
    candidates = _search_pool(query, query_vec, store, bm25, k, is_ok)
    if section_meta and len(candidates) < k:
        wider = _search_pool(query, query_vec, store, bm25, k * settings.no_content_overfetch_factor, is_ok)
        if len(wider) > len(candidates):
            candidates = wider
    return candidates[:k]


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
