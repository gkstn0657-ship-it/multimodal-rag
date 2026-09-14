"""로컬 브루트포스 벡터 저장소 (D-06, ChromaDB 대체).

ChromaDB의 Rust HNSW 백엔드가 Windows + PersistentClient + 대규모(수만 건)
조합에서 인덱스 파일을 디스크에 완전히 쓰지 못하는 버그가 있다
(재실행 시 "Error loading hnsw index"). chroma-core/chroma#4212,
zylon-ai/private-gpt#2180 등 동일 증상이 보고돼 있고 미해결 상태다.
대안으로 고려한 chromadb<1.0(0.5.23)은 chroma-hnswlib==0.7.6을 요구하는데
이 조합엔 Windows용 사전 빌드 휠이 없어 로컬 빌드가 필요하다(Visual Studio 없음).

이 코퍼스는 39,458개 청크 x 1024차원 float32 = 약 161MB로, 전체를 메모리에
올려 브루트포스 코사인 유사도(내적, 벡터가 이미 정규화됨)로 검색해도
질의당 수 밀리초면 충분하다. ANN 인덱스가 필요한 규모(수백만 건)가 아니므로
이 방식이 더 단순하고 안정적이다.

파일 포맷: vectors.npy (N x D float32), meta.jsonl (N줄, 한 줄당 id/document/metadata).
저장은 인덱싱 종료 시 한 번만 한다 — 부분적으로 쓰인 상태가 남을 여지가 없다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class QueryResult:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    scores: list[float]  # 코사인 유사도(내적), 높을수록 유사


class VectorStore:
    """add()로 누적 → save()로 한 번에 기록. load()로 읽어 query()."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None  # load() 또는 build() 이후 채워짐

    def add(self, ids: list[str], vectors: np.ndarray, documents: list[str], metadatas: list[dict]) -> None:
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._vectors.append(np.asarray(vectors, dtype=np.float32))

    def build(self) -> None:
        """누적된 벡터를 하나의 행렬로 굳힌다. save() 전에 자동 호출됨."""
        if self._vectors:
            self._matrix = np.concatenate(self._vectors, axis=0)
            self._vectors = []

    def save(self, path: Path) -> None:
        self.build()
        path.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.save(path / "vectors.npy", self._matrix)
        with (path / "meta.jsonl").open("w", encoding="utf-8") as f:
            for i, doc, meta in zip(self.ids, self.documents, self.metadatas):
                f.write(json.dumps({"id": i, "document": doc, "metadata": meta}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        store = cls()
        store._matrix = np.load(path / "vectors.npy")
        with (path / "meta.jsonl").open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                store.ids.append(rec["id"])
                store.documents.append(rec["document"])
                store.metadatas.append(rec["metadata"])
        return store

    def __len__(self) -> int:
        return len(self.ids)

    def query(self, query_vector: np.ndarray, n_results: int) -> QueryResult:
        """단일 질의 벡터(정규화됨) 대비 코사인 유사도 top-n. 벡터가 없으면 빈 결과."""
        if self._matrix is None or len(self.ids) == 0:
            return QueryResult(ids=[], documents=[], metadatas=[], scores=[])

        q = np.asarray(query_vector, dtype=np.float32)
        scores = self._matrix @ q  # 둘 다 정규화됨 -> 코사인 유사도
        k = min(n_results, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return QueryResult(
            ids=[self.ids[i] for i in top_idx],
            documents=[self.documents[i] for i in top_idx],
            metadatas=[self.metadatas[i] for i in top_idx],
            scores=[float(scores[i]) for i in top_idx],
        )
