"""BM25 어휘 검색 인덱스 (D-20, 하이브리드 검색용).

D-19의 DART 실패(회사명·연도가 다른 페이지가 상위에 옴)는 밀집 임베딩이 고유명사·숫자의 정확한 일치를
보장하지 않아서다. 벡터 저장소의 같은 텍스트(text_for_embedding)에 BM25 인덱스를 하나 더 만들고,
검색 시 두 결과를 RRF로 합친다(servers/retrieve.py).

의존성 없이 numpy로 구현한다. 39,458문서에서 질의당 수 ms.
- 토큰: 소문자, 한글·영숫자 연속 구간을 단어로 분리. 한글 단어는 흔한 조사를 하나 떼어낸 형태도 함께 넣는다
  ("삼성전자의" → "삼성전자의", "삼성전자"). 숫자+단위("2023년")는 숫자만도 함께 넣는다.
  형태소 분석기를 쓰지 않는 이유: 추가 의존성 없이, 우리가 노리는 실패(회사명·연도 일치)에는 이 정도로 충분하다.
- 저장: <store_dir>/bm25.npz + bm25_vocab.json. 벡터 저장소(vectors.npy, meta.jsonl)와 같은 디렉터리에 둔다.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[0-9]+|[a-z]+|[가-힣]+")
_NUMBER_UNIT_RE = re.compile(r"^([0-9]+)([가-힣]+)$")
# 길이순으로 긴 조사를 먼저 시도한다.
_PARTICLES = ("에서는", "으로는", "에서", "으로", "에는", "부터", "까지", "처럼", "보다", "와", "과", "의", "은", "는", "이", "가", "을", "를", "에", "로", "도", "만")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    text = text.lower()
    out: list[str] = []
    # "2023년"처럼 숫자와 한글이 붙은 경우 정규식이 둘로 나누므로 먼저 원형을 잡아 숫자만도 넣는다.
    for raw in re.findall(r"[0-9a-z가-힣]+", text):
        m = _NUMBER_UNIT_RE.match(raw)
        if m:
            out.append(raw)
            out.append(m.group(1))
            continue
        for tok in _TOKEN_RE.findall(raw):
            out.append(tok)
            if len(tok) > 2 and "가" <= tok[0] <= "힣":
                for p in _PARTICLES:
                    if tok.endswith(p) and len(tok) - len(p) >= 2:
                        out.append(tok[: -len(p)])
                        break
    return out


class BM25Index:
    """CSR 형태(term 기준 정렬)로 미리 계산한 BM25 가중치. query()는 질의 term의 포스팅을 모아 문서 점수를 합한다."""

    def __init__(self, vocab: dict[str, int], term_ptr: np.ndarray, doc_idx: np.ndarray, weights: np.ndarray, n_docs: int) -> None:
        self.vocab = vocab
        self.term_ptr = term_ptr  # len(vocab)+1, term t의 포스팅은 [term_ptr[t], term_ptr[t+1])
        self.doc_idx = doc_idx
        self.weights = weights
        self.n_docs = n_docs

    @classmethod
    def build(cls, documents: list[str]) -> "BM25Index":
        n = len(documents)
        vocab: dict[str, int] = {}
        doc_lists: list[list[int]] = []
        term_lists: list[list[int]] = []
        tf_lists: list[list[int]] = []
        dl = np.zeros(n, dtype=np.float32)
        for i, doc in enumerate(documents):
            counts = Counter(tokenize(doc))
            dl[i] = sum(counts.values())
            ts, fs = [], []
            for tok, c in counts.items():
                ts.append(vocab.setdefault(tok, len(vocab)))
                fs.append(c)
            term_lists.append(ts)
            tf_lists.append(fs)
            doc_lists.append([i] * len(ts))
        term_arr = np.fromiter((t for ts in term_lists for t in ts), dtype=np.int32)
        doc_arr = np.fromiter((d for ds in doc_lists for d in ds), dtype=np.int32)
        tf_arr = np.fromiter((f for fs in tf_lists for f in fs), dtype=np.float32)
        del term_lists, doc_lists, tf_lists

        order = np.argsort(term_arr, kind="stable")
        term_arr, doc_arr, tf_arr = term_arr[order], doc_arr[order], tf_arr[order]
        df = np.bincount(term_arr, minlength=len(vocab)).astype(np.float32)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avgdl = float(dl.mean()) if n else 1.0
        denom = tf_arr + K1 * (1.0 - B + B * dl[doc_arr] / avgdl)
        weights = (idf[term_arr] * tf_arr * (K1 + 1.0) / denom).astype(np.float32)
        term_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        term_ptr[1:] = np.cumsum(df).astype(np.int64)
        return cls(vocab, term_ptr, doc_arr, weights, n)

    def query(self, text: str, n_results: int) -> tuple[np.ndarray, np.ndarray]:
        """(doc indices, scores) 내림차순. 점수 0인 문서는 반환하지 않는다."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for tok in set(tokenize(text)):
            t = self.vocab.get(tok)
            if t is None:
                continue
            s, e = self.term_ptr[t], self.term_ptr[t + 1]
            np.add.at(scores, self.doc_idx[s:e], self.weights[s:e])
        k = min(n_results, self.n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        top = top[scores[top] > 0]
        return top, scores[top]

    def save(self, store_dir: Path) -> None:
        np.savez(store_dir / "bm25.npz", term_ptr=self.term_ptr, doc_idx=self.doc_idx, weights=self.weights, n_docs=np.int64(self.n_docs))
        (store_dir / "bm25_vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, store_dir: Path) -> "BM25Index":
        z = np.load(store_dir / "bm25.npz")
        vocab = json.loads((store_dir / "bm25_vocab.json").read_text(encoding="utf-8"))
        return cls(vocab, z["term_ptr"], z["doc_idx"], z["weights"], int(z["n_docs"]))

    @staticmethod
    def exists(store_dir: Path) -> bool:
        return (store_dir / "bm25.npz").exists() and (store_dir / "bm25_vocab.json").exists()


def build_for_store(store_dir: Path) -> BM25Index:
    """벡터 저장소의 meta.jsonl 문서 텍스트로 BM25 인덱스를 만들어 같은 디렉터리에 저장한다."""
    docs = [json.loads(l)["document"] for l in (store_dir / "meta.jsonl").open(encoding="utf-8")]
    index = BM25Index.build(docs)
    index.save(store_dir)
    return index


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("store_dirs", nargs="+")
    args = parser.parse_args()
    for d in args.store_dirs:
        t0 = time.time()
        idx = build_for_store(Path(d))
        print(f"{d}: 문서 {idx.n_docs:,} / 어휘 {len(idx.vocab):,} / 포스팅 {len(idx.doc_idx):,} / {time.time() - t0:.1f}초")
