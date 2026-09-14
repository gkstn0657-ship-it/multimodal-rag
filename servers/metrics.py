"""단계별 소요시간 계측 (Phase 5).

/ask 요청마다 검색/리랭킹/생성 단계 시간을 로깅하고, 평가 실행 시
p50/p95를 report.json에 포함할 수 있도록 최근 기록을 메모리에 보관한다.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class RequestTiming:
    retrieve_sec: float = 0.0
    rerank_sec: float = 0.0
    generate_sec: float = 0.0
    total_sec: float = 0.0

    def as_dict(self) -> dict:
        return {
            "retrieve_sec": round(self.retrieve_sec, 3),
            "rerank_sec": round(self.rerank_sec, 3),
            "generate_sec": round(self.generate_sec, 3),
            "total_sec": round(self.total_sec, 3),
        }


_history: list[RequestTiming] = []


@contextmanager
def stopwatch():
    t0 = time.time()
    yield lambda: time.time() - t0


def record(timing: RequestTiming) -> None:
    _history.append(timing)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def summary() -> dict:
    totals = [t.total_sec for t in _history]
    return {
        "count": len(_history),
        "p50_sec": round(percentile(totals, 0.50), 3),
        "p95_sec": round(percentile(totals, 0.95), 3),
    }
