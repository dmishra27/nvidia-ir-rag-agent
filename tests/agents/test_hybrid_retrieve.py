"""Unit tests for agents/hybrid_retrieve.py — the BM25 + dense -> RRF pass
shared by retrieval_agent and qa_agent.

These own the branch coverage for the degradation contract (dense failure ->
BM25-only, BM25 failure -> hard error, dense_index None -> deliberate skip).
tests/agents/test_retrieval_agent.py and tests/agents/test_qa_agent.py keep
their own retrieve-node tests, which now double as checks that each node maps
HybridResult onto its state shape correctly.
"""

from __future__ import annotations

from unittest.mock import patch

import structlog

from agents.hybrid_retrieve import HybridResult, hybrid_retrieve
from retrieval.candidates import Candidate


def _c(chunk_id: str, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=f"text {chunk_id}", score=1.0, rank=rank)


class _FakeBM25:
    def __init__(self, results: list[Candidate], calls: list) -> None:
        self._results = results
        self.calls = calls

    def search(self, query: str, top_k: int) -> list[Candidate]:
        self.calls.append(("bm25", query, top_k))
        return self._results


class _FakeDense:
    def __init__(self, results: list[Candidate], calls: list) -> None:
        self._results = results
        self.calls = calls

    def search(self, query: str, top_k: int) -> list[Candidate]:
        self.calls.append(("dense", query, top_k))
        return self._results


class _RaisingBM25:
    def search(self, query: str, top_k: int) -> list[Candidate]:
        raise RuntimeError("bm25 index unavailable")


class _RaisingDense:
    def search(self, query: str, top_k: int) -> list[Candidate]:
        raise RuntimeError("404 Collection 'nvidia_ir_chunks' doesn't exist!")


def _call(bm25: object, dense: object, *, query: str = "q", pool_size: int = 100) -> HybridResult:
    return hybrid_retrieve(
        query, "qid00001", bm25_index=bm25, dense_index=dense, pool_size=pool_size  # type: ignore[arg-type]
    )


def test_hybrid_result_defaults() -> None:
    r = HybridResult()
    assert r.bm25_results == []
    assert r.dense_results == []
    assert r.fused_results == []
    assert r.error is None


def test_happy_path_fuses_bm25_and_dense() -> None:
    calls: list = []
    result = _call(_FakeBM25([_c("b1", 1)], calls), _FakeDense([_c("d1", 1)], calls))

    assert result.error is None
    assert result.bm25_results == [_c("b1", 1)]
    assert result.dense_results == [_c("d1", 1)]
    assert {c.chunk_id for c in result.fused_results} == {"b1", "d1"}


def test_pool_size_is_passed_as_top_k_to_both_signals() -> None:
    calls: list = []
    _call(_FakeBM25([], calls), _FakeDense([], calls), pool_size=42)

    assert ("bm25", "q", 42) in calls
    assert ("dense", "q", 42) in calls


def test_none_dense_index_is_a_deliberate_skip_not_a_failure() -> None:
    calls: list = []
    result = _call(_FakeBM25([_c("b1", 1)], calls), None)

    assert result.error is None
    assert result.dense_results == []
    assert [c.chunk_id for c in result.fused_results] == ["b1"]
    assert not any(c[0] == "dense" for c in calls)


def test_dense_failure_degrades_to_bm25_only() -> None:
    """F-14: a dense-search exception must not discard the BM25 half."""
    calls: list = []
    result = _call(_FakeBM25([_c("b1", 1), _c("b2", 2)], calls), _RaisingDense())

    assert result.error is None
    assert result.bm25_results == [_c("b1", 1), _c("b2", 2)]
    assert result.dense_results == []
    assert [c.chunk_id for c in result.fused_results] == ["b1", "b2"]


def test_dense_failure_logs_dense_retrieval_degraded_warning() -> None:
    with structlog.testing.capture_logs() as logs:
        _call(_FakeBM25([_c("b1", 1)], []), _RaisingDense())

    degraded = [e for e in logs if e["event"] == "dense_retrieval_degraded"]
    assert len(degraded) == 1
    assert degraded[0]["log_level"] == "warning"
    assert degraded[0]["degraded_to"] == "bm25_only"
    assert "doesn't exist" in degraded[0]["exc"]


def test_bm25_failure_is_a_hard_error() -> None:
    """BM25 is the floor: nothing to degrade to, so error is set and the
    result carries no partial data."""
    result = _call(_RaisingBM25(), _FakeDense([_c("d1", 1)], []))

    assert result.error == "bm25 index unavailable"
    assert result.bm25_results == []
    assert result.dense_results == []
    assert result.fused_results == []


def test_bm25_failure_short_circuits_before_dense() -> None:
    calls: list = []
    dense = _FakeDense([_c("d1", 1)], calls)
    _call(_RaisingBM25(), dense)

    assert calls == []  # dense never queried once BM25 raised


def test_fusion_failure_is_a_hard_error() -> None:
    with patch("agents.hybrid_retrieve.fuse", side_effect=RuntimeError("rrf blew up")):
        result = _call(_FakeBM25([_c("b1", 1)], []), _FakeDense([_c("d1", 1)], []))

    assert result.error == "rrf blew up"
    assert result.fused_results == []
