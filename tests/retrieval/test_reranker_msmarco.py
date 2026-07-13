"""Unit tests for retrieval/reranker_msmarco.py.

Per AGENTS.md ("Mock all embedding and LLM calls in unit tests"), these
tests never load sentence-transformers/torch. A fake CrossEncoderProtocol
records the (query, text) pairs it was asked to score and returns a
scripted score list, standing in for the real cross-encoder model.
"""

from __future__ import annotations

import pytest

from retrieval.candidates import Candidate
from retrieval.reranker_msmarco import MSMarcoReranker


class _FakeCrossEncoder:
    """Records every predict() call and returns scripted scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        return self._scores[: len(pairs)]


def _c(chunk_id: str, text: str, score: float = 1.0, rank: int = 1) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=text, score=score, rank=rank)


_CANDIDATES = [
    _c("c1", "cudaStreamAddCallback note.", score=11.1, rank=1),
    _c("c2", "cudaMalloc allocates device memory.", score=9.5, rank=2),
    _c("c3", "cudaFree releases device memory.", score=8.2, rank=3),
]


# ---------------------------------------------------------------------------
# rerank() — happy path / re-ordering
# ---------------------------------------------------------------------------


def test_rerank_reorders_by_cross_encoder_score_descending() -> None:
    # Cross-encoder disagrees with the input order: c2 is actually most relevant.
    fake = _FakeCrossEncoder([0.2, 0.9, 0.5])
    reranker = MSMarcoReranker(fake)

    results = reranker.rerank("cudaMalloc parameters", _CANDIDATES, top_k=3)

    assert [r.chunk_id for r in results] == ["c2", "c3", "c1"]


def test_rerank_scores_are_cross_encoder_scores() -> None:
    fake = _FakeCrossEncoder([0.2, 0.9, 0.5])
    reranker = MSMarcoReranker(fake)

    results = reranker.rerank("query", _CANDIDATES, top_k=3)

    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.5)
    assert results[2].score == pytest.approx(0.2)


def test_rerank_ranks_are_sequential_starting_at_one() -> None:
    fake = _FakeCrossEncoder([0.1, 0.2, 0.3])
    reranker = MSMarcoReranker(fake)

    results = reranker.rerank("query", _CANDIDATES, top_k=3)

    assert [r.rank for r in results] == [1, 2, 3]


def test_rerank_preserves_chunk_id_and_text() -> None:
    fake = _FakeCrossEncoder([0.9, 0.1, 0.1])
    reranker = MSMarcoReranker(fake)

    results = reranker.rerank("query", _CANDIDATES, top_k=1)

    assert results[0].chunk_id == "c1"
    assert results[0].text == "cudaStreamAddCallback note."


# ---------------------------------------------------------------------------
# rerank() — query paired with each candidate's raw text
# ---------------------------------------------------------------------------


def test_rerank_builds_query_text_pairs_for_every_candidate() -> None:
    fake = _FakeCrossEncoder([0.1, 0.2, 0.3])
    reranker = MSMarcoReranker(fake)

    reranker.rerank("cudaMalloc parameters", _CANDIDATES, top_k=3)

    assert fake.calls == [
        [
            ("cudaMalloc parameters", "cudaStreamAddCallback note."),
            ("cudaMalloc parameters", "cudaMalloc allocates device memory."),
            ("cudaMalloc parameters", "cudaFree releases device memory."),
        ]
    ]


# ---------------------------------------------------------------------------
# rerank() — top_k boundary
# ---------------------------------------------------------------------------


def test_rerank_top_k_limits_result_count() -> None:
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])
    reranker = MSMarcoReranker(fake)

    results = reranker.rerank("query", _CANDIDATES, top_k=2)

    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["c2", "c3"]


def test_rerank_top_k_zero_returns_empty_without_calling_cross_encoder() -> None:
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])
    reranker = MSMarcoReranker(fake)

    assert reranker.rerank("query", _CANDIDATES, top_k=0) == []
    assert fake.calls == []


def test_rerank_top_k_negative_returns_empty_without_calling_cross_encoder() -> None:
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])
    reranker = MSMarcoReranker(fake)

    assert reranker.rerank("query", _CANDIDATES, top_k=-1) == []
    assert fake.calls == []


# ---------------------------------------------------------------------------
# rerank() — empty candidates
# ---------------------------------------------------------------------------


def test_rerank_empty_candidates_returns_empty_without_calling_cross_encoder() -> None:
    fake = _FakeCrossEncoder([])
    reranker = MSMarcoReranker(fake)

    assert reranker.rerank("query", [], top_k=5) == []
    assert fake.calls == []


# ---------------------------------------------------------------------------
# rerank() — top_k larger than candidate pool
# ---------------------------------------------------------------------------


def test_rerank_top_k_larger_than_pool_returns_all_candidates() -> None:
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])
    reranker = MSMarcoReranker(fake)

    results = reranker.rerank("query", _CANDIDATES, top_k=100)

    assert len(results) == 3
