"""Unit tests for retrieval/rrf_fusion.py.

Pure function over Candidate lists — no mocking needed, RRF math is
deterministic given the input ranks.
"""

from __future__ import annotations

import pytest

from retrieval.candidates import Candidate
from retrieval.rrf_fusion import DEFAULT_K, fuse


def _c(chunk_id: str, text: str, score: float, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=text, score=score, rank=rank)


# ---------------------------------------------------------------------------
# Score math
# ---------------------------------------------------------------------------


def test_doc_in_both_lists_sums_reciprocal_ranks() -> None:
    bm25 = [_c("c1", "nvlink text", 5.0, rank=1)]
    dense = [_c("c1", "nvlink text", 0.9, rank=1)]

    results = fuse(bm25, dense, top_k=5)

    expected = 1.0 / (DEFAULT_K + 1) + 1.0 / (DEFAULT_K + 1)
    assert results[0].chunk_id == "c1"
    assert results[0].score == pytest.approx(expected)


def test_doc_in_only_bm25_scores_single_reciprocal_rank() -> None:
    bm25 = [_c("c1", "text", 5.0, rank=3)]
    dense: list[Candidate] = []

    results = fuse(bm25, dense, top_k=5)

    assert results[0].score == pytest.approx(1.0 / (DEFAULT_K + 3))


def test_doc_in_only_dense_scores_single_reciprocal_rank() -> None:
    bm25: list[Candidate] = []
    dense = [_c("c1", "text", 0.7, rank=2)]

    results = fuse(bm25, dense, top_k=5)

    assert results[0].score == pytest.approx(1.0 / (DEFAULT_K + 2))


def test_custom_k_changes_score() -> None:
    bm25 = [_c("c1", "text", 5.0, rank=1)]
    dense: list[Candidate] = []

    results = fuse(bm25, dense, top_k=5, k=10)

    assert results[0].score == pytest.approx(1.0 / (10 + 1))


# ---------------------------------------------------------------------------
# Ranking / ordering
# ---------------------------------------------------------------------------


def test_doc_in_both_lists_outranks_doc_in_one_list() -> None:
    bm25 = [_c("c1", "both", 5.0, rank=1), _c("c2", "bm25 only", 4.0, rank=2)]
    dense = [_c("c1", "both", 0.9, rank=1)]

    results = fuse(bm25, dense, top_k=5)

    assert [r.chunk_id for r in results] == ["c1", "c2"]


def test_ranks_are_sequential_starting_at_one() -> None:
    bm25 = [_c("c1", "a", 5.0, rank=1), _c("c2", "b", 4.0, rank=2), _c("c3", "c", 3.0, rank=3)]
    dense: list[Candidate] = []

    results = fuse(bm25, dense, top_k=5)

    assert [r.rank for r in results] == [1, 2, 3]


def test_results_sorted_by_rrf_score_descending() -> None:
    bm25 = [_c("low", "x", 1.0, rank=5), _c("high", "y", 2.0, rank=1)]
    dense: list[Candidate] = []

    results = fuse(bm25, dense, top_k=5)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk_id == "high"


def test_ties_break_by_first_seen_order_bm25_before_dense() -> None:
    # Both rank 1 in a single-signal list each -> equal RRF score; bm25 order wins the tie.
    bm25 = [_c("from_bm25", "x", 1.0, rank=1)]
    dense = [_c("from_dense", "y", 1.0, rank=1)]

    results = fuse(bm25, dense, top_k=5)

    assert [r.chunk_id for r in results] == ["from_bm25", "from_dense"]


# ---------------------------------------------------------------------------
# top_k boundary
# ---------------------------------------------------------------------------


def test_top_k_limits_result_count() -> None:
    bm25 = [_c(f"c{i}", "x", float(i), rank=i) for i in range(1, 6)]
    dense: list[Candidate] = []

    results = fuse(bm25, dense, top_k=2)

    assert len(results) == 2


def test_top_k_larger_than_union_returns_all_unique_docs() -> None:
    bm25 = [_c("c1", "x", 1.0, rank=1), _c("c2", "y", 1.0, rank=2)]
    dense = [_c("c1", "x", 1.0, rank=1), _c("c3", "z", 1.0, rank=1)]

    results = fuse(bm25, dense, top_k=100)

    assert {r.chunk_id for r in results} == {"c1", "c2", "c3"}
    assert len(results) == 3


def test_top_k_zero_returns_empty() -> None:
    bm25 = [_c("c1", "x", 1.0, rank=1)]
    assert fuse(bm25, [], top_k=0) == []


def test_top_k_negative_returns_empty() -> None:
    bm25 = [_c("c1", "x", 1.0, rank=1)]
    assert fuse(bm25, [], top_k=-1) == []


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_both_lists_empty_returns_empty() -> None:
    assert fuse([], [], top_k=5) == []


# ---------------------------------------------------------------------------
# Text carries through from the fused candidate's first-seen occurrence
# ---------------------------------------------------------------------------


def test_text_taken_from_bm25_when_present_in_both() -> None:
    bm25 = [_c("c1", "bm25 text", 5.0, rank=1)]
    dense = [_c("c1", "dense text", 0.9, rank=1)]

    results = fuse(bm25, dense, top_k=5)

    assert results[0].text == "bm25 text"


def test_text_taken_from_dense_when_bm25_only_has_other_docs() -> None:
    bm25 = [_c("c2", "other", 1.0, rank=1)]
    dense = [_c("c1", "dense text", 0.9, rank=1)]

    results = fuse(bm25, dense, top_k=5)

    dense_result = next(r for r in results if r.chunk_id == "c1")
    assert dense_result.text == "dense text"
