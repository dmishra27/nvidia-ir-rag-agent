"""Unit tests for evaluation/retrieval_metrics.py.

Pure functions over Candidate lists and a binary relevance dict — no
mocking needed, all metric math is deterministic given ranks and labels.
Unjudged chunk_ids (absent from the relevance dict) are treated as
non-relevant (0), matching standard sparse-judgment IR evaluation
(TREC-style pooling) — this project's relevance_labels.jsonl is a sparse,
Claude-labelled first pass, not exhaustive per-query pooling.
"""

from __future__ import annotations

import math

import pytest

from evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k
from retrieval.candidates import Candidate


def _c(chunk_id: str, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=f"text {chunk_id}", score=1.0, rank=rank)


def _ranked(*chunk_ids: str) -> list[Candidate]:
    return [_c(cid, rank) for rank, cid in enumerate(chunk_ids, start=1)]


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"a": 1, "b": 1, "c": 0}

    assert ndcg_at_k(ranked, relevance, k=3) == pytest.approx(1.0)


def test_ndcg_single_relevant_doc_at_rank_one_is_one() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"a": 1}

    assert ndcg_at_k(ranked, relevance, k=3) == pytest.approx(1.0)


def test_ndcg_relevant_doc_buried_lower_scores_less_than_one() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"c": 1}

    result = ndcg_at_k(ranked, relevance, k=3)

    expected_dcg = 1.0 / math.log2(3 + 1)
    expected_idcg = 1.0 / math.log2(1 + 1)
    assert result == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_no_relevant_docs_at_all_is_zero() -> None:
    ranked = _ranked("a", "b", "c")

    assert ndcg_at_k(ranked, {}, k=3) == 0.0


def test_ndcg_relevant_doc_outside_k_is_zero() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"c": 1}

    assert ndcg_at_k(ranked, relevance, k=2) == 0.0


def test_ndcg_empty_ranking_is_zero() -> None:
    assert ndcg_at_k([], {"a": 1}, k=10) == 0.0


def test_ndcg_unjudged_chunk_ids_treated_as_non_relevant() -> None:
    ranked = _ranked("unjudged1", "b")
    relevance = {"b": 1}

    result = ndcg_at_k(ranked, relevance, k=2)

    expected_dcg = 1.0 / math.log2(2 + 1)
    expected_idcg = 1.0 / math.log2(1 + 1)
    assert result == pytest.approx(expected_dcg / expected_idcg)


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


def test_mrr_relevant_doc_at_rank_one() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"a": 1}

    assert mrr(ranked, relevance) == pytest.approx(1.0)


def test_mrr_relevant_doc_at_rank_three() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"c": 1}

    assert mrr(ranked, relevance) == pytest.approx(1.0 / 3.0)


def test_mrr_no_relevant_doc_is_zero() -> None:
    ranked = _ranked("a", "b", "c")

    assert mrr(ranked, {}) == 0.0


def test_mrr_empty_ranking_is_zero() -> None:
    assert mrr([], {"a": 1}) == 0.0


def test_mrr_uses_first_relevant_doc_only() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"b": 1, "c": 1}

    assert mrr(ranked, relevance) == pytest.approx(1.0 / 2.0)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_all_relevant() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"a": 1, "b": 1, "c": 1}

    assert precision_at_k(ranked, relevance, k=3) == pytest.approx(1.0)


def test_precision_at_k_partial_relevance() -> None:
    ranked = _ranked("a", "b", "c", "d")
    relevance = {"a": 1, "c": 1}

    assert precision_at_k(ranked, relevance, k=4) == pytest.approx(0.5)


def test_precision_at_k_only_considers_top_k() -> None:
    ranked = _ranked("a", "b", "c")
    relevance = {"c": 1}

    assert precision_at_k(ranked, relevance, k=2) == 0.0


def test_precision_at_k_truncates_when_fewer_results_than_k() -> None:
    ranked = _ranked("a", "b")
    relevance = {"a": 1, "b": 1}

    assert precision_at_k(ranked, relevance, k=10) == pytest.approx(1.0)


def test_precision_at_k_empty_ranking_is_zero() -> None:
    assert precision_at_k([], {"a": 1}, k=10) == 0.0


def test_precision_at_k_zero_k_is_zero() -> None:
    ranked = _ranked("a", "b")

    assert precision_at_k(ranked, {"a": 1}, k=0) == 0.0
