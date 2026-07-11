"""Unit tests for retrieval/bm25_index.py.

BM25Index is built directly from in-memory chunk_ids/texts lists (no DB
required for these tests) — from_postgres() is a thin adapter tested
separately via live integration, not unit tests.
"""

from __future__ import annotations

import pytest

from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate

# ---------------------------------------------------------------------------
# Fixtures — small synthetic corpus
# ---------------------------------------------------------------------------

_CHUNK_IDS = ["c1", "c2", "c3", "c4", "c5"]
_TEXTS = [
    "NVLink bandwidth reaches 900 GB/s between GPUs in an NVLink domain.",
    "CUDA kernels are launched with a grid of thread blocks on the GPU.",
    "TensorRT optimises deep learning inference by fusing layers.",
    "The H100 Tensor Core GPU delivers up to 4 petaFLOPS of FP8 performance.",
    "Shared memory within a CUDA block enables low-latency communication.",
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index(list(_CHUNK_IDS), list(_TEXTS))


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        BM25Index(["c1", "c2"], ["only one text"])


# ---------------------------------------------------------------------------
# search() — exact match ranking
# ---------------------------------------------------------------------------


def test_exact_match_ranks_first(index: BM25Index) -> None:
    results = index.search("NVLink bandwidth", top_k=5)
    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1


def test_exact_match_cuda_ranks_first(index: BM25Index) -> None:
    results = index.search("CUDA memory allocation", top_k=5)
    assert results[0].chunk_id in {"c2", "c5"}
    assert results[0].rank == 1


def test_results_are_candidates_sorted_by_score_desc(index: BM25Index) -> None:
    results = index.search("GPU", top_k=5)
    assert all(isinstance(r, Candidate) for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ranks_are_sequential_starting_at_one(index: BM25Index) -> None:
    results = index.search("GPU", top_k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


# ---------------------------------------------------------------------------
# search() — top_k boundary
# ---------------------------------------------------------------------------


def test_top_k_limits_result_count(index: BM25Index) -> None:
    results = index.search("GPU", top_k=2)
    assert len(results) == 2


def test_top_k_larger_than_corpus_returns_all(index: BM25Index) -> None:
    results = index.search("GPU", top_k=1000)
    assert len(results) == len(_TEXTS)


def test_top_k_zero_returns_empty(index: BM25Index) -> None:
    assert index.search("GPU", top_k=0) == []


def test_top_k_negative_returns_empty(index: BM25Index) -> None:
    assert index.search("GPU", top_k=-1) == []


# ---------------------------------------------------------------------------
# search() — empty / degenerate queries
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty_list(index: BM25Index) -> None:
    assert index.search("", top_k=5) == []


def test_whitespace_only_query_returns_empty_list(index: BM25Index) -> None:
    assert index.search("   ", top_k=5) == []


def test_punctuation_only_query_returns_empty_list(index: BM25Index) -> None:
    assert index.search("???!!!", top_k=5) == []


def test_query_with_no_corpus_matches_returns_scored_results(index: BM25Index) -> None:
    # Terms absent from the corpus still tokenize, so all docs score 0.0 —
    # BM25 has no notion of "no match", just zero relevance.
    results = index.search("zzz_absent_term_xyz", top_k=5)
    assert len(results) == len(_TEXTS)
    assert all(r.score == 0.0 for r in results)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(index: BM25Index, tmp_path) -> None:
    path = tmp_path / "bm25_index.pkl"
    index.save(path)
    assert path.exists()

    loaded = BM25Index.load(path)
    original_results = index.search("NVLink bandwidth", top_k=3)
    loaded_results = loaded.search("NVLink bandwidth", top_k=3)

    assert loaded_results == original_results


def test_save_creates_parent_directories(index: BM25Index, tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "bm25_index.pkl"
    index.save(path)
    assert path.exists()
