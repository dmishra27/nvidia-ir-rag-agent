"""Unit tests for retrieval/splade_index.py.

Per AGENTS.md ("Mock all embedding and LLM calls in unit tests"), these tests
never load torch/transformers. A tiny deterministic bag-of-words fake stands
in for the real SPLADE encoder — it maps texts to {token: term_frequency}
sparse vectors, which is enough to exercise ranking, persistence, and
boundary behavior without a neural forward pass.
"""

from __future__ import annotations

import re

import pytest

from retrieval.candidates import Candidate
from retrieval.splade_index import SparseEncoder, SpladeIndex

# ---------------------------------------------------------------------------
# Fixtures — small synthetic corpus + fake encoder
# ---------------------------------------------------------------------------

_CHUNK_IDS = ["c1", "c2", "c3", "c4", "c5"]
_TEXTS = [
    "NVLink bandwidth reaches 900 GB/s between GPUs in an NVLink domain.",
    "CUDA kernels are launched with a grid of thread blocks on the GPU.",
    "TensorRT optimises deep learning inference by fusing layers.",
    "The H100 Tensor Core GPU delivers up to 4 petaFLOPS of FP8 performance.",
    "Shared memory within a CUDA block enables low-latency communication.",
]


def _fake_encoder(texts: list[str]) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for text in texts:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        counts: dict[str, float] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0.0) + 1.0
        results.append(counts)
    return results


@pytest.fixture
def encoder() -> SparseEncoder:
    return _fake_encoder


@pytest.fixture
def index(encoder: SparseEncoder) -> SpladeIndex:
    return SpladeIndex.build(list(_CHUNK_IDS), list(_TEXTS), encoder)


# ---------------------------------------------------------------------------
# Constructor / build
# ---------------------------------------------------------------------------


def test_constructor_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        SpladeIndex(["c1", "c2"], ["only one text"], [{}])


def test_build_produces_one_vector_per_text(encoder: SparseEncoder) -> None:
    index = SpladeIndex.build(list(_CHUNK_IDS), list(_TEXTS), encoder)
    assert len(index._vectors) == len(_TEXTS)


def test_build_batches_calls_to_encoder(encoder: SparseEncoder) -> None:
    calls: list[int] = []

    def counting_encoder(texts: list[str]) -> list[dict[str, float]]:
        calls.append(len(texts))
        return encoder(texts)

    SpladeIndex.build(list(_CHUNK_IDS), list(_TEXTS), counting_encoder, batch_size=2)
    assert calls == [2, 2, 1]


# ---------------------------------------------------------------------------
# search() — exact match ranking
# ---------------------------------------------------------------------------


def test_exact_match_ranks_first(index: SpladeIndex) -> None:
    results = index.search("NVLink bandwidth", top_k=5)
    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1


def test_results_are_candidates_sorted_by_score_desc(index: SpladeIndex) -> None:
    results = index.search("GPU", top_k=5)
    assert all(isinstance(r, Candidate) for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ranks_are_sequential_starting_at_one(index: SpladeIndex) -> None:
    results = index.search("GPU", top_k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


# ---------------------------------------------------------------------------
# search() — top_k boundary
# ---------------------------------------------------------------------------


def test_top_k_limits_result_count(index: SpladeIndex) -> None:
    results = index.search("GPU", top_k=2)
    assert len(results) == 2


def test_top_k_larger_than_corpus_returns_all(index: SpladeIndex) -> None:
    results = index.search("GPU", top_k=1000)
    assert len(results) == len(_TEXTS)


def test_top_k_zero_returns_empty(index: SpladeIndex) -> None:
    assert index.search("GPU", top_k=0) == []


def test_top_k_negative_returns_empty(index: SpladeIndex) -> None:
    assert index.search("GPU", top_k=-1) == []


# ---------------------------------------------------------------------------
# search() — empty / degenerate queries
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty_list(index: SpladeIndex) -> None:
    assert index.search("", top_k=5) == []


def test_whitespace_only_query_returns_empty_list(index: SpladeIndex) -> None:
    assert index.search("   ", top_k=5) == []


def test_query_with_no_corpus_matches_returns_scored_results(index: SpladeIndex) -> None:
    results = index.search("zzz_absent_term_xyz", top_k=5)
    assert len(results) == len(_TEXTS)
    assert all(r.score == 0.0 for r in results)


# ---------------------------------------------------------------------------
# search() — missing encoder
# ---------------------------------------------------------------------------


def test_search_without_encoder_raises() -> None:
    bare = SpladeIndex(list(_CHUNK_IDS), list(_TEXTS), [{} for _ in _TEXTS])
    with pytest.raises(RuntimeError, match="no encoder attached"):
        bare.search("GPU", top_k=5)


def test_attach_encoder_enables_search(encoder: SparseEncoder) -> None:
    vectors = encoder(_TEXTS)
    bare = SpladeIndex(list(_CHUNK_IDS), list(_TEXTS), vectors)
    bare.attach_encoder(encoder)
    results = bare.search("NVLink bandwidth", top_k=5)
    assert results[0].chunk_id == "c1"


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(index: SpladeIndex, encoder: SparseEncoder, tmp_path) -> None:
    path = tmp_path / "splade_index.pkl"
    index.save(path)
    assert path.exists()

    loaded = SpladeIndex.load(path)
    loaded.attach_encoder(encoder)

    original_results = index.search("NVLink bandwidth", top_k=3)
    loaded_results = loaded.search("NVLink bandwidth", top_k=3)
    assert loaded_results == original_results


def test_load_without_attach_encoder_raises(index: SpladeIndex, tmp_path) -> None:
    path = tmp_path / "splade_index.pkl"
    index.save(path)
    loaded = SpladeIndex.load(path)
    with pytest.raises(RuntimeError, match="no encoder attached"):
        loaded.search("GPU", top_k=5)


def test_save_creates_parent_directories(index: SpladeIndex, tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "splade_index.pkl"
    index.save(path)
    assert path.exists()
