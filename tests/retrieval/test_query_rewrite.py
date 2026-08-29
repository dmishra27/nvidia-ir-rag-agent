"""Unit tests for retrieval/query_rewrite.py (hypothesis D-QR).

Pure string logic — deterministic, no retrieval involved. The 15 Round 2
queries are the reference set the D-QR evaluation runs against, so the
classify/gate behaviour on each of them is pinned here.
"""

from __future__ import annotations

import pytest

from retrieval.query_rewrite import classify, rewrite_query

# ---------------------------------------------------------------------------
# classify — the gate's decision
# ---------------------------------------------------------------------------

EXACT_IDENTIFIER_QUERIES = [
    "CUDA cudaMalloc function parameters",           # Q1
    "cudaMemcpyAsync stream parameter",              # Q2
    "CUDA error cudaErrorInvalidValue description",  # Q3
    "cudaDeviceSynchronize return value",            # Q12
    "dim3 struct constructor syntax",                # Q13
    "pinned memory cudaMallocHost benefits and when to use",  # Q14 — mixed, gate still fires
]

CONCEPTUAL_QUERIES = [
    "shader processor count per streaming multiprocessor",   # Q4
    "how to make GPU programs run faster",                   # Q5
    "problems with threads executing different code paths",  # Q6
    "CUDA thread synchronization performance overhead",      # Q7
    "shared memory bank conflicts and how to avoid them",    # Q8
    "memory coalescing rules for global memory access patterns",  # Q9
    "latency hiding through instruction level parallelism",  # Q10
    "occupancy versus performance tradeoffs",                # Q11
    "register pressure and its effect on occupancy",         # Q15
]


@pytest.mark.parametrize("query", EXACT_IDENTIFIER_QUERIES)
def test_classify_exact_identifier(query: str) -> None:
    assert classify(query) == "exact_identifier"


@pytest.mark.parametrize("query", CONCEPTUAL_QUERIES)
def test_classify_conceptual(query: str) -> None:
    assert classify(query) == "conceptual"


def test_bare_cuda_word_is_not_an_identifier() -> None:
    # "CUDA" alone must not trip the cu[A-Z] driver-API branch.
    assert classify("CUDA memory model overview") == "conceptual"


# ---------------------------------------------------------------------------
# gate=True — the conditional arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", EXACT_IDENTIFIER_QUERIES)
def test_gated_leaves_identifier_queries_untouched(query: str) -> None:
    r = rewrite_query(query, gate=True)

    assert r.rewritten is False
    assert r.strategy == "gated-skip"
    assert r.bm25_query == query
    assert r.dense_query == query


def test_gated_expands_the_documented_vocabulary_gap() -> None:
    r = rewrite_query("shader processor count per streaming multiprocessor", gate=True)

    assert r.rewritten is True
    assert r.bm25_query == "shader processor count per streaming multiprocessor"  # literal
    assert "CUDA core" in r.dense_query
    assert r.dense_query.startswith("shader processor count per streaming multiprocessor ")


def test_gated_is_a_noop_when_no_legacy_term_present() -> None:
    r = rewrite_query("occupancy versus performance tradeoffs", gate=True)

    assert r.rewritten is False
    assert r.strategy == "no-op"
    assert r.dense_query == "occupancy versus performance tradeoffs"


# ---------------------------------------------------------------------------
# gate=False — the ungated arm
# ---------------------------------------------------------------------------


def test_ungated_splits_camelcase_identifiers_on_the_dense_side_only() -> None:
    r = rewrite_query("cudaDeviceSynchronize return value", gate=False)

    assert r.rewritten is True
    assert r.strategy == "identifier-split"
    assert r.bm25_query == "cudaDeviceSynchronize return value"  # BM25 keeps the literal token
    assert "cuda device synchronize" in r.dense_query


def test_ungated_and_gated_diverge_only_on_identifier_queries() -> None:
    identifier_q = "CUDA cudaMalloc function parameters"
    conceptual_q = "shader processor count per streaming multiprocessor"

    assert rewrite_query(identifier_q, gate=True).dense_query != rewrite_query(
        identifier_q, gate=False
    ).dense_query
    assert (
        rewrite_query(conceptual_q, gate=True).dense_query
        == rewrite_query(conceptual_q, gate=False).dense_query
    )


def test_legacy_map_covers_streaming_processor_synonym() -> None:
    r = rewrite_query("streaming processor throughput", gate=True)

    assert "CUDA core" in r.dense_query
    assert r.strategy == "legacy-expansion"


def test_expansion_is_deduped() -> None:
    r = rewrite_query("shader processor and shading unit differences", gate=True)

    # both legacy terms map to "CUDA core" — appended once, not twice
    assert r.dense_query.count("CUDA core") == 1
