"""Unit tests for retrieval/reranker_router.py.

Every tier is an injected fake callable, so the router's degradation logic
is fully deterministic and never touches a real model or API.
"""

from __future__ import annotations

import pytest

from retrieval.candidates import Candidate
from retrieval.reranker_router import SERVING_CHAIN, RerankerRouter, _fallback_rerank


def _c(chunk_id: str, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=f"text {chunk_id}", score=1.0, rank=rank)


_CANDIDATES = [_c("c1", 1), _c("c2", 2), _c("c3", 3)]


def _make_tier(name: str, calls: list[str]):
    def tier(query: str, candidates: list[Candidate], top_k: int, query_id: str) -> list[Candidate]:
        calls.append(name)
        return [Candidate(chunk_id=f"{name}_result", text="x", score=1.0, rank=1)]

    return tier


def _raising_tier(name: str, calls: list[str]):
    def tier(query: str, candidates: list[Candidate], top_k: int, query_id: str) -> list[Candidate]:
        calls.append(name)
        raise RuntimeError(f"{name} failed")

    return tier


# ---------------------------------------------------------------------------
# Direct tier dispatch — no degradation needed
# ---------------------------------------------------------------------------


def test_live_fast_mode_calls_live_fast_tier() -> None:
    calls: list[str] = []
    router = RerankerRouter(live_fast=_make_tier("live_fast", calls), mode="live_fast")

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_fast"]
    assert result[0].chunk_id == "live_fast_result"


def test_live_quality_mode_calls_live_quality_tier() -> None:
    calls: list[str] = []
    router = RerankerRouter(live_quality=_make_tier("live_quality", calls), mode="live_quality")

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_quality"]
    assert result[0].chunk_id == "live_quality_result"


def test_live_frontier_mode_calls_live_frontier_tier() -> None:
    calls: list[str] = []
    router = RerankerRouter(live_frontier=_make_tier("live_frontier", calls), mode="live_frontier")

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_frontier"]
    assert result[0].chunk_id == "live_frontier_result"


def test_fallback_mode_returns_bm25_rank_order_without_any_injected_tier() -> None:
    router = RerankerRouter(mode="fallback")

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert [r.chunk_id for r in result] == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# Degradation — unavailable tiers (None) are skipped
# ---------------------------------------------------------------------------


def test_degrades_past_unavailable_tiers_to_first_available() -> None:
    calls: list[str] = []
    router = RerankerRouter(
        live_frontier=None,
        live_quality=None,
        live_fast=_make_tier("live_fast", calls),
        mode="live_frontier",
    )

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert result[0].chunk_id == "live_fast_result"


def test_degrades_all_the_way_to_fallback_when_nothing_wired() -> None:
    router = RerankerRouter(mode="live_frontier")

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert [r.chunk_id for r in result] == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# Degradation — an exception from a wired tier also degrades
# ---------------------------------------------------------------------------


def test_degrades_to_next_tier_when_configured_tier_raises() -> None:
    calls: list[str] = []
    router = RerankerRouter(
        live_quality=_raising_tier("live_quality", calls),
        live_fast=_make_tier("live_fast", calls),
        mode="live_quality",
    )

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_quality", "live_fast"]
    assert result[0].chunk_id == "live_fast_result"


def test_degrades_through_multiple_raising_tiers_to_fallback() -> None:
    calls: list[str] = []
    router = RerankerRouter(
        live_frontier=_raising_tier("live_frontier", calls),
        live_quality=_raising_tier("live_quality", calls),
        live_fast=_raising_tier("live_fast", calls),
        mode="live_frontier",
    )

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_frontier", "live_quality", "live_fast"]
    assert [r.chunk_id for r in result] == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# Serving chain never starts below the configured mode
# ---------------------------------------------------------------------------


def test_live_fast_mode_never_calls_higher_tiers() -> None:
    calls: list[str] = []
    router = RerankerRouter(
        live_frontier=_make_tier("live_frontier", calls),
        live_quality=_make_tier("live_quality", calls),
        live_fast=_make_tier("live_fast", calls),
        mode="live_fast",
    )

    router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_fast"]


# ---------------------------------------------------------------------------
# benchmark mode — not part of the degradation chain
# ---------------------------------------------------------------------------


def test_benchmark_mode_calls_injected_benchmark_runner() -> None:
    calls: list[str] = []
    router = RerankerRouter(benchmark=_make_tier("benchmark", calls), mode="benchmark")

    result = router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["benchmark"]
    assert result[0].chunk_id == "benchmark_result"


def test_benchmark_mode_without_runner_raises_not_implemented() -> None:
    router = RerankerRouter(mode="benchmark")

    with pytest.raises(NotImplementedError):
        router.rerank("query", _CANDIDATES, top_k=3)


def test_benchmark_mode_does_not_fall_back_to_serving_chain_on_missing_runner() -> None:
    calls: list[str] = []
    router = RerankerRouter(live_fast=_make_tier("live_fast", calls), mode="benchmark")

    with pytest.raises(NotImplementedError):
        router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == []


# ---------------------------------------------------------------------------
# Mode resolution — explicit param vs. env var vs. default
# ---------------------------------------------------------------------------


def test_explicit_mode_param_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_MODE", "fallback")
    calls: list[str] = []
    router = RerankerRouter(live_fast=_make_tier("live_fast", calls), mode="live_fast")

    router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_fast"]


def test_mode_read_from_env_var_when_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_MODE", "live_fast")
    calls: list[str] = []
    router = RerankerRouter(live_fast=_make_tier("live_fast", calls))

    router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_fast"]


def test_defaults_to_live_fast_when_no_mode_and_no_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RERANKER_MODE", raising=False)
    calls: list[str] = []
    router = RerankerRouter(live_fast=_make_tier("live_fast", calls))

    router.rerank("query", _CANDIDATES, top_k=3)

    assert calls == ["live_fast"]


def test_unknown_mode_raises_value_error() -> None:
    router = RerankerRouter(mode="not_a_real_mode")

    with pytest.raises(ValueError, match="Unknown RERANKER_MODE"):
        router.rerank("query", _CANDIDATES, top_k=3)


# ---------------------------------------------------------------------------
# Serving chain constant sanity
# ---------------------------------------------------------------------------


def test_serving_chain_ends_with_fallback() -> None:
    assert SERVING_CHAIN[-1] == "fallback"


def test_serving_chain_has_four_tiers() -> None:
    assert len(SERVING_CHAIN) == 4


# ---------------------------------------------------------------------------
# _fallback_rerank() — direct unit tests
# ---------------------------------------------------------------------------


def test_fallback_rerank_preserves_order_and_reassigns_sequential_ranks() -> None:
    unordered_ranks = [
        Candidate(chunk_id="a", text="a", score=1.0, rank=5),
        Candidate(chunk_id="b", text="b", score=1.0, rank=9),
    ]

    result = _fallback_rerank("query", unordered_ranks, top_k=2, query_id="q1")

    assert [r.rank for r in result] == [1, 2]
    assert [r.chunk_id for r in result] == ["a", "b"]


def test_fallback_rerank_top_k_limits_result_count() -> None:
    result = _fallback_rerank("query", _CANDIDATES, top_k=2, query_id="q1")

    assert len(result) == 2


def test_fallback_rerank_top_k_zero_returns_empty() -> None:
    assert _fallback_rerank("query", _CANDIDATES, top_k=0, query_id="q1") == []


def test_fallback_rerank_top_k_negative_returns_empty() -> None:
    assert _fallback_rerank("query", _CANDIDATES, top_k=-1, query_id="q1") == []
