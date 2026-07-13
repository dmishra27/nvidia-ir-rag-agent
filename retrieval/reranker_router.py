"""Routes a query to a re-ranking tier selected by the RERANKER_MODE env var.

Per AGENTS.md, RERANKER_MODE has five values:
  live_fast     -> ms-marco-MiniLM-L-6-v2 (default, <1s CPU)
  live_quality  -> bge-reranker-v2-m3 via sentence-transformers
  live_frontier -> Cohere Rerank v3 API
  benchmark     -> all three run in parallel, results to MLflow
  fallback      -> BM25 rank order, no re-ranking

Four of these (live_frontier, live_quality, live_fast, fallback) form an
ordered serving chain, highest quality first: if the configured tier's
reranker is unavailable (not wired up yet — only live_fast is built as of
Day 6) or raises, the router logs a warning and degrades to the next tier
down, terminating at `fallback`, which needs no model and never fails. Each
tier's reranker is an injected callable so unit tests are fully
deterministic and never load a real model, per the project's rule to mock
all embedding/LLM calls in tests. `benchmark` is not part of the
degradation chain — it is an explicit, non-degrading choice (see
SKILLS.md's run-reranker-benchmark) that requires its own runner.
"""

from __future__ import annotations

import os
from typing import Callable

import structlog

from retrieval.candidates import Candidate

log = structlog.get_logger()

RERANKER_MODE_ENV_VAR = "RERANKER_MODE"
DEFAULT_MODE = "live_fast"

# Highest quality/cost first, most robust last.
SERVING_CHAIN = ["live_frontier", "live_quality", "live_fast", "fallback"]

# (query, candidates, top_k, query_id) -> ranked candidates.
RerankFn = Callable[[str, list[Candidate], int, str], list[Candidate]]


def _fallback_rerank(
    query: str, candidates: list[Candidate], top_k: int, query_id: str
) -> list[Candidate]:
    """No re-ranking: keep the incoming (BM25/RRF) rank order, truncated to top_k."""
    if top_k <= 0:
        return []
    return [
        Candidate(chunk_id=c.chunk_id, text=c.text, score=c.score, rank=rank)
        for rank, c in enumerate(candidates[:top_k], start=1)
    ]


class RerankerRouter:
    """Dispatches to the configured RERANKER_MODE tier with graceful degradation."""

    def __init__(
        self,
        live_fast: RerankFn | None = None,
        live_quality: RerankFn | None = None,
        live_frontier: RerankFn | None = None,
        benchmark: RerankFn | None = None,
        mode: str | None = None,
    ) -> None:
        self._tiers: dict[str, RerankFn | None] = {
            "live_frontier": live_frontier,
            "live_quality": live_quality,
            "live_fast": live_fast,
            "fallback": _fallback_rerank,
        }
        self._benchmark = benchmark
        self._mode = mode or os.environ.get(RERANKER_MODE_ENV_VAR, DEFAULT_MODE)

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int = 10,
        query_id: str = "unknown",
    ) -> list[Candidate]:
        if self._mode == "benchmark":
            if self._benchmark is None:
                raise NotImplementedError(
                    "RERANKER_MODE=benchmark requires an injected benchmark runner"
                )
            log.info(
                "reranker_router_dispatch",
                stage="reranker_router",
                query_id=query_id,
                requested_mode="benchmark",
                tier_used="benchmark",
            )
            return self._benchmark(query, candidates, top_k, query_id)

        if self._mode not in SERVING_CHAIN:
            raise ValueError(f"Unknown RERANKER_MODE: {self._mode!r}")

        start = SERVING_CHAIN.index(self._mode)
        for tier in SERVING_CHAIN[start:]:
            reranker = self._tiers[tier]
            if reranker is None:
                log.warning(
                    "reranker_router_tier_unavailable",
                    stage="reranker_router",
                    query_id=query_id,
                    tier=tier,
                )
                continue
            try:
                result = reranker(query, candidates, top_k, query_id)
            except Exception as exc:
                log.warning(
                    "reranker_router_tier_failed",
                    stage="reranker_router",
                    query_id=query_id,
                    tier=tier,
                    exc=str(exc),
                )
                continue
            log.info(
                "reranker_router_dispatch",
                stage="reranker_router",
                query_id=query_id,
                requested_mode=self._mode,
                tier_used=tier,
            )
            return result

        # Unreachable under normal use: "fallback" is always wired and never raises.
        return _fallback_rerank(query, candidates, top_k, query_id)
