"""Shared BM25 + dense -> RRF retrieval for both LangGraph agents.

agents/retrieval_agent.py and agents/qa_agent.py had byte-for-byte identical
`retrieve` node bodies. That duplication is what let the F-14 dense-degradation
bug get fixed in retrieval_agent (commit 7f3107d) but not qa_agent — a second
commit (f832dcc) had to re-apply the same change by hand. This module is the
single implementation both node factories now wrap; each maps the result onto
its own state shape (AgentState keeps bm25/dense/fused; QAState keeps only
fused).

Behaviour is exactly what 7f3107d/f832dcc established — no change:

- BM25 always runs. A BM25 or fusion failure is a hard error: `error` is set
  and the caller aborts the pipeline (rerank/generate early-return on it).
- `dense_index is None` is a deliberate skip (RERANKER_MODE=fallback, see
  api/dependencies.py) — empty dense pool, not an error.
- `dense_index.search()` raising is degraded, not fatal: log
  `dense_retrieval_degraded`, use an empty dense pool, carry on. On a clean
  clone Qdrant's nvidia_ir_chunks collection doesn't exist until
  populate_qdrant.py runs (~86 min), and the committed BM25 index still
  answers correctly — discarding it would return nothing (and, on /ask, bill
  an Anthropic call for an answer grounded in nothing).

It lives in agents/ rather than retrieval/ because it needs api.telemetry's
`traced_stage`, and nothing under retrieval/ imports api/ — keeping the
dependency direction (api -> agents -> retrieval) intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from api.telemetry import traced_stage
from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.rrf_fusion import fuse

log = structlog.get_logger()


@dataclass(frozen=True)
class HybridResult:
    """Outcome of one hybrid-retrieval pass. On failure only `error` is set
    (the lists stay empty); on success `error` is None and the three lists
    are populated."""

    bm25_results: list[Candidate] = field(default_factory=list)
    dense_results: list[Candidate] = field(default_factory=list)
    fused_results: list[Candidate] = field(default_factory=list)
    error: str | None = None


def hybrid_retrieve(
    query: str,
    query_id: str,
    *,
    bm25_index: BM25Index,
    dense_index: DenseIndex | None,
    pool_size: int,
) -> HybridResult:
    try:
        with traced_stage("bm25", query_id, top_k=pool_size):
            bm25_results = bm25_index.search(query, top_k=pool_size)
        if dense_index is None:
            # RERANKER_MODE=fallback deliberately skipped loading DenseIndex
            # (api/dependencies.py, to fit Render's free-tier 512MB) -- that's
            # a configuration choice, not a failure, so BM25 results still get
            # fused/returned below.
            dense_results: list[Candidate] = []
        else:
            try:
                with traced_stage("dense", query_id, top_k=pool_size):
                    dense_results = dense_index.search(query, top_k=pool_size)
            except Exception as exc:
                # A dense-search failure must NOT discard the BM25 half, which
                # has its committed index (data/indexes/bm25_index.pkl) and
                # 5,389 chunks and answers correctly. Degrade to BM25-only:
                # log the degradation loudly, carry on with an empty dense
                # pool. Render is unaffected -- it runs RERANKER_MODE=fallback
                # and takes the dense_index is None branch above.
                log.warning(
                    "dense_retrieval_degraded",
                    query_id=query_id,
                    stage="retrieve",
                    exc=str(exc),
                    degraded_to="bm25_only",
                )
                dense_results = []
        with traced_stage("rrf", query_id, pool_size=pool_size):
            fused_results = fuse(bm25_results, dense_results, top_k=pool_size)
    except Exception as exc:
        log.error("retrieve_failed", query_id=query_id, stage="retrieve", exc=str(exc))
        return HybridResult(error=str(exc))
    return HybridResult(
        bm25_results=bm25_results,
        dense_results=dense_results,
        fused_results=fused_results,
    )
