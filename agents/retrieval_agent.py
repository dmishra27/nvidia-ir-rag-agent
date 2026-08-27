"""LangGraph retrieval agent: retrieve -> rerank -> return_results.

Per SKILLS.md's build-langgraph-node pattern, each node takes an AgentState
and returns a new AgentState via model_copy(update=...), preserving every
existing field. The BM25 index, dense index, and reranker router are
constructor-injected into the node factories (not imported globally) so
unit tests can supply fakes and contract-test the state schema without
loading a real index, connecting to Qdrant, or loading a cross-encoder —
per AGENTS.md's rule to mock all embedding/LLM calls in unit tests and to
write contract tests on LangGraph state schema, not content.

- retrieve: BM25 top-100 + dense top-100 -> RRF fusion (Day 5 hybrid pipeline),
  via agents/hybrid_retrieve.py (shared with qa_agent). A dense-search failure
  (e.g. Qdrant collection not yet populated) degrades to BM25-only rather than
  failing the whole retrieval; only a BM25 or fusion failure sets state.error.
- rerank: reranker_router.rerank() over the fused pool (Day 6)
- return_results: selects the final ranked list, falling back to the fused
  pool if reranking produced nothing (e.g. an unrecoverable router error)
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

import structlog
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.hybrid_retrieve import hybrid_retrieve
from api.telemetry import traced_stage
from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.reranker_cohere import CohereReranker
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.reranker_router import RerankerRouter

log = structlog.get_logger()

# `dense_index or DenseIndex.connect()` below can't tell "caller omitted
# dense_index" (auto-connect a real one, used by run_hybrid_search.py and
# tests that call run() bare) apart from "caller explicitly passed None"
# (RERANKER_MODE=fallback skipped loading it on purpose -- see
# api/dependencies.py -- and a real DenseIndex.connect() here would defeat
# that, reintroducing the >512MB OOM on Render's free tier). This sentinel
# default lets `is _UNSET_DENSE_INDEX` distinguish the two; `Any` keeps
# mypy strict happy about the DenseIndex | None annotation below.
_UNSET_DENSE_INDEX: Any = object()


class AgentState(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str
    top_k: int = 10
    candidate_pool_size: int = 100
    bm25_results: list[Candidate] = Field(default_factory=list)
    dense_results: list[Candidate] = Field(default_factory=list)
    fused_results: list[Candidate] = Field(default_factory=list)
    reranked_results: list[Candidate] = Field(default_factory=list)
    results: list[Candidate] = Field(default_factory=list)
    error: str | None = None


# ── Node factories ────────────────────────────────────────────────────────────

def make_retrieve_node(
    bm25_index: BM25Index, dense_index: DenseIndex | None
) -> Callable[[AgentState], AgentState]:
    def retrieve(state: AgentState) -> AgentState:
        log.info("retrieve", query_id=state.query_id, stage="retrieve")
        result = hybrid_retrieve(
            state.query,
            state.query_id,
            bm25_index=bm25_index,
            dense_index=dense_index,
            pool_size=state.candidate_pool_size,
        )
        if result.error is not None:
            return state.model_copy(update={"error": result.error})
        return state.model_copy(
            update={
                "bm25_results": result.bm25_results,
                "dense_results": result.dense_results,
                "fused_results": result.fused_results,
            }
        )

    return retrieve


def make_rerank_node(router: RerankerRouter) -> Callable[[AgentState], AgentState]:
    def rerank(state: AgentState) -> AgentState:
        log.info("rerank", query_id=state.query_id, stage="rerank")
        if state.error:
            return state
        try:
            with traced_stage("rerank", state.query_id, top_k=state.top_k):
                reranked = router.rerank(
                    state.query, state.fused_results, top_k=state.top_k, query_id=state.query_id
                )
        except Exception as exc:
            log.error("rerank_failed", query_id=state.query_id, stage="rerank", exc=str(exc))
            return state.model_copy(update={"error": str(exc)})
        return state.model_copy(update={"reranked_results": reranked})

    return rerank


def return_results(state: AgentState) -> AgentState:
    log.info("return_results", query_id=state.query_id, stage="return_results")
    if state.error:
        return state.model_copy(update={"results": []})
    results = state.reranked_results if state.reranked_results else state.fused_results[: state.top_k]
    return state.model_copy(update={"results": results})


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(bm25_index: BM25Index, dense_index: DenseIndex | None, router: RerankerRouter) -> Any:
    graph = StateGraph(AgentState)
    # mypy strict can't unify NodeInputT through add_node's StateNode Union-of-Protocols
    # overloads for a plain Callable[[AgentState], AgentState] (known langgraph/mypy stub
    # limitation, not a real type error -- each node's signature is correct). Bare
    # ignore because the reported error code flips between call-overload/arg-type.
    graph.add_node("retrieve", make_retrieve_node(bm25_index, dense_index))  # type: ignore
    graph.add_node("rerank", make_rerank_node(router))  # type: ignore
    graph.add_node("return_results", return_results)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "return_results")
    graph.add_edge("return_results", END)
    return graph.compile()


def build_default_router(mode: str | None = None) -> RerankerRouter:
    """Wire live_fast to a freshly loaded ms-marco cross-encoder and
    live_frontier to Cohere Rerank v3 (requires COHERE_API_KEY). live_quality
    (bge-reranker-v2-m3) and benchmark degrade or raise until that tier is
    built — bge is hardware-blocked (OOM at model load) on this machine, per
    Day 8's evaluation/benchmark_runner.py."""
    msmarco = MSMarcoReranker.load()
    cohere_reranker = CohereReranker.load()
    return RerankerRouter(live_fast=msmarco.rerank, live_frontier=cohere_reranker.rerank, mode=mode)


def run_state(
    query: str,
    top_k: int = 10,
    candidate_pool_size: int = 100,
    query_id: str | None = None,
    bm25_index: BM25Index | None = None,
    dense_index: DenseIndex | None = _UNSET_DENSE_INDEX,
    router: RerankerRouter | None = None,
) -> AgentState:
    """Like run() but returns the full final AgentState instead of just the
    ranked list. Callers that need to tell "retrieval failed" (state.error
    set) apart from "no matches found" (empty results, error None) use this --
    e.g. api/routers/search.py surfaces state.error as an error field and a
    503 rather than a bare HTTP 200 {"results": []}."""
    bm25_index = bm25_index or BM25Index.load()
    dense_index = DenseIndex.connect() if dense_index is _UNSET_DENSE_INDEX else dense_index
    router = router or build_default_router()
    graph = build_graph(bm25_index, dense_index, router)
    initial = AgentState(
        query=query,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        query_id=query_id or str(uuid.uuid4())[:8],
    )
    result = graph.invoke(initial)
    return AgentState(**result) if isinstance(result, dict) else result


def run(
    query: str,
    top_k: int = 10,
    candidate_pool_size: int = 100,
    query_id: str | None = None,
    bm25_index: BM25Index | None = None,
    dense_index: DenseIndex | None = _UNSET_DENSE_INDEX,
    router: RerankerRouter | None = None,
) -> list[Candidate]:
    return run_state(
        query,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        query_id=query_id,
        bm25_index=bm25_index,
        dense_index=dense_index,
        router=router,
    ).results
