"""LangGraph retrieval agent: retrieve -> rerank -> return_results.

Per SKILLS.md's build-langgraph-node pattern, each node takes an AgentState
and returns a new AgentState via model_copy(update=...), preserving every
existing field. The BM25 index, dense index, and reranker router are
constructor-injected into the node factories (not imported globally) so
unit tests can supply fakes and contract-test the state schema without
loading a real index, connecting to Qdrant, or loading a cross-encoder —
per AGENTS.md's rule to mock all embedding/LLM calls in unit tests and to
write contract tests on LangGraph state schema, not content.

- retrieve: BM25 top-100 + dense top-100 -> RRF fusion (Day 5 hybrid pipeline)
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

from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.reranker_router import RerankerRouter
from retrieval.rrf_fusion import fuse

log = structlog.get_logger()


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

def make_retrieve_node(bm25_index: BM25Index, dense_index: DenseIndex) -> Callable[[AgentState], AgentState]:
    def retrieve(state: AgentState) -> AgentState:
        log.info("retrieve", query_id=state.query_id, stage="retrieve")
        try:
            bm25_results = bm25_index.search(state.query, top_k=state.candidate_pool_size)
            dense_results = dense_index.search(state.query, top_k=state.candidate_pool_size)
            fused_results = fuse(bm25_results, dense_results, top_k=state.candidate_pool_size)
        except Exception as exc:
            log.error("retrieve_failed", query_id=state.query_id, stage="retrieve", exc=str(exc))
            return state.model_copy(update={"error": str(exc)})
        return state.model_copy(
            update={
                "bm25_results": bm25_results,
                "dense_results": dense_results,
                "fused_results": fused_results,
            }
        )

    return retrieve


def make_rerank_node(router: RerankerRouter) -> Callable[[AgentState], AgentState]:
    def rerank(state: AgentState) -> AgentState:
        log.info("rerank", query_id=state.query_id, stage="rerank")
        if state.error:
            return state
        try:
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

def build_graph(bm25_index: BM25Index, dense_index: DenseIndex, router: RerankerRouter) -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", make_retrieve_node(bm25_index, dense_index))
    graph.add_node("rerank", make_rerank_node(router))
    graph.add_node("return_results", return_results)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "return_results")
    graph.add_edge("return_results", END)
    return graph.compile()


def build_default_router(mode: str | None = None) -> RerankerRouter:
    """Wire live_fast to a freshly loaded ms-marco cross-encoder; other tiers
    (live_quality, live_frontier, benchmark) degrade or raise until Layer 3b's
    remaining re-rankers are built."""
    msmarco = MSMarcoReranker.load()
    return RerankerRouter(live_fast=msmarco.rerank, mode=mode)


def run(
    query: str,
    top_k: int = 10,
    candidate_pool_size: int = 100,
    query_id: str | None = None,
    bm25_index: BM25Index | None = None,
    dense_index: DenseIndex | None = None,
    router: RerankerRouter | None = None,
) -> list[Candidate]:
    bm25_index = bm25_index or BM25Index.load()
    dense_index = dense_index or DenseIndex.connect()
    router = router or build_default_router()
    graph = build_graph(bm25_index, dense_index, router)
    initial = AgentState(
        query=query,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        query_id=query_id or str(uuid.uuid4())[:8],
    )
    result = graph.invoke(initial)
    final_state = AgentState(**result) if isinstance(result, dict) else result
    return final_state.results
