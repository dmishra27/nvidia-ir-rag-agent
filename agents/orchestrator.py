"""LangGraph multi-agent orchestrator: retrieve -> rerank -> generate -> evaluate.

Wires agents/retrieval_agent.py's retrieve/rerank nodes, agents/qa_agent.py's
generate node, and agents/eval_agent.py's evaluate node into a single graph
over EvalState (a QAState superset). Every node is reused as-is from its own
agent module — this file only composes the graph and default constructors,
following the same constructor-injection convention all three modules
already use, so unit tests supply fakes and contract-test state transitions
without loading a real index, connecting to Qdrant, loading a cross-encoder,
or calling a live API.
"""

from __future__ import annotations

import uuid
from typing import Any

import anthropic
import structlog
from langgraph.graph import END, StateGraph

from agents.eval_agent import EvalState, make_evaluate_node
from agents.qa_agent import MODEL, build_default_router, make_generate_node, make_rerank_node, make_retrieve_node
from retrieval.bm25_index import BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.reranker_router import RerankerRouter

log = structlog.get_logger()


def build_graph(
    bm25_index: BM25Index,
    dense_index: DenseIndex,
    router: RerankerRouter,
    client: anthropic.Anthropic,
    model: str = MODEL,
) -> Any:
    graph = StateGraph(EvalState)
    graph.add_node("retrieve", make_retrieve_node(bm25_index, dense_index))
    graph.add_node("rerank", make_rerank_node(router))
    graph.add_node("generate", make_generate_node(model))
    graph.add_node("evaluate", make_evaluate_node(client, model))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", END)
    return graph.compile()


def run(
    query: str,
    top_k: int = 10,
    candidate_pool_size: int = 100,
    query_id: str | None = None,
    bm25_index: BM25Index | None = None,
    dense_index: DenseIndex | None = None,
    router: RerankerRouter | None = None,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> EvalState:
    bm25_index = bm25_index or BM25Index.load()
    dense_index = dense_index or DenseIndex.connect()
    router = router or build_default_router()
    client = client or anthropic.Anthropic()
    graph = build_graph(bm25_index, dense_index, router, client, model)
    initial = EvalState(
        query=query,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        query_id=query_id or str(uuid.uuid4())[:8],
    )
    result = graph.invoke(initial)
    return EvalState(**result) if isinstance(result, dict) else result
