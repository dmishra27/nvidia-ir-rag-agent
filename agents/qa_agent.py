"""LangGraph QA agent: retrieve -> rerank -> generate.

Per SKILLS.md's build-langgraph-node pattern, each node takes a QAState and
returns a new QAState via model_copy(update=...), preserving every existing
field. Retrieve/rerank mirror agents/retrieval_agent.py's constructor-injected
BM25/dense/router pattern so unit tests can contract-test the state schema
without loading a real index, connecting to Qdrant, or loading a
cross-encoder.

The generate node grounds Claude Sonnet in the top-`top_k` re-ranked
passages and forces a tool call (`answer_with_citations`) so citations come
back as structured (claim, chunk_ids) pairs per claim rather than parsed
out of free text. Per agents/text_to_sql_agent.py's established convention
for LLM nodes, the Anthropic client is instantiated directly inside the
node (not constructor-injected) and unit tests patch
`agents.qa_agent.anthropic.Anthropic` — per AGENTS.md's rule to mock all
LLM calls in tests.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

import anthropic
import structlog
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.reranker_router import RerankerRouter
from retrieval.rrf_fusion import fuse

load_dotenv()
log = structlog.get_logger()

MODEL = "claude-sonnet-5"

CITE_TOOL: dict[str, Any] = {
    "name": "answer_with_citations",
    "description": (
        "Provide a grounded answer to the user's question, decomposed into "
        "claims, each citing the chunk_id(s) of the passages that support it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The full answer to the question, grounded only in the provided passages.",
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "A single claim or sentence from the answer.",
                        },
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "chunk_id(s) of the passages supporting this claim.",
                        },
                    },
                    "required": ["claim", "chunk_ids"],
                },
            },
        },
        "required": ["answer", "citations"],
    },
}


class Citation(BaseModel):
    claim: str
    chunk_ids: list[str] = Field(default_factory=list)


class QAState(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str
    top_k: int = 10
    candidate_pool_size: int = 100
    fused_results: list[Candidate] = Field(default_factory=list)
    reranked_results: list[Candidate] = Field(default_factory=list)
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    error: str | None = None


# ── Node factories ────────────────────────────────────────────────────────────

def make_retrieve_node(bm25_index: BM25Index, dense_index: DenseIndex) -> Callable[[QAState], QAState]:
    def retrieve(state: QAState) -> QAState:
        log.info("retrieve", query_id=state.query_id, stage="retrieve")
        try:
            bm25_results = bm25_index.search(state.query, top_k=state.candidate_pool_size)
            dense_results = dense_index.search(state.query, top_k=state.candidate_pool_size)
            fused_results = fuse(bm25_results, dense_results, top_k=state.candidate_pool_size)
        except Exception as exc:
            log.error("retrieve_failed", query_id=state.query_id, stage="retrieve", exc=str(exc))
            return state.model_copy(update={"error": str(exc)})
        return state.model_copy(update={"fused_results": fused_results})

    return retrieve


def make_rerank_node(router: RerankerRouter) -> Callable[[QAState], QAState]:
    def rerank(state: QAState) -> QAState:
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


def _build_context(passages: list[Candidate]) -> str:
    return "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in passages)


def make_generate_node(model: str = MODEL) -> Callable[[QAState], QAState]:
    def generate(state: QAState) -> QAState:
        log.info("generate", query_id=state.query_id, stage="generate")
        if state.error:
            return state

        passages = (state.reranked_results or state.fused_results)[: state.top_k]
        if not passages:
            return state.model_copy(update={"answer": "", "citations": []})

        client = anthropic.Anthropic()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=[CITE_TOOL],
                tool_choice={"type": "tool", "name": "answer_with_citations"},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the question using only the passages below. Each "
                            "passage is prefixed with its chunk_id in brackets. Every "
                            "claim in your answer must cite the chunk_id(s) that "
                            "support it.\n\n"
                            f"Question: {state.query}\n\nPassages:\n{_build_context(passages)}"
                        ),
                    }
                ],
            )
        except Exception as exc:
            log.error("generate_failed", query_id=state.query_id, stage="generate", exc=str(exc))
            return state.model_copy(update={"error": str(exc)})

        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            return state.model_copy(update={"error": "No structured answer produced by model."})

        citations: list[Citation] = []
        for c in tool_block.input.get("citations", []):
            if not isinstance(c, dict) or "claim" not in c:
                log.warning(
                    "generate_malformed_citation_skipped",
                    query_id=state.query_id,
                    stage="generate",
                    citation=repr(c)[:200],
                )
                continue
            citations.append(Citation(claim=c["claim"], chunk_ids=list(c.get("chunk_ids", []))))

        return state.model_copy(
            update={"answer": tool_block.input.get("answer", ""), "citations": citations}
        )

    return generate


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(
    bm25_index: BM25Index, dense_index: DenseIndex, router: RerankerRouter, model: str = MODEL
) -> Any:
    graph = StateGraph(QAState)
    graph.add_node("retrieve", make_retrieve_node(bm25_index, dense_index))
    graph.add_node("rerank", make_rerank_node(router))
    graph.add_node("generate", make_generate_node(model))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def build_default_router(mode: str | None = None) -> RerankerRouter:
    from agents.retrieval_agent import build_default_router as _build

    return _build(mode)


def run(
    query: str,
    top_k: int = 10,
    candidate_pool_size: int = 100,
    query_id: str | None = None,
    bm25_index: BM25Index | None = None,
    dense_index: DenseIndex | None = None,
    router: RerankerRouter | None = None,
    model: str = MODEL,
) -> QAState:
    bm25_index = bm25_index or BM25Index.load()
    dense_index = dense_index or DenseIndex.connect()
    router = router or build_default_router()
    graph = build_graph(bm25_index, dense_index, router, model)
    initial = QAState(
        query=query,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        query_id=query_id or str(uuid.uuid4())[:8],
    )
    result = graph.invoke(initial)
    return QAState(**result) if isinstance(result, dict) else result
