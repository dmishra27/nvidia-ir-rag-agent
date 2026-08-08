"""Agent-to-Agent (A2A) handoff protocol between agents/retrieval_agent.py
and agents/qa_agent.py.

Today, `qa_agent.py`'s own `retrieve`/`rerank` nodes (`make_retrieve_node`/
`make_rerank_node`) duplicate `retrieval_agent.py`'s logic almost exactly
rather than one agent handing its work to the other -- each independently
runs BM25 + dense + RRF + re-ranking over the same query. This module
formalizes that handoff instead: `retrieval_agent.py` computes ranked
results once, packages them in a typed `AgentMessage` envelope, and
`qa_agent.py` consumes the envelope directly into `QAState.reranked_results`
-- its `generate` node already reads `state.reranked_results or
state.fused_results` (see `make_generate_node`), so no code in either agent
changes; a handed-off state just skips `qa_agent`'s own retrieve/rerank
nodes entirely instead of re-running them.

Deliberately a small, project-local envelope (sender/recipient/query_id/
payload) rather than an implementation of Google's full A2A spec (HTTP/
JSON-RPC transport, AgentCard discovery, etc.) -- every agent in this
project runs in-process via LangGraph (see AGENTS.md's Layer 1), so there
is no transport to protocol-ize, only the message shape between two
Python function calls.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from agents.qa_agent import QAState, make_generate_node
from agents.retrieval_agent import AgentState
from agents.retrieval_agent import build_default_router as _build_default_retrieval_router
from agents.retrieval_agent import build_graph as _build_retrieval_graph
from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.reranker_router import RerankerRouter

log = structlog.get_logger()

MessageType = Literal["retrieval_handoff", "error"]


class RetrievalHandoffPayload(BaseModel):
    """retrieval_agent's final ranked results, ready for qa_agent to
    generate an answer over -- no re-retrieval, no re-ranking."""

    query: str
    top_k: int
    results: list[Candidate]


class ErrorPayload(BaseModel):
    message: str


class AgentMessage(BaseModel):
    """Typed inter-agent handoff envelope. `payload`'s shape is determined
    by `message_type`: `RetrievalHandoffPayload` for "retrieval_handoff",
    `ErrorPayload` for "error" -- pydantic validates whichever was actually
    constructed, not a union discriminator, since this project only ever
    builds messages through `build_retrieval_handoff`/`build_error_handoff`
    below rather than deserializing arbitrary envelopes off a wire."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query_id: str
    sender: str
    recipient: str
    message_type: MessageType
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: RetrievalHandoffPayload | ErrorPayload


def build_error_handoff(query_id: str, sender: str, recipient: str, message: str) -> AgentMessage:
    return AgentMessage(
        query_id=query_id,
        sender=sender,
        recipient=recipient,
        message_type="error",
        payload=ErrorPayload(message=message),
    )


def build_retrieval_handoff(state: AgentState, recipient: str = "qa_agent") -> AgentMessage:
    """Package retrieval_agent's AgentState as a handoff message. If the
    retrieval agent itself errored, packages an "error" message instead of
    a "retrieval_handoff" one -- callers don't need to check state.error
    separately before handing it off."""
    if state.error:
        return build_error_handoff(
            state.query_id, sender="retrieval_agent", recipient=recipient, message=state.error
        )
    return AgentMessage(
        query_id=state.query_id,
        sender="retrieval_agent",
        recipient=recipient,
        message_type="retrieval_handoff",
        payload=RetrievalHandoffPayload(query=state.query, top_k=state.top_k, results=state.results),
    )


def apply_retrieval_handoff(message: AgentMessage, qa_state: QAState | None = None) -> QAState:
    """Consume a handoff message into a QAState. A "retrieval_handoff"
    message populates `reranked_results` with the handed-off results
    (qa_agent's `generate` node reads `reranked_results or fused_results`,
    so this is a direct substitute for running its retrieve/rerank nodes);
    an "error" message short-circuits straight to `QAState.error`, matching
    every node's own `if state.error: return state` early-return
    convention. `qa_state` lets a caller hand off into an
    already-in-progress state rather than always starting fresh."""
    base = qa_state or QAState(query_id=message.query_id, query="")

    if message.message_type == "error":
        if not isinstance(message.payload, ErrorPayload):
            raise TypeError(f"error message with non-ErrorPayload payload: {type(message.payload)!r}")
        return base.model_copy(update={"error": message.payload.message})

    if not isinstance(message.payload, RetrievalHandoffPayload):
        raise TypeError(f"retrieval_handoff message with non-RetrievalHandoffPayload payload: {type(message.payload)!r}")
    return base.model_copy(
        update={
            "query": message.payload.query,
            "top_k": message.payload.top_k,
            "reranked_results": message.payload.results,
        }
    )


def _run_retrieval_agent(
    query: str,
    top_k: int,
    candidate_pool_size: int,
    query_id: str,
    bm25_index: BM25Index,
    dense_index: DenseIndex,
    router: RerankerRouter,
) -> AgentState:
    """Runs retrieval_agent's real graph and returns the full final
    AgentState (error included) -- retrieval_agent.run() itself only
    returns `list[Candidate]`, which loses the error/query_id context this
    module's handoff needs, so this mirrors its body rather than changing
    that public signature."""
    graph = _build_retrieval_graph(bm25_index, dense_index, router)
    initial = AgentState(
        query=query, top_k=top_k, candidate_pool_size=candidate_pool_size, query_id=query_id
    )
    result = graph.invoke(initial)
    return AgentState(**result) if isinstance(result, dict) else result


def run_handoff(
    query: str,
    top_k: int = 10,
    candidate_pool_size: int = 100,
    query_id: str | None = None,
    bm25_index: BM25Index | None = None,
    dense_index: DenseIndex | None = None,
    router: RerankerRouter | None = None,
    model: str = "claude-sonnet-5",
) -> QAState:
    """End-to-end demo of the handoff this protocol exists for:
    retrieval_agent computes ranked results once; qa_agent's `generate`
    node runs directly on the handed-off results instead of re-retrieving
    and re-ranking via its own nodes -- one retrieval pass shared across
    two agents, not two independent ones."""
    query_id = query_id or str(uuid.uuid4())[:8]
    bm25_index = bm25_index or BM25Index.load()
    dense_index = dense_index or DenseIndex.connect()
    router = router or _build_default_retrieval_router()

    log.info("a2a_handoff_start", query_id=query_id, stage="a2a_protocol", sender="retrieval_agent")
    retrieval_state = _run_retrieval_agent(
        query, top_k, candidate_pool_size, query_id, bm25_index, dense_index, router
    )
    message = build_retrieval_handoff(retrieval_state)
    log.info(
        "a2a_handoff_sent",
        query_id=query_id,
        stage="a2a_protocol",
        message_id=message.message_id,
        message_type=message.message_type,
    )

    qa_state = apply_retrieval_handoff(message)
    if qa_state.error:
        return qa_state

    return make_generate_node(model)(qa_state)
