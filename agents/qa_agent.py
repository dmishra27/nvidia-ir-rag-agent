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

`make_generate_node`'s `prompt_variant` selects between `PROMPT_VARIANTS`
("baseline", the original single instruction this project ran through Day
9; "cite_verify", a self-verification pass before the tool call). Compared
head-to-head in docs/uat/prompt_variant_comparison.md over the same 10
saved Config A contexts run_day9_ragas.py/run_day9_citation_judge.py used —
`DEFAULT_PROMPT_VARIANT` is that comparison's outcome, not an assumption.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

import anthropic
import structlog
from anthropic.types import MessageParam, ToolChoiceToolParam, ToolParam
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from api.telemetry import traced_stage
from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.reranker_router import RerankerRouter
from retrieval.rrf_fusion import fuse

load_dotenv()
log = structlog.get_logger()

# See agents/retrieval_agent.py's identical sentinel for why: `dense_index or
# DenseIndex.connect()` can't distinguish "omitted" (auto-connect, used by
# monitoring/quality_regression.py and evaluation/ragas_suite.py) from
# "explicitly None" (RERANKER_MODE=fallback skipped loading it on purpose --
# api/dependencies.py -- and connecting for real here would reintroduce the
# >512MB OOM on Render's free tier).
_UNSET_DENSE_INDEX: Any = object()

MODEL = "claude-sonnet-5"

CITE_TOOL: ToolParam = {
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

def make_retrieve_node(
    bm25_index: BM25Index, dense_index: DenseIndex | None
) -> Callable[[QAState], QAState]:
    def retrieve(state: QAState) -> QAState:
        log.info("retrieve", query_id=state.query_id, stage="retrieve")
        try:
            with traced_stage("bm25", state.query_id, top_k=state.candidate_pool_size):
                bm25_results = bm25_index.search(state.query, top_k=state.candidate_pool_size)
            if dense_index is None:
                # RERANKER_MODE=fallback deliberately skipped loading DenseIndex
                # (api/dependencies.py) -- see agents/retrieval_agent.py's identical
                # guard: a configuration choice, not a failure, so BM25 results
                # still get fused/returned instead of being discarded.
                dense_results: list[Candidate] = []
            else:
                with traced_stage("dense", state.query_id, top_k=state.candidate_pool_size):
                    dense_results = dense_index.search(state.query, top_k=state.candidate_pool_size)
            with traced_stage("rrf", state.query_id, pool_size=state.candidate_pool_size):
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
            with traced_stage("rerank", state.query_id, top_k=state.top_k):
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


# Prompt variants for the `generate` node, per docs/uat/prompt_variant_comparison.md.
# "baseline" is the original, unmodified instruction that shipped through Day 9 —
# kept verbatim so the comparison is against what actually ran in production, not
# a reworded version of it. "cite_verify" is a genuinely different generation
# strategy (not a reworded instruction): it asks the model to check each claim
# against its cited passage before finalizing the tool call, rather than
# citing and answering in one pass. Evaluated head-to-head in
# docs/uat/prompt_variant_comparison.md; DEFAULT_PROMPT_VARIANT below reflects
# that evaluation's outcome, not an assumption.
PROMPT_VARIANTS: dict[str, str] = {
    "baseline": (
        "Answer the question using only the passages below. Each "
        "passage is prefixed with its chunk_id in brackets. Every "
        "claim in your answer must cite the chunk_id(s) that "
        "support it.\n\n"
        "Question: {query}\n\nPassages:\n{context}"
    ),
    "cite_verify": (
        "Answer the question using only the passages below. Each "
        "passage is prefixed with its chunk_id in brackets.\n\n"
        "Before calling the tool, work through this privately: draft "
        "each claim your answer needs, then for each claim re-read its "
        "candidate chunk_id(s) and confirm the passage actually states "
        "what the claim asserts — not merely that it mentions the same "
        "topic. Drop or re-cite any claim whose passage doesn't "
        "substantiate it once you check. Only the final, verified "
        "answer and citations go into the tool call.\n\n"
        "Question: {query}\n\nPassages:\n{context}"
    ),
}
DEFAULT_PROMPT_VARIANT = "baseline"


def make_generate_node(
    model: str = MODEL, prompt_variant: str = DEFAULT_PROMPT_VARIANT
) -> Callable[[QAState], QAState]:
    prompt_template = PROMPT_VARIANTS[prompt_variant]

    def generate(state: QAState) -> QAState:
        log.info("generate", query_id=state.query_id, stage="generate", prompt_variant=prompt_variant)
        if state.error:
            return state

        passages = (state.reranked_results or state.fused_results)[: state.top_k]
        if not passages:
            return state.model_copy(update={"answer": "", "citations": []})

        client = anthropic.Anthropic()
        tool_choice: ToolChoiceToolParam = {"type": "tool", "name": "answer_with_citations"}
        messages: list[MessageParam] = [
            {
                "role": "user",
                "content": prompt_template.format(query=state.query, context=_build_context(passages)),
            }
        ]
        try:
            with traced_stage("llm", state.query_id, model=model):
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    tools=[CITE_TOOL],
                    tool_choice=tool_choice,
                    messages=messages,
                )
        except Exception as exc:
            log.error("generate_failed", query_id=state.query_id, stage="generate", exc=str(exc))
            return state.model_copy(update={"error": str(exc)})

        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            return state.model_copy(update={"error": "No structured answer produced by model."})

        raw_citations = tool_block.input.get("citations", [])
        if isinstance(raw_citations, str):
            # Observed live on Day 9 (query Q10): Claude occasionally returns
            # the whole `citations` value as a stringified JSON blob rather
            # than a native array, despite the forced tool schema. Recover it
            # if it parses to a list; iterating the raw string would silently
            # walk individual characters instead of citation objects.
            try:
                parsed = json.loads(raw_citations)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                log.warning(
                    "generate_citations_stringified_recovered",
                    query_id=state.query_id,
                    stage="generate",
                    num_recovered=len(parsed),
                )
                raw_citations = parsed
            else:
                log.warning(
                    "generate_citations_stringified_unparseable",
                    query_id=state.query_id,
                    stage="generate",
                    raw=raw_citations[:200],
                )
                raw_citations = []
        elif not isinstance(raw_citations, list):
            raw_citations = []

        citations: list[Citation] = []
        for c in raw_citations:
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
    bm25_index: BM25Index,
    dense_index: DenseIndex | None,
    router: RerankerRouter,
    model: str = MODEL,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> Any:
    graph = StateGraph(QAState)
    # mypy strict can't unify NodeInputT through add_node's StateNode Union-of-Protocols
    # overloads for a plain Callable[[QAState], QAState] (known langgraph/mypy stub
    # limitation, not a real type error -- each node's signature is correct). Bare
    # ignore because the reported error code flips between call-overload/arg-type.
    graph.add_node("retrieve", make_retrieve_node(bm25_index, dense_index))  # type: ignore
    graph.add_node("rerank", make_rerank_node(router))  # type: ignore
    graph.add_node("generate", make_generate_node(model, prompt_variant))  # type: ignore
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
    dense_index: DenseIndex | None = _UNSET_DENSE_INDEX,
    router: RerankerRouter | None = None,
    model: str = MODEL,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> QAState:
    bm25_index = bm25_index or BM25Index.load()
    dense_index = DenseIndex.connect() if dense_index is _UNSET_DENSE_INDEX else dense_index
    router = router or build_default_router()
    graph = build_graph(bm25_index, dense_index, router, model, prompt_variant)
    initial = QAState(
        query=query,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        query_id=query_id or str(uuid.uuid4())[:8],
    )
    result = graph.invoke(initial)
    return QAState(**result) if isinstance(result, dict) else result
