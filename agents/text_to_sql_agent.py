from __future__ import annotations

import uuid
from typing import Any

import anthropic
import structlog
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from schema.models import Chunk, ChunkQuality, DocMetadata, get_engine, get_session_factory

load_dotenv()
log = structlog.get_logger()

MODEL = "claude-opus-4-8"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "count_docs_by_gpu_family",
        "description": "Count indexed documents grouped by GPU family (Hopper, Ampere, Ada Lovelace, etc.).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "chunks_below_quality_threshold",
        "description": "Return chunks whose quality score is below a given threshold (0.0–1.0).",
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Chunks with quality_score < threshold are returned.",
                }
            },
            "required": ["threshold"],
        },
    },
    {
        "name": "avg_quality_per_doc",
        "description": "Return the average chunk quality score for each indexed document.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "chunk_count_for_doc",
        "description": "Return the number of chunks for documents whose title contains a given fragment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title_fragment": {
                    "type": "string",
                    "description": "Case-insensitive substring matched against document titles.",
                }
            },
            "required": ["title_fragment"],
        },
    },
    {
        "name": "list_indexed_documents",
        "description": "List all indexed documents with title, GPU family, doc type, and page count.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


class TextToSQLState(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str
    tool_name: str | None = None
    tool_params: dict[str, Any] = Field(default_factory=dict)
    sql_result: list[dict[str, Any]] | None = None
    answer: str | None = None
    error: str | None = None


# ── ORM query functions ───────────────────────────────────────────────────────

def _count_docs_by_gpu_family(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(DocMetadata.gpu_family, func.count(DocMetadata.doc_id).label("count"))
        .group_by(DocMetadata.gpu_family)
        .order_by(func.count(DocMetadata.doc_id).desc())
    ).all()
    return [{"gpu_family": r.gpu_family or "Unknown", "count": r.count} for r in rows]


def _chunks_below_threshold(session: Session, threshold: float) -> list[dict[str, Any]]:
    rows = session.execute(
        select(ChunkQuality.chunk_id, ChunkQuality.quality_score, ChunkQuality.flagged)
        .where(ChunkQuality.quality_score < threshold)
        .order_by(ChunkQuality.quality_score)
    ).all()
    return [
        {"chunk_id": r.chunk_id, "quality_score": r.quality_score, "flagged": r.flagged}
        for r in rows
    ]


def _avg_quality_per_doc(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            DocMetadata.title,
            func.avg(ChunkQuality.quality_score).label("avg_quality"),
            func.count(ChunkQuality.chunk_id).label("chunk_count"),
        )
        .join(Chunk, Chunk.doc_id == DocMetadata.doc_id)
        .join(ChunkQuality, ChunkQuality.chunk_id == Chunk.chunk_id)
        .group_by(DocMetadata.title)
        .order_by(func.avg(ChunkQuality.quality_score).desc())
    ).all()
    return [
        {
            "title": r.title,
            "avg_quality": round(float(r.avg_quality), 4),
            "chunk_count": r.chunk_count,
        }
        for r in rows
    ]


def _chunk_count_for_doc(session: Session, title_fragment: str) -> list[dict[str, Any]]:
    rows = session.execute(
        select(DocMetadata.title, func.count(Chunk.chunk_id).label("chunk_count"))
        .join(Chunk, Chunk.doc_id == DocMetadata.doc_id)
        .where(DocMetadata.title.ilike(f"%{title_fragment}%"))
        .group_by(DocMetadata.title)
    ).all()
    return [{"title": r.title, "chunk_count": r.chunk_count} for r in rows]


def _list_indexed_documents(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            DocMetadata.doc_id,
            DocMetadata.title,
            DocMetadata.gpu_family,
            DocMetadata.doc_type,
            DocMetadata.page_count,
        ).order_by(DocMetadata.title)
    ).all()
    return [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "gpu_family": r.gpu_family,
            "doc_type": r.doc_type,
            "page_count": r.page_count,
        }
        for r in rows
    ]


def _dispatch(session: Session, tool_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    match tool_name:
        case "count_docs_by_gpu_family":
            return _count_docs_by_gpu_family(session)
        case "chunks_below_quality_threshold":
            return _chunks_below_threshold(session, float(params["threshold"]))
        case "avg_quality_per_doc":
            return _avg_quality_per_doc(session)
        case "chunk_count_for_doc":
            return _chunk_count_for_doc(session, str(params["title_fragment"]))
        case "list_indexed_documents":
            return _list_indexed_documents(session)
        case _:
            raise ValueError(f"Unknown tool: {tool_name!r}")


# ── LangGraph nodes ───────────────────────────────────────────────────────────

def plan_query(state: TextToSQLState) -> TextToSQLState:
    log.info("plan_query", query_id=state.query_id, stage="plan_query")
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        tools=TOOLS,
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a text-to-SQL assistant for an NVIDIA documentation RAG system. "
                    f"User question: {state.question}\n\n"
                    "Select the most appropriate tool to answer this question."
                ),
            }
        ],
    )
    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        return state.model_copy(update={"error": "No tool selected by model."})
    return state.model_copy(update={"tool_name": tool_block.name, "tool_params": tool_block.input})


def execute_query(state: TextToSQLState) -> TextToSQLState:
    log.info("execute_query", query_id=state.query_id, stage="execute_query", tool=state.tool_name)
    if state.error or state.tool_name is None:
        return state
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    try:
        with SessionFactory() as session:
            rows = _dispatch(session, state.tool_name, state.tool_params)
        return state.model_copy(update={"sql_result": rows})
    except Exception as exc:
        log.error(
            "execute_query_failed",
            query_id=state.query_id,
            stage="execute_query",
            exc=str(exc),
        )
        return state.model_copy(update={"error": str(exc)})


def format_answer(state: TextToSQLState) -> TextToSQLState:
    log.info("format_answer", query_id=state.query_id, stage="format_answer")
    if state.error:
        return state.model_copy(update={"answer": f"Error: {state.error}"})
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a helpful assistant answering questions about an NVIDIA documentation "
                    "RAG system database.\n\n"
                    f"User question: {state.question}\n\n"
                    f"Database results: {state.sql_result}\n\n"
                    "Write a concise, informative answer in plain English."
                ),
            }
        ],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return state.model_copy(update={"answer": text})


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(TextToSQLState)
    graph.add_node("plan_query", plan_query)
    graph.add_node("execute_query", execute_query)
    graph.add_node("format_answer", format_answer)
    graph.set_entry_point("plan_query")
    graph.add_edge("plan_query", "execute_query")
    graph.add_edge("execute_query", "format_answer")
    graph.add_edge("format_answer", END)
    return graph.compile()


def run(question: str, query_id: str | None = None) -> TextToSQLState:
    graph = build_graph()
    initial = TextToSQLState(
        question=question,
        query_id=query_id or str(uuid.uuid4())[:8],
    )
    result = graph.invoke(initial)
    if isinstance(result, dict):
        return TextToSQLState(**result)
    return result
