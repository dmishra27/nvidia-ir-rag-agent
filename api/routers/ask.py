"""POST /ask — grounded answer generation with per-claim citations."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response

from agents import qa_agent
from api.dependencies import get_bm25_index, get_dense_index, get_msmarco_reranker
from api.schemas import AskRequest, AskResponse, CitationOut
from retrieval.bm25_index import BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.reranker_router import RerankerRouter, resolve_mode

log = structlog.get_logger()

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    request: Request,
    response: Response,
    reranker_mode: str | None = Query(default=None, alias="RERANKER_MODE"),
    bm25_index: BM25Index = Depends(get_bm25_index),
    dense_index: DenseIndex | None = Depends(get_dense_index),
    msmarco: MSMarcoReranker | None = Depends(get_msmarco_reranker),
) -> AskResponse:
    query_id = str(uuid.uuid4())[:8]
    resolved_mode = resolve_mode(reranker_mode)
    request.state.query_id = query_id
    request.state.reranker_config = reranker_mode

    log.info("ask_request", query_id=query_id, stage="ask", reranker_mode=resolved_mode)

    # msmarco is None when RERANKER_MODE=fallback skipped loading it (see
    # api/dependencies.py) -- RerankerRouter already degrades gracefully to
    # its fallback tier when a higher tier is None, per reranker_router.py.
    reranker_router = RerankerRouter(
        live_fast=msmarco.rerank if msmarco is not None else None, mode=reranker_mode
    )
    state = qa_agent.run(
        body.query,
        top_k=body.top_k,
        candidate_pool_size=body.candidate_pool_size,
        query_id=query_id,
        bm25_index=bm25_index,
        dense_index=dense_index,
        router=reranker_router,
    )
    if state.error is not None:
        # Retrieval or generation failed outright (a dense-only failure degrades
        # to BM25 upstream and never lands here -- see agents/qa_agent.py). Return
        # the error in the body AND a 503, mirroring api/routers/search.py: a
        # caller must be able to tell this from an empty-but-successful answer,
        # and a failed retrieval must not read as "the model had nothing to say".
        log.error("ask_failed", query_id=query_id, stage="ask", error=state.error)
        response.status_code = 503
    return AskResponse(
        query_id=query_id,
        query=body.query,
        reranker_mode=resolved_mode,
        answer=state.answer,
        citations=[CitationOut(claim=c.claim, chunk_ids=c.chunk_ids) for c in state.citations],
        error=state.error,
    )
