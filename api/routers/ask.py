"""POST /ask — grounded answer generation with per-claim citations."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, Query, Request

from agents import qa_agent
from api.dependencies import get_bm25_index, get_dense_index, get_msmarco_reranker
from api.schemas import AskRequest, AskResponse, CitationOut
from retrieval.bm25_index import BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.reranker_router import RerankerRouter

log = structlog.get_logger()

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    request: Request,
    reranker_mode: str | None = Query(default=None, alias="RERANKER_MODE"),
    bm25_index: BM25Index = Depends(get_bm25_index),
    dense_index: DenseIndex = Depends(get_dense_index),
    msmarco: MSMarcoReranker = Depends(get_msmarco_reranker),
) -> AskResponse:
    query_id = str(uuid.uuid4())[:8]
    request.state.query_id = query_id
    request.state.reranker_config = reranker_mode

    log.info("ask_request", query_id=query_id, stage="ask", reranker_mode=reranker_mode)

    reranker_router = RerankerRouter(live_fast=msmarco.rerank, mode=reranker_mode)
    state = qa_agent.run(
        body.query,
        top_k=body.top_k,
        candidate_pool_size=body.candidate_pool_size,
        query_id=query_id,
        bm25_index=bm25_index,
        dense_index=dense_index,
        router=reranker_router,
    )
    return AskResponse(
        query_id=query_id,
        query=body.query,
        reranker_mode=reranker_mode,
        answer=state.answer,
        citations=[CitationOut(claim=c.claim, chunk_ids=c.chunk_ids) for c in state.citations],
        error=state.error,
    )
