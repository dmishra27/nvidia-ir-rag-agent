"""POST /search — re-ranked hybrid retrieval, no generation.

RERANKER_MODE is accepted as a query parameter (not the JSON body) so a
client can flip re-ranking tiers per-call without altering the request
shape — mirrors the RERANKER_MODE env var from AGENTS.md, just scoped to
a single request instead of the whole process.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, Query, Request

from agents import retrieval_agent
from api.dependencies import get_bm25_index, get_dense_index, get_msmarco_reranker
from api.schemas import CandidateOut, SearchRequest, SearchResponse
from retrieval.bm25_index import BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.reranker_router import RerankerRouter

log = structlog.get_logger()

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    request: Request,
    reranker_mode: str | None = Query(default=None, alias="RERANKER_MODE"),
    bm25_index: BM25Index = Depends(get_bm25_index),
    dense_index: DenseIndex = Depends(get_dense_index),
    msmarco: MSMarcoReranker = Depends(get_msmarco_reranker),
) -> SearchResponse:
    query_id = str(uuid.uuid4())[:8]
    request.state.query_id = query_id
    request.state.reranker_config = reranker_mode

    log.info("search_request", query_id=query_id, stage="search", reranker_mode=reranker_mode)

    reranker_router = RerankerRouter(live_fast=msmarco.rerank, mode=reranker_mode)
    results = retrieval_agent.run(
        body.query,
        top_k=body.top_k,
        candidate_pool_size=body.candidate_pool_size,
        query_id=query_id,
        bm25_index=bm25_index,
        dense_index=dense_index,
        router=reranker_router,
    )
    return SearchResponse(
        query_id=query_id,
        query=body.query,
        reranker_mode=reranker_mode,
        results=[CandidateOut.from_candidate(c) for c in results],
    )
