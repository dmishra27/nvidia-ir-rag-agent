"""POST /search — re-ranked hybrid retrieval, no generation.

RERANKER_MODE is accepted as a query parameter (not the JSON body) so a
client can flip re-ranking tiers per-call without altering the request
shape — mirrors the RERANKER_MODE env var from AGENTS.md, just scoped to
a single request instead of the whole process.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response

from agents import retrieval_agent
from api.dependencies import get_bm25_index, get_dense_index, get_msmarco_reranker
from api.schemas import CandidateOut, SearchRequest, SearchResponse
from retrieval.bm25_index import BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.reranker_router import RerankerRouter, resolve_mode

log = structlog.get_logger()

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    request: Request,
    response: Response,
    reranker_mode: str | None = Query(default=None, alias="RERANKER_MODE"),
    bm25_index: BM25Index = Depends(get_bm25_index),
    dense_index: DenseIndex | None = Depends(get_dense_index),
    msmarco: MSMarcoReranker | None = Depends(get_msmarco_reranker),
) -> SearchResponse:
    query_id = str(uuid.uuid4())[:8]
    resolved_mode = resolve_mode(reranker_mode)
    request.state.query_id = query_id
    request.state.reranker_config = reranker_mode

    log.info("search_request", query_id=query_id, stage="search", reranker_mode=resolved_mode)

    # msmarco is None when RERANKER_MODE=fallback skipped loading it (see
    # api/dependencies.py) -- RerankerRouter already degrades gracefully to
    # its fallback tier when a higher tier is None, per reranker_router.py.
    reranker_router = RerankerRouter(
        live_fast=msmarco.rerank if msmarco is not None else None, mode=reranker_mode
    )
    state = retrieval_agent.run_state(
        body.query,
        top_k=body.top_k,
        candidate_pool_size=body.candidate_pool_size,
        query_id=query_id,
        bm25_index=bm25_index,
        dense_index=dense_index,
        router=reranker_router,
    )
    if state.error is not None:
        # Retrieval failed outright (BM25 or fusion raised -- a dense-only
        # failure degrades to BM25 upstream and never lands here). Return the
        # error in the body AND a 503 so a caller can't mistake it for an
        # empty result set the way a bare HTTP 200 {"results": []} invites.
        log.error("search_failed", query_id=query_id, stage="search", error=state.error)
        response.status_code = 503
    return SearchResponse(
        query_id=query_id,
        query=body.query,
        reranker_mode=resolved_mode,
        results=[CandidateOut.from_candidate(c) for c in state.results],
        error=state.error,
    )
