"""FastAPI dependency providers for retrieval/reranking components.

Each provider loads its real component once per process (cached via
lru_cache) but is expressed as a plain Depends()-able function so unit
tests can override it with `app.dependency_overrides` — per AGENTS.md's
rule to mock all embedding/LLM calls in tests, and to never load a real
index, connect to Qdrant, or load a cross-encoder in a unit test.

get_dense_index/get_msmarco_reranker are skipped entirely when
RERANKER_MODE=fallback (see retrieval/reranker_router.py's serving chain):
Render's free tier caps a process at 512MB, and DenseIndex.connect() +
MSMarcoReranker.load() pull in qdrant_client and two sentence-transformers
models (torch + model weights), which alone exceed that before a single
request is served — confirmed via Render's Events log ("Ran out of memory
(used over 512MB)") on every /search call. BM25Index.load() alone is
~50MB. api/routers/search.py and api/routers/ask.py already default the
deployed UI to RERANKER_MODE=fallback (api/static/index.html), so this is
the mode Render actually serves, not a secondary path.
"""

from __future__ import annotations

import os
from functools import lru_cache

from retrieval.bm25_index import BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.reranker_router import RERANKER_MODE_ENV_VAR, DEFAULT_MODE


def _reranker_mode() -> str:
    return os.environ.get(RERANKER_MODE_ENV_VAR, DEFAULT_MODE)


@lru_cache
def get_bm25_index() -> BM25Index:
    return BM25Index.load()


@lru_cache
def get_dense_index() -> DenseIndex | None:
    """Connect to Qdrant + load the query encoder — skipped in fallback mode.

    Returning None (rather than raising) lets /search and /ask keep serving
    real BM25-ranked results: agents/retrieval_agent.py's and
    agents/qa_agent.py's retrieve nodes both treat a None dense_index as a
    deliberate skip (dense_results=[]) rather than a failure, so BM25's
    results still get fused/returned instead of being discarded the way an
    actual dense-search exception is.
    """
    if _reranker_mode() == "fallback":
        return None
    return DenseIndex.connect()


@lru_cache
def get_msmarco_reranker() -> MSMarcoReranker | None:
    """Load the ms-marco cross-encoder — skipped in fallback mode (see get_dense_index)."""
    if _reranker_mode() == "fallback":
        return None
    return MSMarcoReranker.load()
