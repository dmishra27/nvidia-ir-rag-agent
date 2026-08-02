"""FastAPI dependency providers for retrieval/reranking components.

Each provider loads its real component once per process (cached via
lru_cache) but is expressed as a plain Depends()-able function so unit
tests can override it with `app.dependency_overrides` — per AGENTS.md's
rule to mock all embedding/LLM calls in tests, and to never load a real
index, connect to Qdrant, or load a cross-encoder in a unit test.
"""

from __future__ import annotations

from functools import lru_cache

from retrieval.bm25_index import BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.reranker_msmarco import MSMarcoReranker


@lru_cache
def get_bm25_index() -> BM25Index:
    return BM25Index.load()


@lru_cache
def get_dense_index() -> DenseIndex:
    return DenseIndex.connect()


@lru_cache
def get_msmarco_reranker() -> MSMarcoReranker:
    return MSMarcoReranker.load()
