"""Cross-encoder re-ranking using cross-encoder/ms-marco-MiniLM-L-6-v2.

Mirrors retrieval/dense_index.py's shape: a class wrapping an injectable
scoring callable, with a separate `load()` classmethod that does the real
sentence-transformers import and model load. Unlike BM25/dense/SPLADE this
module scores (query, candidate_text) pairs directly rather than searching
an index — it re-orders a candidate list already produced by a first-stage
retriever (BM25, dense, or RRF). The cross-encoder is constructor-injected
so unit tests never load torch/transformers, per the project's rule to mock
all embedding/LLM calls in tests.
"""

from __future__ import annotations

from typing import Protocol

import structlog

from retrieval.candidates import Candidate

log = structlog.get_logger()

DEFAULT_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderProtocol(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class MSMarcoReranker:
    """Cross-encoder re-ranker over a pre-retrieved candidate list."""

    def __init__(self, cross_encoder: CrossEncoderProtocol) -> None:
        self._cross_encoder = cross_encoder

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL_ID) -> MSMarcoReranker:
        """Load the real cross-encoder backed by sentence-transformers.

        Imports sentence_transformers lazily so importing this module never
        requires torch — unit tests inject a fake CrossEncoderProtocol instead.
        """
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_id, device="cpu")
        log.info(
            "reranker_msmarco_loaded",
            stage="reranker_msmarco",
            query_id="startup",
            model_id=model_id,
        )
        return cls(model)

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int = 10,
        query_id: str = "unknown",
    ) -> list[Candidate]:
        if top_k <= 0 or not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._cross_encoder.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]

        log.info(
            "reranker_msmarco_reranked",
            stage="reranker_msmarco",
            query_id=query_id,
            num_candidates=len(candidates),
            top_k=top_k,
        )
        return [
            Candidate(chunk_id=c.chunk_id, text=c.text, score=float(s), rank=rank)
            for rank, (c, s) in enumerate(scored, start=1)
        ]
