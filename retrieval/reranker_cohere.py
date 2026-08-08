"""Re-ranking via the Cohere Rerank v3 API.

Mirrors retrieval/reranker_msmarco.py's shape exactly: a class wrapping an
injectable scoring callable, with a separate `load()` classmethod that does
the real cohere.Client() construction. Unlike the cross-encoder, this tier
calls a hosted API rather than a local model, so it never touches torch —
but per AGENTS.md's rule to mock all embedding/LLM calls in tests, the
client is still constructor-injected so unit tests never make a network
call or need a live COHERE_API_KEY.
"""

from __future__ import annotations

import os
from typing import Protocol

import structlog

from retrieval.candidates import Candidate

log = structlog.get_logger()

DEFAULT_MODEL_ID = "rerank-v3.5"


class RerankResultProtocol(Protocol):
    index: int
    relevance_score: float


class CohereRerankClientProtocol(Protocol):
    def rerank(
        self, query: str, documents: list[str], model: str, top_n: int
    ) -> list[RerankResultProtocol]: ...


class CohereReranker:
    """Cohere Rerank v3 re-ranker over a pre-retrieved candidate list."""

    def __init__(self, client: CohereRerankClientProtocol, model_id: str = DEFAULT_MODEL_ID) -> None:
        self._client = client
        self._model_id = model_id

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL_ID, api_key: str | None = None) -> CohereReranker:
        """Build the real Cohere-backed reranker.

        Imports cohere lazily so importing this module never requires the
        cohere package to be configured — unit tests inject a fake
        CohereRerankClientProtocol instead.
        """
        import cohere

        client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
        log.info("reranker_cohere_loaded", stage="reranker_cohere", query_id="startup", model_id=model_id)
        return cls(_CohereClientAdapter(client), model_id=model_id)

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int = 10,
        query_id: str = "unknown",
    ) -> list[Candidate]:
        if top_k <= 0 or not candidates:
            return []

        documents = [c.text for c in candidates]
        results = self._client.rerank(
            query=query, documents=documents, model=self._model_id, top_n=min(top_k, len(candidates))
        )

        log.info(
            "reranker_cohere_reranked",
            stage="reranker_cohere",
            query_id=query_id,
            num_candidates=len(candidates),
            top_k=top_k,
        )
        return [
            Candidate(
                chunk_id=candidates[result.index].chunk_id,
                text=candidates[result.index].text,
                score=float(result.relevance_score),
                rank=rank,
            )
            for rank, result in enumerate(results, start=1)
        ]


class _CohereClientAdapter:
    """Adapts cohere.Client's keyword-only `rerank()` to CohereRerankClientProtocol's shape."""

    def __init__(self, client: object) -> None:
        self._client = client

    def rerank(self, query: str, documents: list[str], model: str, top_n: int) -> list[RerankResultProtocol]:
        response = self._client.rerank(query=query, documents=documents, model=model, top_n=top_n)  # type: ignore[attr-defined]
        results: list[RerankResultProtocol] = response.results
        return results
