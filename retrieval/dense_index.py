"""Dense vector search over the Qdrant collection populated by populate_qdrant.py.

Mirrors retrieval/bm25_index.py and retrieval/splade_index.py's search()
signature and Candidate return shape. Unlike those two, there is no local
index to build or persist here — Day 4's populate_qdrant.py already embedded
the full corpus with the winning bi-encoder (e5-base-v2) and upserted it into
Qdrant, so DenseIndex only has to embed the query and delegate ranking to
Qdrant itself. The Qdrant client and the query encoder are both injectable so
unit tests never open a network connection or load torch/transformers, per
the project's rule to mock all embedding/LLM calls in tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Protocol

import structlog
from dotenv import load_dotenv

from retrieval.candidates import Candidate

load_dotenv()
log = structlog.get_logger()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION_NAME = "nvidia_ir_chunks"
EVAL_RESULTS_PATH = Path("biencoder_eval_results.json")
FALLBACK_HF_ID = "intfloat/e5-base-v2"
FALLBACK_QUERY_PREFIX = "query: "

# Maps a query string to its embedding vector.
QueryEncoder = Callable[[str], list[float]]


class QdrantSearchClient(Protocol):
    def query_points(self, collection_name: str, query: list[float], limit: int) -> Any: ...


def _winning_model() -> tuple[str, str]:
    """Return (hf_id, query_prefix) for the Day-4 bi-encoder winner, or a safe fallback.

    Same lookup as mcp/mcp_qdrant/server.py so query-time embedding stays
    consistent between the MCP tool and this module.
    """
    if EVAL_RESULTS_PATH.exists():
        data = json.loads(EVAL_RESULTS_PATH.read_text())
        winner_label = data.get("winner")
        for r in data.get("results", []):
            if r["model"] == winner_label:
                prefix = FALLBACK_QUERY_PREFIX if r["model"] == "e5-base-v2" else ""
                return r["hf_id"], prefix
    return FALLBACK_HF_ID, FALLBACK_QUERY_PREFIX


def build_default_encoder() -> QueryEncoder:
    """Build a real query encoder backed by sentence-transformers.

    Imports sentence_transformers lazily so importing this module never
    requires torch — unit tests inject a fake QueryEncoder instead.
    """
    from sentence_transformers import SentenceTransformer

    hf_id, query_prefix = _winning_model()
    model = SentenceTransformer(hf_id, device="cpu")
    log.info("dense_index_encoder_loaded", stage="dense_index", query_id="startup", hf_id=hf_id)

    def encode(query: str) -> list[float]:
        return model.encode(query_prefix + query, normalize_embeddings=True).tolist()

    return encode


class DenseIndex:
    """Cosine search over a Qdrant collection of dense chunk embeddings."""

    def __init__(
        self,
        client: QdrantSearchClient,
        encoder: QueryEncoder,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self._client = client
        self._encoder = encoder
        self._collection_name = collection_name

    @classmethod
    def connect(
        cls,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        url: str | None = None,
    ) -> DenseIndex:
        """Connect to a live Qdrant instance and load the real query encoder."""
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url or QDRANT_URL)
        encoder = build_default_encoder()
        log.info(
            "dense_index_connected",
            stage="dense_index",
            query_id="startup",
            collection=collection_name,
        )
        return cls(client, encoder, collection_name=collection_name)

    def search(self, query: str, top_k: int = 10) -> list[Candidate]:
        if top_k <= 0 or not query or not query.strip():
            return []
        vector = self._encoder(query)
        response = self._client.query_points(
            collection_name=self._collection_name, query=vector, limit=top_k
        )
        return [
            Candidate(
                chunk_id=(point.payload or {}).get("chunk_id"),
                text=(point.payload or {}).get("text", ""),
                score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]
