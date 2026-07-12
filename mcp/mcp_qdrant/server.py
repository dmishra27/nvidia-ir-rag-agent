"""MCP server exposing Qdrant vector search and collection stats.

search_vectors embeds query_text with the winning Day-4 bi-encoder
(read from biencoder_eval_results.json, falling back to e5-base-v2)
and runs a cosine search against a live Qdrant collection.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# MCP stdio transport reserves stdout for JSON-RPC framing; structlog's
# default PrintLoggerFactory writes to stdout, which corrupts every tool
# call's response. Logs must go to stderr instead.
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
log = structlog.get_logger()

mcp = FastMCP("nvidia-ir-qdrant")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EVAL_RESULTS_PATH = Path(__file__).resolve().parents[2] / "biencoder_eval_results.json"
FALLBACK_HF_ID = "intfloat/e5-base-v2"
FALLBACK_QUERY_PREFIX = "query: "

_client: QdrantClient | None = None
_model: SentenceTransformer | None = None
_query_prefix: str = ""


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _winning_model() -> tuple[str, str]:
    """Return (hf_id, query_prefix) for the Day-4 bi-encoder winner, or a safe fallback."""
    if EVAL_RESULTS_PATH.exists():
        data = json.loads(EVAL_RESULTS_PATH.read_text())
        winner_label = data.get("winner")
        for r in data.get("results", []):
            if r["model"] == winner_label:
                prefix = FALLBACK_QUERY_PREFIX if r["model"] == "e5-base-v2" else ""
                return r["hf_id"], prefix
    return FALLBACK_HF_ID, FALLBACK_QUERY_PREFIX


def _get_model() -> SentenceTransformer:
    global _model, _query_prefix
    if _model is None:
        hf_id, prefix = _winning_model()
        _query_prefix = prefix
        log.info("mcp_qdrant_model_loaded", stage="mcp_qdrant", query_id="server", hf_id=hf_id)
        _model = SentenceTransformer(hf_id, device="cpu")
    return _model


@mcp.tool()
def search_vectors(collection_name: str, query_text: str, top_k: int = 10) -> dict[str, Any]:
    """Embed query_text with the winning bi-encoder and search a Qdrant collection."""
    log.info(
        "search_vectors", stage="mcp_qdrant", query_id="tool_call",
        collection=collection_name, top_k=top_k,
    )
    model = _get_model()
    vector = model.encode(_query_prefix + query_text, normalize_embeddings=True).tolist()
    client = _get_client()
    response = client.query_points(collection_name=collection_name, query=vector, limit=top_k)
    return {
        "collection": collection_name,
        "query": query_text,
        "results": [
            {
                "chunk_id": (point.payload or {}).get("chunk_id"),
                "score": point.score,
                "text_snippet": (point.payload or {}).get("text", "")[:200],
            }
            for point in response.points
        ],
    }


@mcp.tool()
def collection_stats(collection_name: str) -> dict[str, Any]:
    """Return point count, indexed vector count, and status for a Qdrant collection."""
    log.info("collection_stats", stage="mcp_qdrant", query_id="tool_call", collection=collection_name)
    client = _get_client()
    info = client.get_collection(collection_name)

    vector_size: int | None = None
    vectors_config = info.config.params.vectors if info.config and info.config.params else None
    if vectors_config is not None:
        if hasattr(vectors_config, "size"):
            vector_size = vectors_config.size
        elif isinstance(vectors_config, dict):
            first = next(iter(vectors_config.values()), None)
            vector_size = getattr(first, "size", None)

    return {
        "collection": collection_name,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "segments_count": info.segments_count,
        "status": str(info.status),
        "vector_size": vector_size,
    }


if __name__ == "__main__":
    mcp.run()
