"""Populate the Qdrant collection with the winning bi-encoder's dense embeddings.

Reads the winner (model + hf_id) from biencoder_eval_results.json, embeds
every chunk in Postgres with that model, and upserts into a Qdrant
collection sized to match its embedding dimension.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import structlog
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from retrieval.biencoder_eval import MODELS, load_corpus_with_doc_ids

load_dotenv()
log = structlog.get_logger()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "nvidia_ir_chunks"
EVAL_RESULTS_PATH = Path("biencoder_eval_results.json")
EMBEDDINGS_CACHE_PATH = Path("data/indexes/qdrant_corpus_embeddings.npy")


def _winning_config():
    data = json.loads(EVAL_RESULTS_PATH.read_text())
    winner_label = data["winner"]
    for r in data["results"]:
        if r["model"] == winner_label:
            for c in MODELS:
                if c.label == winner_label:
                    return c
    raise RuntimeError(f"Winner {winner_label!r} not found in MODELS config")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = _winning_config()
    log.info(
        "populate_qdrant_start", stage="populate_qdrant", query_id="ingestion",
        model=config.label, hf_id=config.hf_id,
    )

    chunk_ids, texts, doc_ids = load_corpus_with_doc_ids()
    model = SentenceTransformer(config.hf_id, device="cpu")

    if EMBEDDINGS_CACHE_PATH.exists():
        log.info("populate_qdrant_cache_hit", stage="populate_qdrant", query_id="ingestion")
        embeddings = np.load(EMBEDDINGS_CACHE_PATH)
    else:
        passages = [config.passage_prefix + t for t in texts]
        t0 = time.perf_counter()
        embeddings = model.encode(
            passages,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        log.info(
            "populate_qdrant_embedded", stage="populate_qdrant", query_id="ingestion",
            seconds=round(time.perf_counter() - t0, 2),
        )
        EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_CACHE_PATH, embeddings)

    client = QdrantClient(url=QDRANT_URL)
    vector_size = embeddings.shape[1]
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        log.info(
            "populate_qdrant_collection_created", stage="populate_qdrant", query_id="ingestion",
            collection=COLLECTION_NAME, vector_size=vector_size,
        )

    upsert_batch = 256
    for i in range(0, len(chunk_ids), upsert_batch):
        points = [
            PointStruct(
                id=i + j,
                vector=embeddings[i + j].tolist(),
                payload={
                    "chunk_id": chunk_ids[i + j],
                    "doc_id": doc_ids[i + j],
                    "text": texts[i + j],
                },
            )
            for j in range(min(upsert_batch, len(chunk_ids) - i))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    log.info(
        "populate_qdrant_done", stage="populate_qdrant", query_id="ingestion",
        collection=COLLECTION_NAME, num_points=len(chunk_ids),
    )
    print(f"Upserted {len(chunk_ids)} points into Qdrant collection {COLLECTION_NAME!r} ({config.label}, dim={vector_size}).")


if __name__ == "__main__":
    main()
