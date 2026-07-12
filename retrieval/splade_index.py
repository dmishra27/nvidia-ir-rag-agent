"""Learned sparse retrieval using SPLADE term-weight vectors.

Mirrors retrieval/bm25_index.py's shape (from_postgres / search / save / load)
but the neural encoder is never persisted — indexed sparse vectors are cheap
to pickle, the transformer is not. Callers must attach_encoder() after load()
before searching. This also lets unit tests inject a tiny deterministic fake
encoder instead of loading torch/transformers, per the project's rule to mock
all embedding/LLM calls in tests.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from retrieval.candidates import Candidate
from schema.models import Chunk, get_engine, get_session_factory

log = structlog.get_logger()

DEFAULT_INDEX_PATH = Path("data/indexes/splade_index.pkl")
DEFAULT_MODEL_ID = "naver/splade-cocondenser-ensemble-distil"
DEFAULT_MAX_LENGTH = 256

# Maps a batch of texts to a batch of sparse term-weight vectors (token -> weight).
SparseEncoder = Callable[[list[str]], list[dict[str, float]]]


def build_default_encoder(
    model_id: str = DEFAULT_MODEL_ID, max_length: int = DEFAULT_MAX_LENGTH
) -> SparseEncoder:
    """Build a real SPLADE encoder backed by a HuggingFace MLM model.

    Imports torch/transformers lazily so importing this module never requires
    them — unit tests inject a fake SparseEncoder instead.
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()

    def encode(texts: list[str]) -> list[dict[str, float]]:
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        weights = torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1)
        pooled, _ = torch.max(weights, dim=1)
        results: list[dict[str, float]] = []
        for row in pooled:
            nz = torch.nonzero(row).squeeze(-1)
            results.append({str(i.item()): float(row[i]) for i in nz.flatten()})
        return results

    return encode


def _sparse_dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b[k] for k, w in a.items() if k in b)


class SpladeIndex:
    """Sparse search over precomputed SPLADE term-weight vectors."""

    def __init__(
        self,
        chunk_ids: list[str],
        texts: list[str],
        vectors: list[dict[str, float]],
        encoder: SparseEncoder | None = None,
    ) -> None:
        if not (len(chunk_ids) == len(texts) == len(vectors)):
            raise ValueError("chunk_ids, texts, and vectors must be the same length")
        self._chunk_ids = chunk_ids
        self._texts = texts
        self._vectors = vectors
        self._encoder = encoder

    def attach_encoder(self, encoder: SparseEncoder) -> None:
        self._encoder = encoder

    @classmethod
    def build(
        cls,
        chunk_ids: list[str],
        texts: list[str],
        encoder: SparseEncoder,
        batch_size: int = 32,
    ) -> SpladeIndex:
        vectors: list[dict[str, float]] = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(encoder(texts[i : i + batch_size]))
        return cls(chunk_ids, texts, vectors, encoder=encoder)

    @classmethod
    def from_postgres(
        cls, session: Session, encoder: SparseEncoder | None = None, batch_size: int = 32
    ) -> SpladeIndex:
        rows = session.execute(select(Chunk.chunk_id, Chunk.chunk_text)).all()
        chunk_ids = [r.chunk_id for r in rows]
        texts = [r.chunk_text for r in rows]
        enc = encoder or build_default_encoder()
        log.info(
            "splade_index_built",
            stage="splade_index",
            query_id="ingestion",
            num_chunks=len(chunk_ids),
        )
        return cls.build(chunk_ids, texts, enc, batch_size=batch_size)

    def search(self, query: str, top_k: int = 10) -> list[Candidate]:
        if top_k <= 0 or not query or not query.strip():
            return []
        if self._encoder is None:
            raise RuntimeError(
                "SpladeIndex has no encoder attached; call attach_encoder() first "
                "(e.g. after load())."
            )
        q_vec = self._encoder([query])[0]
        if not q_vec:
            return []
        scores = [_sparse_dot(q_vec, v) for v in self._vectors]
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            Candidate(
                chunk_id=self._chunk_ids[i],
                text=self._texts[i],
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(ranked_indices, start=1)
        ]

    def save(self, path: str | Path = DEFAULT_INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"chunk_ids": self._chunk_ids, "texts": self._texts, "vectors": self._vectors}, f
            )
        log.info("splade_index_saved", stage="splade_index", query_id="ingestion", path=str(path))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INDEX_PATH) -> SpladeIndex:
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(data["chunk_ids"], data["texts"], data["vectors"])


def build_and_save(path: str | Path = DEFAULT_INDEX_PATH) -> SpladeIndex:
    """Build a fresh SpladeIndex from Postgres using the real model and persist it to disk."""
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        index = SpladeIndex.from_postgres(session)
    index.save(path)
    return index
