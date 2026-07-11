from __future__ import annotations

import pickle
import re
from pathlib import Path

import structlog
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from retrieval.candidates import Candidate
from schema.models import Chunk, get_engine, get_session_factory

log = structlog.get_logger()

DEFAULT_INDEX_PATH = Path("data/indexes/bm25_index.pkl")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class BM25Index:
    """BM25Okapi search over chunk_text, sourced from Postgres and persistable to disk."""

    def __init__(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self._chunk_ids = chunk_ids
        self._texts = texts
        tokenized_corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @classmethod
    def from_postgres(cls, session: Session) -> BM25Index:
        rows = session.execute(select(Chunk.chunk_id, Chunk.chunk_text)).all()
        chunk_ids = [r.chunk_id for r in rows]
        texts = [r.chunk_text for r in rows]
        log.info(
            "bm25_index_built",
            stage="bm25_index",
            query_id="ingestion",
            num_chunks=len(chunk_ids),
        )
        return cls(chunk_ids, texts)

    def search(self, query: str, top_k: int = 10) -> list[Candidate]:
        if top_k <= 0:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
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
                {"chunk_ids": self._chunk_ids, "texts": self._texts, "bm25": self._bm25}, f
            )
        log.info("bm25_index_saved", stage="bm25_index", query_id="ingestion", path=str(path))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INDEX_PATH) -> BM25Index:
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance._chunk_ids = data["chunk_ids"]
        instance._texts = data["texts"]
        instance._bm25 = data["bm25"]
        return instance


def build_and_save(path: str | Path = DEFAULT_INDEX_PATH) -> BM25Index:
    """Build a fresh BM25Index from Postgres and persist it to disk."""
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        index = BM25Index.from_postgres(session)
    index.save(path)
    return index
