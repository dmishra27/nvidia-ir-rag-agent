from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A single retrieval result, shared across BM25, dense, and RRF fusion."""

    chunk_id: str
    text: str
    score: float
    rank: int
