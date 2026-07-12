"""Reciprocal Rank Fusion over ranked candidate lists.

Combines any number of ranked lists that already share the Candidate shape
(BM25, SPLADE, dense, ...) into a single fused ranking, using each list's
own `rank` field rather than its list position — callers can hand in a
truncated or reordered slice and RRF still scores it correctly.
"""

from __future__ import annotations

from retrieval.candidates import Candidate

DEFAULT_K = 60


def fuse(
    bm25_results: list[Candidate],
    dense_results: list[Candidate],
    top_k: int = 10,
    k: int = DEFAULT_K,
) -> list[Candidate]:
    """RRF(d) = sum(1 / (k + rank_i)) over every list d appears in."""
    if top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    first_seen_order: list[str] = []

    for results in (bm25_results, dense_results):
        for candidate in results:
            if candidate.chunk_id not in rrf_scores:
                rrf_scores[candidate.chunk_id] = 0.0
                texts[candidate.chunk_id] = candidate.text
                first_seen_order.append(candidate.chunk_id)
            rrf_scores[candidate.chunk_id] += 1.0 / (k + candidate.rank)

    ranked_ids = sorted(first_seen_order, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    return [
        Candidate(chunk_id=cid, text=texts[cid], score=rrf_scores[cid], rank=rank)
        for rank, cid in enumerate(ranked_ids, start=1)
    ]
