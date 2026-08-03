"""NDCG@k, MRR, and Precision@k over a ranked Candidate list and a binary
relevance judgment dict.

All three take the same (ranked, relevance) shape: `ranked` is a Candidate
list in rank order (as returned by any retrieval_agent/reranker), and
`relevance` maps chunk_id -> a 0/1 (or graded) label. A chunk_id absent from
`relevance` is treated as non-relevant — this project's relevance_labels.jsonl
is a sparse, Claude-labelled first pass (evaluation/relevance_labeller.py),
not exhaustive per-query pooling, so "unjudged" and "judged non-relevant"
are deliberately the same case, per standard TREC-style sparse-judgment IR
evaluation.

Pure functions, no I/O — evaluation/benchmark_runner.py is the caller that
wires these to live candidate pools and the relevance judgment file.
"""

from __future__ import annotations

import math

from retrieval.candidates import Candidate


def _relevance_at_ranks(ranked: list[Candidate], relevance: dict[str, int]) -> list[int]:
    return [relevance.get(c.chunk_id, 0) for c in ranked]


def _dcg(gains: list[int]) -> float:
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))


def ndcg_at_k(ranked: list[Candidate], relevance: dict[str, int], k: int = 10) -> float:
    if not ranked or k <= 0:
        return 0.0

    gains = _relevance_at_ranks(ranked[:k], relevance)
    dcg = _dcg(gains)

    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    idcg = _dcg(ideal_gains)

    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: list[Candidate], relevance: dict[str, int]) -> float:
    for i, candidate in enumerate(ranked, start=1):
        if relevance.get(candidate.chunk_id, 0) > 0:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked: list[Candidate], relevance: dict[str, int], k: int) -> float:
    if not ranked or k <= 0:
        return 0.0

    top_k = ranked[:k]
    relevant_count = sum(1 for c in top_k if relevance.get(c.chunk_id, 0) > 0)
    return relevant_count / len(top_k)
