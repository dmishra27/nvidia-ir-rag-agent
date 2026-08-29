"""Day 6 UAT regression: re-rank cached RRF candidate pools with the real
ms-marco cross-encoder for Q1 and Q12 only.

Memory-safe by design: loads the cached RRF candidate lists from
docs/uat/uat_superiority_cases_raw.json instead of re-running BM25/dense/RRF,
and loads only the small ms-marco cross-encoder (no dense bi-encoder, no
Qdrant connection). Intended for the 8GB dev box per the project's memory
constraint (see AGENTS.md coding standards / host memory notes).
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import json
import sys
from pathlib import Path

from retrieval.candidates import Candidate
from retrieval.reranker_msmarco import MSMarcoReranker

RAW_PATH = Path("docs/uat/uat_superiority_cases_raw.json")
TARGET_QUERY_IDS = {"Q1", "Q12"}


def load_cases() -> list[dict]:
    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    return [case for case in data if case["query_id"] in TARGET_QUERY_IDS]


def to_candidates(pool: list[dict]) -> list[Candidate]:
    return [
        Candidate(chunk_id=c["chunk_id"], text=c["text"], score=c["score"], rank=c["rank"])
        for c in pool
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cases = load_cases()
    reranker = MSMarcoReranker.load()

    results = []
    for case in cases:
        query_id = case["query_id"]
        query = case["query"]
        rrf_pool = to_candidates(case["method_c_rrf"])

        reranked = reranker.rerank(
            query=query,
            candidates=rrf_pool,
            top_k=len(rrf_pool),
            query_id=query_id,
        )
        top = reranked[0]

        print(f"\n=== {query_id}: {case['case']} ===")
        print(f"Query: {query}")
        print(f"RRF-pool size (cached): {len(rrf_pool)}")
        print(f"Rank 1 after ms-marco rerank: chunk_id={top.chunk_id} score={top.score:.4f}")
        print(f"Text: {top.text}")

        results.append(
            {
                "query_id": query_id,
                "case": case["case"],
                "query": query,
                "rrf_pool_size": len(rrf_pool),
                "rank1_chunk_id": top.chunk_id,
                "rank1_score": top.score,
                "rank1_text": top.text,
                "rrf_rank1_chunk_id": case["method_c_rrf"][0]["chunk_id"],
            }
        )

    out_path = Path("docs/uat/uat_day6_regression_raw.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
