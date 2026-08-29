"""Experiment A5 prep: full top-100 Config A (ms-marco) vs Config C (Cohere)
orderings for the 15-query/6-case Round 2 UAT set (docs/uat/uat_superiority_
cases_raw.json), so A5's head-weighted follow-up (Spearman@10, Overlap@10/@5,
top-1 match, RBO p=0.9 -- see run_a5_head_weighted_analysis.py) can run on
genuine retained data instead of the 3-candidate pools Day 9's
run_day9_benchmark_ac.py was limited to (TOP_K=3 there).

Sequenced per the host's memory-safety rule (one heavy thing at a time, see
[[user-host-memory-constraint]]):
  1. BM25 top-100 -- local pickle, no live infra.
  2. Dense top-100 -- live Qdrant + e5-base-v2 query encoder (DenseIndex.connect()).
  3. RRF top-100 fuse over both.
  4. Config A (ms-marco, local cross-encoder) reranks the full pool
     (top_k=len(pool)) to get a complete reordering, not just top-3; model
     released before Cohere loads.
  5. Config C (Cohere Rerank v3) reranks the same full pool in one API call
     per query (top_n=len(pool), <=100 fits Cohere's per-call document
     limit), throttled to the 10-calls/min trial-key limit (same fix as
     run_day9_benchmark_c_fixed_latency.py).

Persists the RRF pool and both full orderings to evaluation/ so the analysis
step never needs to touch retrieval/reranking again.
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import gc
import json
import time
from pathlib import Path

from retrieval.bm25_index import DEFAULT_INDEX_PATH, BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.reranker_cohere import CohereReranker
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.rrf_fusion import fuse

CASES_PATH = Path("docs/uat/uat_superiority_cases_raw.json")
POOL_PATH = Path("evaluation/a5_top100_pools.json")
ORDERINGS_PATH = Path("evaluation/a5_top100_orderings.json")
POOL_SIZE = 100
COHERE_THROTTLE_SECONDS = 6.5  # >6.0s/call keeps 15 calls under the 10/min trial-key cap


def load_queries() -> list[dict]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return [{"query_id": c["query_id"], "query": c["query"], "case": c["case"]} for c in data]


def build_pools(queries: list[dict], bm25: BM25Index, dense: DenseIndex) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = {}
    for q in queries:
        bm25_hits = bm25.search(q["query"], top_k=POOL_SIZE)
        dense_hits = dense.search(q["query"], top_k=POOL_SIZE)
        fused = fuse(bm25_hits, dense_hits, top_k=POOL_SIZE)
        pools[q["query_id"]] = fused
        print(f"{q['query_id']}: bm25={len(bm25_hits)} dense={len(dense_hits)} fused={len(fused)}")
    return pools


def save_pools(queries: list[dict], pools: dict[str, list[Candidate]]) -> None:
    out = {
        q["query_id"]: {
            "query": q["query"],
            "case": q["case"],
            "pool": [
                {"chunk_id": c.chunk_id, "rank": c.rank, "score": c.score} for c in pools[q["query_id"]]
            ],
        }
        for q in queries
    }
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved RRF top-{POOL_SIZE} pool -> {POOL_PATH}")


def main() -> None:
    queries = load_queries()
    num_cases = len({q["case"] for q in queries})
    print(f"Loaded {len(queries)} queries across {num_cases} cases from {CASES_PATH}.")

    print("Loading BM25 index (local pickle, no live infra)...")
    bm25 = BM25Index.load(DEFAULT_INDEX_PATH)

    print("Connecting to live Qdrant + loading e5-base-v2 query encoder...")
    dense = DenseIndex.connect()

    pools = build_pools(queries, bm25, dense)
    save_pools(queries, pools)

    del dense
    gc.collect()

    orderings: dict[str, dict] = {
        q["query_id"]: {"query": q["query"], "case": q["case"]} for q in queries
    }

    print("Loading ms-marco cross-encoder (Config A)...")
    msmarco = MSMarcoReranker.load()
    for q in queries:
        pool = pools[q["query_id"]]
        reranked = msmarco.rerank(q["query"], pool, top_k=len(pool), query_id=q["query_id"])
        orderings[q["query_id"]]["config_a"] = [c.chunk_id for c in reranked]
        print(f"{q['query_id']}: Config A reranked {len(reranked)} of {len(pool)}")
    del msmarco
    gc.collect()

    print("Loading Cohere client (Config C)...")
    cohere_reranker = CohereReranker.load()
    for i, q in enumerate(queries):
        pool = pools[q["query_id"]]
        reranked = cohere_reranker.rerank(q["query"], pool, top_k=len(pool), query_id=q["query_id"])
        orderings[q["query_id"]]["config_c"] = [c.chunk_id for c in reranked]
        print(f"{q['query_id']}: Config C reranked {len(reranked)} of {len(pool)}")
        if i < len(queries) - 1:
            time.sleep(COHERE_THROTTLE_SECONDS)

    ORDERINGS_PATH.write_text(json.dumps(orderings, indent=2), encoding="utf-8")
    print(f"Saved full top-{POOL_SIZE} orderings -> {ORDERINGS_PATH}")


if __name__ == "__main__":
    main()
