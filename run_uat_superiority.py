"""Retrieval superiority UAT: BM25 vs Dense vs RRF across 15 queries, 6 cases.

Mirrors run_uat_day5.py's pattern (persisted BM25 index + live Qdrant dense
index + RRF fusion over a top-100 candidate pool per signal), but targets
15 queries deliberately chosen to demonstrate each method's structural
strengths and weaknesses (lexical exactness, semantic vocabulary gaps, and
hybrid corroboration) rather than a general-purpose query sample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval.bm25_index import DEFAULT_INDEX_PATH, BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.rrf_fusion import fuse

CANDIDATE_POOL_SIZE = 100
TOP_K_DISPLAY = 3
OUTPUT_PATH = Path("docs/uat/uat_superiority_cases_raw.json")

QUERIES = [
    ("case1_bm25_lexical_superiority", "Q1", "CUDA cudaMalloc function parameters"),
    ("case1_bm25_lexical_superiority", "Q2", "cudaMemcpyAsync stream parameter"),
    ("case1_bm25_lexical_superiority", "Q3", "CUDA error cudaErrorInvalidValue description"),
    ("case2_dense_semantic_superiority", "Q4", "shader processor count per streaming multiprocessor"),
    ("case2_dense_semantic_superiority", "Q5", "how to make GPU programs run faster"),
    ("case2_dense_semantic_superiority", "Q6", "problems with threads executing different code paths"),
    ("case3_rrf_hybrid_superiority", "Q7", "CUDA thread synchronization performance overhead"),
    ("case3_rrf_hybrid_superiority", "Q8", "shared memory bank conflicts and how to avoid them"),
    ("case3_rrf_hybrid_superiority", "Q9", "memory coalescing rules for global memory access patterns"),
    ("case4_bm25_failure_dense_advantage", "Q10", "latency hiding through instruction level parallelism"),
    ("case4_bm25_failure_dense_advantage", "Q11", "occupancy versus performance tradeoffs"),
    ("case5_dense_failure_bm25_advantage", "Q12", "cudaDeviceSynchronize return value"),
    ("case5_dense_failure_bm25_advantage", "Q13", "dim3 struct constructor syntax"),
    ("case6_rrf_hybrid_advantage_mixed", "Q14", "pinned memory cudaMallocHost benefits and when to use"),
    ("case6_rrf_hybrid_advantage_mixed", "Q15", "register pressure and its effect on occupancy"),
]


def _dump(candidates) -> list[dict]:
    out = []
    for c in candidates[:TOP_K_DISPLAY]:
        char_limit = 150 if c.rank == 1 else 100
        out.append(
            {
                "rank": c.rank,
                "chunk_id": c.chunk_id,
                "score": round(c.score, 4),
                "text": " ".join(c.text.split())[:char_limit],
            }
        )
    return out


def main() -> None:
    bm25 = BM25Index.load(DEFAULT_INDEX_PATH)
    dense = DenseIndex.connect()

    results = []
    for case, qid, query in QUERIES:
        bm25_pool = bm25.search(query, top_k=CANDIDATE_POOL_SIZE)
        dense_pool = dense.search(query, top_k=CANDIDATE_POOL_SIZE)
        rrf_top = fuse(bm25_pool, dense_pool, top_k=TOP_K_DISPLAY)

        results.append(
            {
                "case": case,
                "query_id": qid,
                "query": query,
                "method_a_bm25": _dump(bm25_pool),
                "method_b_dense": _dump(dense_pool),
                "method_c_rrf": _dump(rrf_top),
            }
        )
        print(f"done: {qid} {query!r}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} query results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
