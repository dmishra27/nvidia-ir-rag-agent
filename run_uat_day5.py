"""Day 5 UAT runner: BM25-only vs Dense-only vs RRF-hybrid across 9 queries.

Loads the persisted BM25 index and the live Qdrant dense index, retrieves a
top-100 candidate pool per signal for each query, fuses with RRF, and dumps
the top-3 per config (chunk_id, score, text) to a JSON file. Relevance
judgment and the summary table are composed separately from this raw data,
not by this script.
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
OUTPUT_PATH = Path("docs/uat/uat_day5_raw.json")

QUERIES = [
    ("type1_exact_technical", "NVLink 4.0 bandwidth specifications"),
    ("type1_exact_technical", "CUDA cudaMalloc function parameters"),
    ("type1_exact_technical", "H100 HBM2e memory capacity"),
    ("type2_semantic_conceptual", "How does GPU memory work for parallel processing"),
    ("type2_semantic_conceptual", "best practices for optimising neural network training"),
    ("type2_semantic_conceptual", "what causes memory errors in GPU applications"),
    ("type3_legacy_terminology", "shader processor count"),
    ("type3_legacy_terminology", "global memory coalescing techniques"),
    ("type3_legacy_terminology", "warp divergence performance impact"),
]


def _dump(candidates) -> list[dict]:
    return [
        {
            "rank": c.rank,
            "chunk_id": c.chunk_id,
            "score": round(c.score, 4),
            "text_100": " ".join(c.text.split())[:100],
        }
        for c in candidates[:TOP_K_DISPLAY]
    ]


def main() -> None:
    bm25 = BM25Index.load(DEFAULT_INDEX_PATH)
    dense = DenseIndex.connect()

    results = []
    for query_type, query in QUERIES:
        bm25_pool = bm25.search(query, top_k=CANDIDATE_POOL_SIZE)
        dense_pool = dense.search(query, top_k=CANDIDATE_POOL_SIZE)
        rrf_top = fuse(bm25_pool, dense_pool, top_k=TOP_K_DISPLAY)

        results.append(
            {
                "query_type": query_type,
                "query": query,
                "config_a_bm25": _dump(bm25_pool),
                "config_b_dense": _dump(dense_pool),
                "config_c_rrf": _dump(rrf_top),
            }
        )
        print(f"done: {query!r}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} query results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
