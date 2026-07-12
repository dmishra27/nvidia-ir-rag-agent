"""Run live BM25 + dense + RRF hybrid search and compare against BM25-only.

Loads the persisted BM25 index (Day 3) and connects to the live Qdrant
collection (Day 4) for dense search, fuses both with RRF (Day 5), and prints
top-5 results per query for BM25-only vs RRF-fused so the two can be
compared side by side.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from retrieval.bm25_index import DEFAULT_INDEX_PATH, BM25Index
from retrieval.dense_index import DenseIndex
from retrieval.rrf_fusion import fuse

QUERIES = [
    "NVLink bandwidth",
    "CUDA memory allocation",
]

CANDIDATE_POOL_SIZE = 100
TOP_K_DISPLAY = 5

SEP = "=" * 72


def _snippet(text: str, width: int = 160) -> str:
    text = " ".join(text.split())
    return text[:width] + ("..." if len(text) > width else "")


def _print_results(label: str, results) -> None:
    print(f"\n  [{label}]")
    for c in results[:TOP_K_DISPLAY]:
        print(f"    rank={c.rank}  chunk_id={c.chunk_id}  score={c.score:.4f}")
        print(f"      {_snippet(c.text)}")


def main() -> None:
    bm25 = BM25Index.load(DEFAULT_INDEX_PATH)
    dense = DenseIndex.connect()

    for query in QUERIES:
        print(f"\n{SEP}")
        print(f"Query: {query!r}")
        print(SEP)

        bm25_results = bm25.search(query, top_k=CANDIDATE_POOL_SIZE)
        dense_results = dense.search(query, top_k=CANDIDATE_POOL_SIZE)
        rrf_results = fuse(bm25_results, dense_results, top_k=TOP_K_DISPLAY)

        _print_results("BM25-only (Day 3)", bm25_results)
        _print_results("Dense-only (e5-base-v2)", dense_results)
        _print_results("RRF fused (Day 5)", rrf_results)

        bm25_top5_ids = [c.chunk_id for c in bm25_results[:TOP_K_DISPLAY]]
        rrf_top5_ids = [c.chunk_id for c in rrf_results[:TOP_K_DISPLAY]]
        overlap = len(set(bm25_top5_ids) & set(rrf_top5_ids))
        print(f"\n  BM25-only vs RRF top-5 overlap: {overlap}/{TOP_K_DISPLAY}")
        if rrf_top5_ids != bm25_top5_ids:
            print("  RRF re-ranked or introduced new chunks vs BM25-only.")
        else:
            print("  RRF top-5 identical to BM25-only top-5.")

    print(f"\n{SEP}")
    print("Hybrid search complete.")


if __name__ == "__main__":
    main()
