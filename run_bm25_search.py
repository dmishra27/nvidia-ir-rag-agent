"""Run live BM25 search queries against the persisted index built from Postgres chunks."""
from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval.bm25_index import DEFAULT_INDEX_PATH, BM25Index

QUERIES = [
    "NVLink bandwidth",
    "CUDA memory allocation",
]

SEP = "=" * 72


def _snippet(text: str, width: int = 160) -> str:
    text = " ".join(text.split())
    return text[:width] + ("..." if len(text) > width else "")


def main() -> None:
    index = BM25Index.load(DEFAULT_INDEX_PATH)

    for query in QUERIES:
        print(f"\n{SEP}")
        print(f"Query: {query!r}")
        print(SEP)
        results = index.search(query, top_k=5)
        for c in results:
            print(f"  rank={c.rank}  chunk_id={c.chunk_id}  score={c.score:.4f}")
            print(f"    {_snippet(c.text)}")

    print(f"\n{SEP}")
    print("Search complete.")


if __name__ == "__main__":
    main()
