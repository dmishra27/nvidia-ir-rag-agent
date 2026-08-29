"""Day 9 Task 1: live relevance labelling for the 50-query benchmark set.

Memory-safe variant of evaluation/relevance_labeller.py's main(): the
tested build_pairs() takes a BM25 index AND a dense index (DenseIndex.
connect() loads e5-base-v2 via sentence-transformers/torch), but this
session's free memory (0.06-0.51GB of 7.65GB) rules out any torch model
load for this task. _NullDenseIndex stands in for DenseIndex — its
search() returns [], so retrieval/rrf_fusion.py's fuse() degenerates to a
pure BM25 ranking (already an exercised path: see
tests/retrieval/test_rrf_fusion.py's "doc in only bm25" cases). No
modification to evaluation/relevance_labeller.py was needed; only the
dense_index argument is swapped, mirroring run_uat_day6_regression.py's
pattern of orchestrating tested library code differently for a
memory-constrained live run rather than editing the library itself.
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import sys

import anthropic
import structlog

from evaluation.relevance_labeller import (
    BENCHMARK_QUERIES,
    BENCHMARK_QUERIES_PATH,
    RELEVANCE_LABELS_PATH,
    build_pairs,
    label_pairs,
    write_jsonl,
)
from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate

log = structlog.get_logger()


class _NullDenseIndex:
    """Stands in for DenseIndex so build_pairs() never loads e5-base-v2/torch."""

    def search(self, query: str, top_k: int) -> list[Candidate]:
        return []


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    bm25_index = BM25Index.load()
    print(f"Loaded BM25 index ({len(BENCHMARK_QUERIES)} queries to retrieve).")

    pairs = build_pairs(BENCHMARK_QUERIES, bm25_index, _NullDenseIndex())
    print(f"Built {len(pairs)} (query, passage) pairs via BM25-only retrieval.")

    write_jsonl(BENCHMARK_QUERIES_PATH, BENCHMARK_QUERIES)
    print(f"Wrote {len(BENCHMARK_QUERIES)} queries to {BENCHMARK_QUERIES_PATH}")

    client = anthropic.Anthropic()
    labels = label_pairs(pairs, client)
    write_jsonl(RELEVANCE_LABELS_PATH, labels)
    print(f"Wrote {len(labels)} relevance labels to {RELEVANCE_LABELS_PATH}")

    positives = sum(1 for label in labels if label.label == 1)
    print(f"Label distribution: {positives}/{len(labels)} relevant")


if __name__ == "__main__":
    main()
