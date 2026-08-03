"""Day 9 Task 2+3: live Config A (ms-marco) + Config C (Cohere) benchmark,
memory-safe by reusing cached candidate pools instead of live retrieval.

docs/uat/uat_superiority_cases_raw.json's `method_c_rrf` field already holds
a real, live-computed RRF top-3 pool per query (from Day 5's UAT run) — so
this script never touches BM25/Qdrant/e5-base-v2 again, matching Day 9's
"do NOT load e5-base-v2 or query Qdrant fresh" instruction. Only 15 queries
x 3 candidates are available this way (not the full 50-query x top-100 pool
evaluation/benchmark_runner.py's Config A/C is designed for), so this is a
smaller live smoke benchmark, not a replacement for the full run once
memory allows fresh retrieval.

NDCG@10/MRR/Precision@K need a relevance judgment per pooled candidate, not
just the top-1 evaluation/relevance_labeller.py normally produces — so this
script labels every (query, chunk) pair in the cached pool via Claude
(reusing label_pairs()) rather than the sparse 1-per-query set, writing a
separate evaluation/relevance_labels_superiority_pool.jsonl. This is
additional real Claude-judged ground truth, not a workaround.

Sequenced per Day 9's memory-safe instruction: ms-marco (~90MB, local) is
loaded, run, and released before Cohere (hosted API, no local model) loads.
"""

from __future__ import annotations

import gc
import json
import sys
import uuid
from pathlib import Path

import anthropic
import structlog

from evaluation.benchmark_runner import (
    COHERE_COST_PER_QUERY_USD,
    CONFIG_B_DEFERRED_REASON,
    aggregate,
    log_to_mlflow,
    run_config,
    write_to_postgres,
)
from evaluation.relevance_labeller import BenchmarkQuery, RelevancePair, label_pairs, write_jsonl
from retrieval.candidates import Candidate
from retrieval.reranker_cohere import CohereReranker
from retrieval.reranker_msmarco import MSMarcoReranker
from schema.models import get_engine, get_session_factory

log = structlog.get_logger()

CACHED_POOLS_PATH = Path("docs/uat/uat_superiority_cases_raw.json")
POOL_LABELS_PATH = Path("evaluation/relevance_labels_superiority_pool.jsonl")
CONFIG_A_CONTEXT_PATH = Path("evaluation/day9_config_a_contexts.json")
TOP_K = 3  # cached pools only have 3 candidates per query


def load_cached_pools() -> tuple[list[BenchmarkQuery], dict[str, list[Candidate]]]:
    data = json.loads(CACHED_POOLS_PATH.read_text(encoding="utf-8"))
    queries: list[BenchmarkQuery] = []
    pools: dict[str, list[Candidate]] = {}
    for case in data:
        query_id = case["query_id"]
        queries.append(BenchmarkQuery(query_id=query_id, query=case["query"]))
        pools[query_id] = [
            Candidate(chunk_id=c["chunk_id"], text=c["text"], score=c["score"], rank=c["rank"])
            for c in case["method_c_rrf"]
        ]
    return queries, pools


def label_pool(
    queries: list[BenchmarkQuery], pools: dict[str, list[Candidate]], client: anthropic.Anthropic
) -> dict[str, dict[str, int]]:
    pairs = [
        RelevancePair(query_id=bq.query_id, query=bq.query, chunk_id=c.chunk_id, passage_text=c.text)
        for bq in queries
        for c in pools[bq.query_id]
    ]
    labels = label_pairs(pairs, client)
    write_jsonl(POOL_LABELS_PATH, labels)

    relevance: dict[str, dict[str, int]] = {}
    for label in labels:
        relevance.setdefault(label.query_id, {})[label.chunk_id] = label.label
    return relevance


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run_id = str(uuid.uuid4())[:8]
    session_factory = get_session_factory(get_engine())

    queries, pools = load_cached_pools()
    print(f"Loaded {len(queries)} cached queries from {CACHED_POOLS_PATH}, {TOP_K} candidates each.")

    client = anthropic.Anthropic()
    relevance = label_pool(queries, pools, client)
    num_judgments = sum(len(v) for v in relevance.values())
    print(f"Labelled pool: {num_judgments} (query, chunk) judgments -> {POOL_LABELS_PATH}")

    # Config A: ms-marco (~90MB local model) — loaded, run, released before Cohere.
    msmarco = MSMarcoReranker.load()
    rows_a = run_config(
        "config_A_ms_marco", msmarco.rerank, queries, pools, relevance, top_k=TOP_K, cost_per_query=0.0
    )
    log_to_mlflow("config_A_ms_marco", rows_a)
    written_a = write_to_postgres(rows_a, run_id, session_factory)
    print(f"config_A_ms_marco: {aggregate(rows_a)} ({written_a} rows written to benchmark_results)")

    # Persist Config A's reranked contexts for Day 9 Task 4 (RAGAS) reuse,
    # before releasing the model — avoids a second live retrieve+rerank pass.
    contexts = {
        bq.query_id: {
            "query": bq.query,
            "reranked": [
                {"chunk_id": c.chunk_id, "text": c.text, "score": c.score, "rank": c.rank}
                for c in msmarco.rerank(bq.query, pools[bq.query_id], top_k=TOP_K, query_id=bq.query_id)
            ],
        }
        for bq in queries
    }
    CONFIG_A_CONTEXT_PATH.write_text(json.dumps(contexts, indent=2), encoding="utf-8")
    print(f"Saved Config A reranked contexts -> {CONFIG_A_CONTEXT_PATH}")

    del msmarco
    gc.collect()

    # Config C: Cohere Rerank v3 — hosted API, no local model.
    cohere_reranker = CohereReranker.load()
    rows_c = run_config(
        "config_C_cohere_rerank",
        cohere_reranker.rerank,
        queries,
        pools,
        relevance,
        top_k=TOP_K,
        cost_per_query=COHERE_COST_PER_QUERY_USD,
    )
    log_to_mlflow("config_C_cohere_rerank", rows_c)
    written_c = write_to_postgres(rows_c, run_id, session_factory)
    print(f"config_C_cohere_rerank: {aggregate(rows_c)} ({written_c} rows written to benchmark_results)")

    print(f"\nConfig B skipped: {CONFIG_B_DEFERRED_REASON}")
    print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
