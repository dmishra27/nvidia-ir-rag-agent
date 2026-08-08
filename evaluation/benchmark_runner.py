"""Three-way re-ranker benchmark, per SKILLS.md's run-reranker-benchmark
pattern: the same 50-query set and the same RRF top-100 candidate pool per
query is fed to every config, so differences in NDCG@10/MRR/Precision@K
isolate the re-ranker itself, not the first-stage retrieval.

Only Config A (ms-marco, local cross-encoder) and Config C (Cohere Rerank
v3, hosted API) are run here — Config B (bge-reranker-v2-m3 via
sentence-transformers) OOMs at model load on this machine's 8GB RAM /
CPU-only constraint (see CONFIG_B_DEFERRED_REASON) and is deferred to a GPU
environment, matching the project's established pattern of documenting
(not silently skipping) hardware-blocked work — see Day 6's UAT regression
deferral in docs/daily_progress/day_06_storyline.md.

Ground truth comes from evaluation/relevance_labeller.py's sparse,
Claude-labelled relevance_labels.jsonl (one judged passage per query);
evaluation/retrieval_metrics.py treats every unjudged chunk_id as
non-relevant, per standard TREC-style sparse-judgment evaluation.

Results are logged twice, per SKILLS.md ("Log per-config per-query to
MLflow experiment 'reranker_benchmark'") and AGENTS.md ("SQLAlchemy ORM for
all database writes"): one MLflow run per config with per-query metric
history plus run-level aggregates, and one benchmark_results row per
(config, query) via the BenchmarkResults ORM model.
"""

from __future__ import annotations

import io
import sys
import uuid
from pathlib import Path
from typing import Callable

import mlflow
import structlog
from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from evaluation.relevance_labeller import (
    BENCHMARK_QUERIES_PATH,
    RELEVANCE_LABELS_PATH,
    BenchmarkQuery,
    RelevanceLabel,
)
from evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k
from retrieval.bm25_index import BM25Index
from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex
from retrieval.reranker_cohere import CohereReranker
from retrieval.reranker_msmarco import MSMarcoReranker
from retrieval.rrf_fusion import fuse
from schema.models import BenchmarkResults, get_engine, get_session_factory

load_dotenv()
log = structlog.get_logger()

MLFLOW_EXPERIMENT = "reranker_benchmark"
CANDIDATE_POOL_SIZE = 100
TOP_K = 10

# Cohere Rerank v3 list pricing is ~$2.00 / 1,000 search units (one search
# unit = one rerank call over <=100 documents); this project's calls stay
# under that per-call document limit, so cost is a flat per-query estimate,
# not metered from a live invoice.
COHERE_COST_PER_QUERY_USD = 0.002

CONFIG_B_DEFERRED_REASON = (
    "Config B (bge-reranker-v2-m3 via sentence-transformers) is hardware-blocked "
    "on this machine: loading the model OOMs under the 8GB RAM / CPU-only "
    "constraint documented in AGENTS.md. Deferred to a GPU environment; not run "
    "by this benchmark."
)

# (query, candidates, top_k, query_id) -> ranked candidates — same shape as
# retrieval/reranker_router.py's RerankFn.
RerankFn = Callable[[str, list[Candidate], int, str], list[Candidate]]


class BenchmarkRow(BaseModel):
    config: str
    query_id: str
    ndcg_at_10: float
    mrr: float
    prec_at_3: float
    prec_at_5: float
    prec_at_10: float
    latency_ms: float
    cost_usd: float


# ── Input loading ────────────────────────────────────────────────────────


def load_benchmark_queries(path: Path = BENCHMARK_QUERIES_PATH) -> list[BenchmarkQuery]:
    return [
        BenchmarkQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_relevance_labels(path: Path = RELEVANCE_LABELS_PATH) -> dict[str, dict[str, int]]:
    relevance: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = RelevanceLabel.model_validate_json(line)
        relevance.setdefault(item.query_id, {})[item.chunk_id] = item.label
    return relevance


# ── Shared candidate pool: identical input to every config ─────────────────


def build_candidate_pools(
    queries: list[BenchmarkQuery],
    bm25_index: BM25Index,
    dense_index: DenseIndex,
    pool_size: int = CANDIDATE_POOL_SIZE,
) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = {}
    for bq in queries:
        log.info("build_candidate_pools", query_id=bq.query_id, stage="benchmark_runner")
        bm25_results = bm25_index.search(bq.query, top_k=pool_size)
        dense_results = dense_index.search(bq.query, top_k=pool_size)
        pools[bq.query_id] = fuse(bm25_results, dense_results, top_k=pool_size)
    return pools


# ── Per-config run ──────────────────────────────────────────────────────


def run_config(
    config: str,
    rerank_fn: RerankFn,
    queries: list[BenchmarkQuery],
    pools: dict[str, list[Candidate]],
    relevance: dict[str, dict[str, int]],
    top_k: int = TOP_K,
    cost_per_query: float = 0.0,
) -> list[BenchmarkRow]:
    import time

    rows: list[BenchmarkRow] = []
    for bq in queries:
        pool = pools.get(bq.query_id)
        if not pool:
            log.warning("run_config_no_pool", query_id=bq.query_id, stage="benchmark_runner", config=config)
            continue
        query_relevance = relevance.get(bq.query_id, {})

        start = time.perf_counter()
        ranked = rerank_fn(bq.query, pool, top_k, bq.query_id)
        latency_ms = (time.perf_counter() - start) * 1000

        rows.append(
            BenchmarkRow(
                config=config,
                query_id=bq.query_id,
                ndcg_at_10=ndcg_at_k(ranked, query_relevance, k=10),
                mrr=mrr(ranked, query_relevance),
                prec_at_3=precision_at_k(ranked, query_relevance, k=3),
                prec_at_5=precision_at_k(ranked, query_relevance, k=5),
                prec_at_10=precision_at_k(ranked, query_relevance, k=10),
                latency_ms=latency_ms,
                cost_usd=cost_per_query,
            )
        )
        log.info(
            "run_config_query_done",
            query_id=bq.query_id,
            stage="benchmark_runner",
            config=config,
            latency_ms=round(latency_ms, 2),
        )
    return rows


def aggregate(rows: list[BenchmarkRow]) -> dict[str, float]:
    keys = ["ndcg_at_10", "mrr", "prec_at_3", "prec_at_5", "prec_at_10", "latency_ms", "cost_usd"]
    if not rows:
        return {k: 0.0 for k in keys}
    n = len(rows)
    result = {k: sum(getattr(r, k) for r in rows) / n for k in keys}
    result["cost_usd"] = sum(r.cost_usd for r in rows)
    return result


# ── Logging: MLflow + Postgres ──────────────────────────────────────────


def log_to_mlflow(config: str, rows: list[BenchmarkRow], experiment: str = MLFLOW_EXPERIMENT) -> None:
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=config):
        for i, row in enumerate(rows):
            mlflow.log_metric("ndcg_at_10", row.ndcg_at_10, step=i)
            mlflow.log_metric("mrr", row.mrr, step=i)
            mlflow.log_metric("prec_at_3", row.prec_at_3, step=i)
            mlflow.log_metric("prec_at_5", row.prec_at_5, step=i)
            mlflow.log_metric("prec_at_10", row.prec_at_10, step=i)
            mlflow.log_metric("latency_ms", row.latency_ms, step=i)

        agg = aggregate(rows)
        mlflow.log_metrics({f"mean_{k}": v for k, v in agg.items() if k != "cost_usd"})
        mlflow.log_metric("total_cost_usd", agg["cost_usd"])
        mlflow.log_param("config", config)
        mlflow.log_param("num_queries", len(rows))


def write_to_postgres(rows: list[BenchmarkRow], run_id: str, session_factory: sessionmaker[Session]) -> int:
    written = 0
    try:
        with session_factory() as session:
            for row in rows:
                session.add(
                    BenchmarkResults(
                        result_id=str(uuid.uuid4()),
                        run_id=run_id,
                        config=row.config,
                        query_id=row.query_id,
                        ndcg_at_10=row.ndcg_at_10,
                        mrr=row.mrr,
                        prec_at_3=row.prec_at_3,
                        prec_at_5=row.prec_at_5,
                        prec_at_10=row.prec_at_10,
                        latency_ms=row.latency_ms,
                        cost_usd=row.cost_usd,
                    )
                )
                written += 1
            session.commit()
    except Exception as exc:
        log.error("write_to_postgres_failed", stage="benchmark_runner", run_id=run_id, exc=str(exc))
        return 0
    return written


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    run_id = str(uuid.uuid4())[:8]

    bm25_index = BM25Index.load()
    dense_index = DenseIndex.connect()
    queries = load_benchmark_queries()
    relevance = load_relevance_labels()
    pools = build_candidate_pools(queries, bm25_index, dense_index)

    msmarco = MSMarcoReranker.load()
    cohere_reranker = CohereReranker.load()
    session_factory = get_session_factory(get_engine())

    configs: list[tuple[str, RerankFn, float]] = [
        ("config_A_ms_marco", msmarco.rerank, 0.0),
        ("config_C_cohere_rerank", cohere_reranker.rerank, COHERE_COST_PER_QUERY_USD),
    ]

    for config, rerank_fn, cost_per_query in configs:
        rows = run_config(config, rerank_fn, queries, pools, relevance, cost_per_query=cost_per_query)
        log_to_mlflow(config, rows)
        written = write_to_postgres(rows, run_id, session_factory)
        print(f"{config}: {aggregate(rows)} ({written} rows written to benchmark_results)")

    print(f"\nConfig B skipped: {CONFIG_B_DEFERRED_REASON}")


if __name__ == "__main__":
    main()
