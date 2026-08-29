"""Day 9 Task 3 correction: Config C (Cohere) with honest latency_ms.

run_day9_benchmark_c_retry.py's throttle wrapper slept *inside* the
function run_config() times as `rerank_fn(...)`, so its logged latency_ms
(~6.0-8.8s/query) was almost entirely the 6.5s rate-limit wait, not real
Cohere API latency — Q1 (the one unthrottled call, 350ms) was the only
honest sample in that run. This script reruns Config C with the same
throttling (still required — Cohere trial key is 10 calls/min) but times
only the rerank() call itself, sleeping between calls outside the timed
region, so latency_ms reflects real API latency. NDCG@10/MRR/Precision@K
are unaffected by the bug (they don't depend on wall-clock time) and are
expected to reproduce run_day9_benchmark_c_retry.py's values.

Both runs are left in MLflow/benchmark_results rather than deleted, per
this project's practice of documenting mistakes rather than erasing them
(see day_09_storyline.md) — this run's run_id is the one to read for
Config C's true latency_ms.
"""

from __future__ import annotations

# F-04 / F-07: fail fast on the wrong interpreter before any third-party
# import does so with an error that never names the version. See
# utils/require_python.py and docs/uat/clean_clone_test_findings.md.
from utils.require_python import require_python

require_python()

import sys
import time
import uuid

from evaluation.benchmark_runner import BenchmarkRow, COHERE_COST_PER_QUERY_USD, aggregate, log_to_mlflow, write_to_postgres
from evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k
from retrieval.reranker_cohere import CohereReranker
from run_day9_benchmark_ac import load_cached_pools
from run_day9_benchmark_c_retry import COHERE_TRIAL_MIN_INTERVAL_S, load_pool_relevance
from schema.models import get_engine, get_session_factory

TOP_K = 3


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run_id = str(uuid.uuid4())[:8]
    session_factory = get_session_factory(get_engine())

    queries, pools = load_cached_pools()
    relevance = load_pool_relevance()

    cohere_reranker = CohereReranker.load()

    rows: list[BenchmarkRow] = []
    last_call = 0.0
    for bq in queries:
        pool = pools[bq.query_id]
        query_relevance = relevance.get(bq.query_id, {})

        elapsed = time.monotonic() - last_call
        if elapsed < COHERE_TRIAL_MIN_INTERVAL_S:
            time.sleep(COHERE_TRIAL_MIN_INTERVAL_S - elapsed)

        start = time.perf_counter()
        ranked = cohere_reranker.rerank(bq.query, pool, top_k=TOP_K, query_id=bq.query_id)
        latency_ms = (time.perf_counter() - start) * 1000
        last_call = time.monotonic()

        rows.append(
            BenchmarkRow(
                config="config_C_cohere_rerank",
                query_id=bq.query_id,
                ndcg_at_10=ndcg_at_k(ranked, query_relevance, k=10),
                mrr=mrr(ranked, query_relevance),
                prec_at_3=precision_at_k(ranked, query_relevance, k=3),
                prec_at_5=precision_at_k(ranked, query_relevance, k=5),
                prec_at_10=precision_at_k(ranked, query_relevance, k=10),
                latency_ms=latency_ms,
                cost_usd=COHERE_COST_PER_QUERY_USD,
            )
        )
        print(f"{bq.query_id}: latency_ms={latency_ms:.2f}")

    log_to_mlflow("config_C_cohere_rerank", rows)
    written = write_to_postgres(rows, run_id, session_factory)
    print(f"config_C_cohere_rerank (corrected latency): {aggregate(rows)} ({written} rows written)")
    print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
