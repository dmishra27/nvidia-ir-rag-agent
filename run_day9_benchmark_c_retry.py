"""Day 9 Task 3 retry: Config C (Cohere) alone, throttled.

run_day9_benchmark_ac.py's Config C run hit Cohere's trial-key rate limit
(429 TooManyRequestsError: "Trial key... limited to 10 API calls / minute")
after 10/15 queries — the loop fired all 15 rerank() calls in under a
second. This retry wraps CohereReranker.rerank in a minimum-interval
throttle (6.5s, safely under the 10/min = 6.0s/call limit) and reuses the
relevance labels run_day9_benchmark_ac.py already wrote (Config A's run
already succeeded and is logged; no need to re-spend Claude calls
re-labelling the same 45 pairs).
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

from evaluation.benchmark_runner import (
    COHERE_COST_PER_QUERY_USD,
    aggregate,
    log_to_mlflow,
    run_config,
    write_to_postgres,
)
from evaluation.relevance_labeller import RelevanceLabel
from retrieval.reranker_cohere import CohereReranker
from run_day9_benchmark_ac import POOL_LABELS_PATH, load_cached_pools
from schema.models import get_engine, get_session_factory

COHERE_TRIAL_MIN_INTERVAL_S = 6.5


def load_pool_relevance(path: Path = POOL_LABELS_PATH) -> dict[str, dict[str, int]]:
    relevance: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = RelevanceLabel.model_validate_json(line)
        relevance.setdefault(label.query_id, {})[label.chunk_id] = label.label
    return relevance


def throttled(rerank_fn, min_interval_s: float):
    last_call = [0.0]

    def wrapped(query, candidates, top_k, query_id):
        elapsed = time.monotonic() - last_call[0]
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        last_call[0] = time.monotonic()
        return rerank_fn(query, candidates, top_k, query_id)

    return wrapped


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run_id = str(uuid.uuid4())[:8]
    session_factory = get_session_factory(get_engine())

    queries, pools = load_cached_pools()
    relevance = load_pool_relevance()
    print(f"Loaded {len(queries)} cached queries and {sum(len(v) for v in relevance.values())} relevance judgments.")

    cohere_reranker = CohereReranker.load()
    rerank_fn = throttled(cohere_reranker.rerank, COHERE_TRIAL_MIN_INTERVAL_S)

    rows_c = run_config(
        "config_C_cohere_rerank", rerank_fn, queries, pools, relevance, top_k=3, cost_per_query=COHERE_COST_PER_QUERY_USD
    )
    log_to_mlflow("config_C_cohere_rerank", rows_c)
    written = write_to_postgres(rows_c, run_id, session_factory)
    print(f"config_C_cohere_rerank: {aggregate(rows_c)} ({written} rows written to benchmark_results)")
    print(f"run_id: {run_id}")


if __name__ == "__main__":
    main()
