"""Live data fetchers for benchmark_tab.py and eval_dashboard.py -- MLflow
(reranker_benchmark/citation_judge/ragas_eval experiments) and Postgres
(benchmark_results) queries, plus evaluation/day9_citation_judgments.json,
replacing streamlit_app/mock_data.py's hardcoded Day 9/11 constants with the
same real numbers fetched live.

Every fetcher matches an existing, already-verified pattern elsewhere in
this project rather than inventing a new one: MLflow access mirrors
mcp/mcp_mlflow/server.py's MlflowClient usage (same MLFLOW_TRACKING_URI env
var, same experiment names); Postgres access mirrors schema/models.py's
SQLAlchemy ORM convention (AGENTS.md: "Never raw SQL strings").

Each fetcher raises on failure rather than swallowing errors -- callers
(the tab modules) decide the fallback/warning UX, matching this project's
constructor-injection convention (streamlit_app/benchmark_tab.py's
`SummariesFn` was already designed for exactly this swap, per its own
docstring) rather than baking a silent fallback in here.

mlflow's own HTTP client defaults to a 120s timeout and 7 retries with
exponential backoff (`MLFLOW_HTTP_REQUEST_TIMEOUT`/`_MAX_RETRIES`) -- fine
for a flaky-but-present server, but when MLflow is simply not running (a
fresh clone before `docker-compose up`, or CI, which never runs live
services at all per AGENTS.md) that turns "unreachable" into a multi-minute
hang instead of the fast, gracefully-handled failure `_load_summaries()`'s
try/except is designed around. Set short here -- only if the environment
hasn't already set them, so a real deployment can still tune for a slow
network -- before any MlflowClient is created.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.tracking import MlflowClient

from schema.models import BenchmarkResults, get_engine, get_session_factory
from streamlit_app.mock_data import BenchmarkConfigSummary

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
CITATION_JUDGMENTS_PATH = Path("evaluation/day9_citation_judgments.json")


def _mlflow_client() -> MlflowClient:
    # Only if unset, so a real deployment can still tune for a slow network.
    # 15s/1 retry: generous enough for a real-but-loaded server's round trip
    # (a tight 5s timeout was observed to false-positive against this
    # project's own shared dev MLflow instance under load), while still
    # capping the worst case at ~30s instead of mlflow's default ~15 minutes
    # (120s timeout x 7 retries) when nothing is listening at all.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def fetch_live_benchmark_summaries(experiment_name: str = "reranker_benchmark") -> list[BenchmarkConfigSummary]:
    """Most recent MLflow run per re-ranker config, in benchmark_tab.py's
    `summaries_fn` shape. Mirrors mcp/mcp_mlflow/server.py's
    get_benchmark_experiment, keeping only each config's newest run (Day 9
    logged config_C twice -- an initial run plus a fixed-latency retry)."""
    client = _mlflow_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["attribute.start_time DESC"], max_results=50
    )
    latest_by_config: dict[str, Any] = {}
    for run in runs:
        config = run.data.params.get("config", run.info.run_name)
        if config not in latest_by_config:
            latest_by_config[config] = run

    summaries = [
        BenchmarkConfigSummary(
            config=config,
            run_id=run.info.run_id[:8],
            ndcg_at_10=float(run.data.metrics.get("mean_ndcg_at_10", 0.0)),
            mrr=float(run.data.metrics.get("mean_mrr", 0.0)),
            prec_at_3=float(run.data.metrics.get("mean_prec_at_3", 0.0)),
            latency_ms=float(run.data.metrics.get("mean_latency_ms", 0.0)),
            cost_usd=float(run.data.metrics.get("total_cost_usd", 0.0)),
        )
        for config, run in latest_by_config.items()
    ]
    return sorted(summaries, key=lambda s: s.config)


def fetch_per_query_ndcg() -> pd.DataFrame:
    """Every benchmark_results row's (config, ndcg_at_10) pair -- a live
    Postgres read via the ORM, for a per-query distribution chart rather
    than just the aggregate mean MLflow already logged."""
    engine = get_engine()
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        rows = session.query(BenchmarkResults.config, BenchmarkResults.ndcg_at_10).all()
    if not rows:
        raise RuntimeError("benchmark_results table is empty")
    return pd.DataFrame(rows, columns=["config", "ndcg_at_10"])


def fetch_ragas_scores(experiment_name: str = "ragas_eval") -> tuple[dict[str, float], str, int]:
    """Most recent *complete* (non-null-metric) run's faithfulness/
    answer_relevancy from MLflow, plus its run_id and a fixed query count
    (Day 11's live run sampled DAY11_RAGAS_NUM_QUERIES queries; MLflow
    itself doesn't log query counts)."""
    client = _mlflow_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["attribute.start_time DESC"], max_results=20
    )
    for run in runs:
        faithfulness = run.data.metrics.get("faithfulness")
        answer_relevancy = run.data.metrics.get("answer_relevancy")
        if faithfulness is not None and answer_relevancy is not None:
            return (
                {"faithfulness": float(faithfulness), "answer_relevancy": float(answer_relevancy)},
                run.info.run_id[:8],
                10,
            )
    raise RuntimeError(f"no complete run found in MLflow experiment {experiment_name!r}")


def fetch_citation_accuracy(experiment_name: str = "citation_judge") -> float:
    """Most recent citation_judge run's citation_accuracy metric."""
    client = _mlflow_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["attribute.start_time DESC"], max_results=1
    )
    if not runs or "citation_accuracy" not in runs[0].data.metrics:
        raise RuntimeError(f"no citation_accuracy metric found in MLflow experiment {experiment_name!r}")
    return float(runs[0].data.metrics["citation_accuracy"])


def load_citation_judgments() -> pd.DataFrame:
    """Day 9's 27 real per-claim citation judgments (query_id, claim,
    chunk_id, supported, rationale) -- a committed JSON artifact, per
    agents/eda_agent.py's module docstring on why this file (not the empty
    eval_results Postgres table) is this project's real per-claim data."""
    if not CITATION_JUDGMENTS_PATH.exists():
        raise RuntimeError(f"{CITATION_JUDGMENTS_PATH} not found")
    records: list[dict[str, Any]] = json.loads(CITATION_JUDGMENTS_PATH.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


def citation_accuracy_by_query(judgments: pd.DataFrame) -> pd.DataFrame:
    """Per-query supported-claim fraction, for a per-query drill-down chart
    alongside the single aggregate citation_accuracy headline number."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # query_id -> [supported, total]
    for _, row in judgments.iterrows():
        bucket = counts[row["query_id"]]
        bucket[1] += 1
        if row["supported"]:
            bucket[0] += 1
    return pd.DataFrame(
        [
            {"query_id": qid, "supported": supported, "total": total, "accuracy": supported / total}
            for qid, (supported, total) in sorted(counts.items())
        ]
    )
