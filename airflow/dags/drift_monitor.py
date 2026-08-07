"""Airflow 3 TaskFlow DAG — nvidia-ir-rag-agent Layer 8 drift monitoring.

Daily pipeline:
  fetch_query_windows → check_query_drift ─┐
                       → check_term_shift  ─┼→ aggregate_and_alert
                    check_quality_regression ┘

Wires monitoring/drift_detector.py, monitoring/term_shift_monitor.py, and
monitoring/quality_regression.py's pure cores (Day 11/12, unit-tested) to
real Postgres reads (query_log for query-drift, evaluation/relevance_labeller.py's
fixed benchmark set for quality regression) and a real embedding call
(retrieval/dense_index.py's encoder), mirroring
airflow/dags/ingest_nvidia_docs.py's project-root-on-sys.path convention,
per-task SessionFactory construction, and structlog usage.

Day 12's remaining-work spec asks for ONE DAG wiring all three monitors
(not the three separate DAG files AGENTS.md's folder-structure table
sketches under airflow/dags/: drift_monitor, quality_regression,
feedback_aggregator) -- kept as a single `drift_monitor` DAG per that
explicit instruction; standalone `quality_regression`/`feedback_aggregator`
DAG files remain open items.

Query-drift baseline: query_log rows from
[now - BASELINE_WINDOW_START_DAYS, now - BASELINE_WINDOW_END_DAYS) -- a
fixed 7-day window ending 30 days back, recomputed fresh each run rather
than frozen/persisted, since no baseline-embedding table exists yet (open
item: persist a frozen baseline once traffic volume justifies it, instead
of a rolling 30-37-day-ago window). Current window: last
CURRENT_WINDOW_DAYS of query_log. The same two windows feed both PSI drift
(on embeddings) and term shift (on raw query text).

Quality-regression history has no Postgres table yet either -- persisted to
a local JSON log (QUALITY_HISTORY_PATH) via persist_fn, append-only,
mirroring how data/ (gitignored) already holds other build/data artifacts
in this project (BM25's index.pkl, etc). alert_fn is not wired to Slack
yet -- monitoring/quality_regression.py's run() already structlog.warning()s
on its own, so a missing alert_fn only means "no Slack DM", not "silent
regression" (Slackbot HITL feedback loop is a known open item, see
docs/daily_progress/day_11_storyline.md §5).

Not live-verified against a running Airflow scheduler or real Postgres
data this session -- see docs/daily_progress/day_12_storyline.md. Same
"unit-tested, not yet DAG-registered-and-run" status Day 11 left
drift_detector.py and term_shift_monitor.py in.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pendulum

# ── project root on sys.path so monitoring/evaluation/schema imports resolve ──
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import structlog

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from evaluation.relevance_labeller import BENCHMARK_QUERIES
from monitoring import drift_detector, quality_regression, term_shift_monitor
from monitoring.quality_regression import DailyScore
from schema.models import QueryLog, get_engine, get_session_factory

log = structlog.get_logger()

BASELINE_WINDOW_START_DAYS = 37
BASELINE_WINDOW_END_DAYS = 30
CURRENT_WINDOW_DAYS = 1
MIN_QUERIES_FOR_DRIFT = 5  # below this, skip PSI/term-shift for the day rather than compute on noise

QUALITY_HISTORY_PATH = Path("data/monitoring/quality_regression_history.json")


def _query_texts_between(session_factory: Any, start: datetime, end: datetime) -> list[str]:
    with session_factory() as session:
        rows = (
            session.query(QueryLog.query_text)
            .filter(QueryLog.created_at >= start, QueryLog.created_at < end)
            .all()
        )
        return [r[0] for r in rows]


def _load_quality_history() -> list[DailyScore]:
    if not QUALITY_HISTORY_PATH.exists():
        return []
    raw = json.loads(QUALITY_HISTORY_PATH.read_text(encoding="utf-8"))
    return [DailyScore(**d) for d in raw]


def _persist_quality_score(today_score: DailyScore) -> None:
    history = _load_quality_history()
    history.append(today_score)
    QUALITY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_HISTORY_PATH.write_text(
        json.dumps([d.model_dump() for d in history], indent=2), encoding="utf-8"
    )


@dag(
    dag_id="drift_monitor",
    schedule="@daily",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=1),
    },
    tags=["nvidia-ir", "monitoring", "layer-8"],
)
def drift_monitor() -> None:

    @task()
    def fetch_query_windows() -> dict[str, list[str]]:
        ctx = get_current_context()
        run_id: str = ctx["run_id"]
        now = datetime.now(timezone.utc)

        engine = get_engine()
        SessionFactory = get_session_factory(engine)

        baseline_start = now - timedelta(days=BASELINE_WINDOW_START_DAYS)
        baseline_end = now - timedelta(days=BASELINE_WINDOW_END_DAYS)
        current_start = now - timedelta(days=CURRENT_WINDOW_DAYS)

        baseline_queries = _query_texts_between(SessionFactory, baseline_start, baseline_end)
        current_queries = _query_texts_between(SessionFactory, current_start, now)

        log.info(
            "fetch_query_windows",
            stage="fetch_query_windows",
            query_id=run_id,
            num_baseline=len(baseline_queries),
            num_current=len(current_queries),
        )
        return {"baseline_queries": baseline_queries, "current_queries": current_queries}

    @task()
    def check_query_drift(windows: dict[str, list[str]]) -> dict[str, Any]:
        ctx = get_current_context()
        run_id: str = ctx["run_id"]
        baseline_queries = windows["baseline_queries"]
        current_queries = windows["current_queries"]

        if len(baseline_queries) < MIN_QUERIES_FOR_DRIFT or len(current_queries) < MIN_QUERIES_FOR_DRIFT:
            log.warning(
                "check_query_drift_skipped_sparse_data",
                stage="check_query_drift",
                query_id=run_id,
                num_baseline=len(baseline_queries),
                num_current=len(current_queries),
            )
            return {"skipped": True, "reason": "insufficient_query_volume"}

        from retrieval.dense_index import build_default_encoder

        encoder = build_default_encoder()
        baseline_embeddings = np.array([encoder(q) for q in baseline_queries])

        def embed_fn(texts: list[str]) -> np.ndarray:
            return np.array([encoder(t) for t in texts])

        result = drift_detector.run(
            baseline_embeddings=baseline_embeddings,
            fetch_queries_fn=lambda: current_queries,
            embed_fn=embed_fn,
        )
        return {"skipped": False, **result.model_dump()}

    @task()
    def check_term_shift(windows: dict[str, list[str]]) -> dict[str, Any]:
        result = term_shift_monitor.run(
            fetch_baseline_texts_fn=lambda: windows["baseline_queries"],
            fetch_current_texts_fn=lambda: windows["current_queries"],
        )
        return result.model_dump()

    @task()
    def check_quality_regression() -> dict[str, Any]:
        result = quality_regression.run(
            fetch_history_fn=_load_quality_history,
            sample_queries_fn=lambda n: BENCHMARK_QUERIES[:n],
            ragas_runner_fn=quality_regression.build_default_ragas_runner(),
            persist_fn=_persist_quality_score,
            alert_fn=None,  # Slackbot HITL wiring is an open item; run() already structlog.warning()s
        )
        return result.model_dump()

    @task()
    def aggregate_and_alert(
        drift_result: dict[str, Any],
        term_shift_result: dict[str, Any],
        regression_result: dict[str, Any],
    ) -> None:
        ctx = get_current_context()
        run_id: str = ctx["run_id"]

        drifted = not drift_result.get("skipped") and drift_result.get("is_drifted")
        num_shifted_terms = len(term_shift_result.get("shifted_terms", []))
        regressing = regression_result.get("is_regressing", False)

        log.info(
            "drift_monitor_summary",
            stage="aggregate_and_alert",
            query_id=run_id,
            query_drift_severity=drift_result.get("severity", "skipped"),
            num_shifted_terms=num_shifted_terms,
            quality_regressing=regressing,
        )

        if drifted or num_shifted_terms > 0 or regressing:
            log.warning(
                "drift_monitor_alert",
                stage="aggregate_and_alert",
                query_id=run_id,
                query_drift_severity=drift_result.get("severity"),
                num_shifted_terms=num_shifted_terms,
                quality_regression_message=regression_result.get("alert_message"),
            )

    # ── Wire the DAG ────────────────────────────────────────────────────
    windows = fetch_query_windows()
    drift_result = check_query_drift(windows)
    term_shift_result = check_term_shift(windows)
    regression_result = check_quality_regression()
    aggregate_and_alert(drift_result, term_shift_result, regression_result)


drift_monitor()
