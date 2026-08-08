"""Airflow 3 TaskFlow DAG — nvidia-ir-rag-agent Layer 8 feedback aggregation.

Weekly pipeline:
  fetch_feedback_rows → aggregate_and_persist

Wires monitoring/feedback_aggregator.py's pure, unit-tested core (`run` /
`aggregate_feedback`) to a real Postgres read (schema/models.py's
FeedbackLog, populated by slackbot/feedback_handler.py's 👍/👎 HITL
reactions) and local-JSON-history persistence -- mirrors
airflow/dags/drift_monitor.py's project-root-on-sys.path convention,
per-task SessionFactory construction, structlog usage, and
no-dedicated-Postgres-aggregate-table-yet local JSON persistence (same
reasoning as drift_monitor.py's quality-regression history).

alert_fn is not wired to Slack yet, even though slackbot/app.py now exists
(Day 13) -- monitoring/feedback_aggregator.py's run() already
structlog.warning()s on its own when the down-rate gate trips, so a missing
alert_fn only means "no Slack DM", not "silent regression". Posting that
alert back into Slack (via slack_sdk.WebClient.chat_postMessage) is a
natural next step, left as an open item alongside drift_monitor.py's own
unwired alert_fn.

Not live-verified against a running Airflow scheduler this session -- same
status drift_monitor.py documents for itself (see docs/daily_progress/
day_12_storyline.md); this project's local dev venv deliberately doesn't
install apache-airflow (DAGs run inside docker-compose.yml's `airflow`
service image, apache/airflow:3.0.2, not requirements.txt), so
monitoring/feedback_aggregator.py carries the real test coverage
(tests/monitoring/test_feedback_aggregator.py) and this file is checked by
matching airflow/dags/drift_monitor.py's already-established shape closely,
not by a local DAG-parse run.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── project root on sys.path so monitoring/schema imports resolve ──────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pendulum
import structlog

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from monitoring.feedback_aggregator import FeedbackRow, FeedbackSummary, run
from schema.models import FeedbackLog, get_engine, get_session_factory

log = structlog.get_logger()

WINDOW_DAYS = 7
FEEDBACK_HISTORY_PATH = Path("data/monitoring/feedback_aggregation_history.json")


def _fetch_feedback_rows(window_start: datetime) -> list[FeedbackRow]:
    engine = get_engine()
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        rows = session.query(FeedbackLog).filter(FeedbackLog.created_at >= window_start).all()
        return [
            FeedbackRow(feedback_id=r.feedback_id, query_id=r.query_id, user_id=r.user_id, reaction=r.reaction)
            for r in rows
        ]


def _load_history() -> list[dict[str, Any]]:
    if not FEEDBACK_HISTORY_PATH.exists():
        return []
    result: list[dict[str, Any]] = json.loads(FEEDBACK_HISTORY_PATH.read_text(encoding="utf-8"))
    return result


def _persist_summary(summary: FeedbackSummary) -> None:
    history = _load_history()
    history.append(summary.model_dump())
    FEEDBACK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


@dag(
    dag_id="feedback_aggregator",
    schedule="@weekly",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=30),
    },
    tags=["nvidia-ir", "monitoring", "layer-8", "hitl"],
)
def feedback_aggregator() -> None:

    @task()
    def fetch_feedback_rows() -> list[dict[str, Any]]:
        ctx = get_current_context()
        run_id: str = ctx["run_id"]
        window_start = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)

        rows = _fetch_feedback_rows(window_start)
        log.info(
            "fetch_feedback_rows",
            stage="fetch_feedback_rows",
            query_id=run_id,
            num_rows=len(rows),
            window_days=WINDOW_DAYS,
        )
        # XCom needs plain JSON-serializable data, not pydantic models.
        return [r.model_dump() for r in rows]

    @task()
    def aggregate_and_persist(row_dicts: list[dict[str, Any]]) -> dict[str, Any]:
        ctx = get_current_context()
        run_id: str = ctx["run_id"]
        rows = [FeedbackRow(**d) for d in row_dicts]

        summary = run(
            fetch_rows_fn=lambda: rows,
            persist_fn=_persist_summary,
            alert_fn=None,  # Slack alert wiring is an open item; run() already structlog.warning()s
            week_ending=datetime.now(timezone.utc).date().isoformat(),
        )
        log.info(
            "feedback_aggregator_task_done",
            stage="aggregate_and_persist",
            query_id=run_id,
            total=summary.total,
            down_rate=summary.down_rate,
            should_alert=summary.should_alert,
        )
        return summary.model_dump()

    # ── Wire the DAG ────────────────────────────────────────────────────
    rows = fetch_feedback_rows()
    aggregate_and_persist(rows)


feedback_aggregator()
