"""Layer 8 HITL feedback aggregator: weekly summary of slackbot/
feedback_handler.py's 👍/👎 reactions (schema/models.py's FeedbackLog),
alerting when the thumbs-down rate over the window is high enough to be
worth a human look.

Mirrors monitoring/quality_regression.py's shape: a pure, deterministic core
(`aggregate_feedback`) fully unit-tested with no I/O, plus an
Airflow-task-shaped `run()` that wires a real week's feedback rows via
injected callables. airflow/dags/feedback_aggregator.py is the thin DAG that
wires a real Postgres read (FeedbackLog) and local-JSON-history persistence
to this module's `run()`, exactly like airflow/dags/drift_monitor.py wires
this file's siblings.

"Worth a look" is defined as: at least MIN_FEEDBACK_FOR_ALERT reactions in
the window (below that, a bad ratio is just small-sample noise) AND a
thumbs-down rate >= NEGATIVE_RATE_ALERT_THRESHOLD -- deliberately a volume
+ rate gate together, not a rate alone, so one grumpy reactor on a quiet
week doesn't page anyone.
"""

from __future__ import annotations

import datetime
from collections import Counter
from typing import Callable

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger()

NEGATIVE_RATE_ALERT_THRESHOLD = 0.30  # >=30% thumbs-down in the window is worth a look
MIN_FEEDBACK_FOR_ALERT = 5  # below this, a bad ratio is just small-sample noise
TOP_N_WORST_QUERIES = 5


class FeedbackRow(BaseModel):
    feedback_id: str
    query_id: str | None = None
    user_id: str
    reaction: str  # "up" | "down"


class QueryDownCount(BaseModel):
    query_id: str
    down_count: int


class FeedbackSummary(BaseModel):
    week_ending: str  # ISO date
    total: int
    up: int
    down: int
    down_rate: float
    worst_queries: list[QueryDownCount] = Field(default_factory=list)
    should_alert: bool


# ---------------------------------------------------------------------------
# Pure, deterministic core
# ---------------------------------------------------------------------------


def aggregate_feedback(
    rows: list[FeedbackRow],
    week_ending: str,
    alert_threshold: float = NEGATIVE_RATE_ALERT_THRESHOLD,
    min_for_alert: int = MIN_FEEDBACK_FOR_ALERT,
    top_n_worst: int = TOP_N_WORST_QUERIES,
) -> FeedbackSummary:
    total = len(rows)
    up = sum(1 for r in rows if r.reaction == "up")
    down = sum(1 for r in rows if r.reaction == "down")
    down_rate = round(down / total, 4) if total else 0.0

    down_counts = Counter(r.query_id for r in rows if r.reaction == "down" and r.query_id)
    worst_queries = [
        QueryDownCount(query_id=qid, down_count=count) for qid, count in down_counts.most_common(top_n_worst)
    ]

    return FeedbackSummary(
        week_ending=week_ending,
        total=total,
        up=up,
        down=down,
        down_rate=down_rate,
        worst_queries=worst_queries,
        should_alert=total >= min_for_alert and down_rate >= alert_threshold,
    )


# ---------------------------------------------------------------------------
# Airflow-task-shaped entry point -- injected fetch/persist/alert
# ---------------------------------------------------------------------------


def run(
    fetch_rows_fn: Callable[[], list[FeedbackRow]],
    persist_fn: Callable[[FeedbackSummary], None] | None = None,
    alert_fn: Callable[[FeedbackSummary], None] | None = None,
    week_ending: str | None = None,
) -> FeedbackSummary:
    """Fetch this window's feedback rows, aggregate, persist, and alert if
    the down-rate gate trips. `week_ending` is injectable so callers/tests
    get a deterministic date without patching `datetime`."""
    week_ending = week_ending or datetime.date.today().isoformat()
    rows = fetch_rows_fn()
    log.info("feedback_aggregator_run", stage="feedback_aggregator", week_ending=week_ending, num_rows=len(rows))

    summary = aggregate_feedback(rows, week_ending=week_ending)

    if persist_fn is not None:
        persist_fn(summary)

    if summary.should_alert:
        log.warning(
            "feedback_aggregator_alert",
            stage="feedback_aggregator",
            week_ending=week_ending,
            down_rate=summary.down_rate,
            total=summary.total,
            worst_queries=[q.model_dump() for q in summary.worst_queries],
        )
        if alert_fn is not None:
            alert_fn(summary)
    else:
        log.info(
            "feedback_aggregator_ok",
            stage="feedback_aggregator",
            week_ending=week_ending,
            down_rate=summary.down_rate,
            total=summary.total,
        )

    return summary
