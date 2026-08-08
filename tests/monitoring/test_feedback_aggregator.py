"""Unit tests for monitoring/feedback_aggregator.py.

Per AGENTS.md ("Deterministic functions: strict TDD"), `aggregate_feedback`
is a pure function tested directly with fabricated FeedbackRow lists -- no
Postgres, no Airflow, no I/O. `run()` is tested with injected fake
fetch/persist/alert callables, mirroring
tests/monitoring/test_quality_regression.py's constructor-injection
convention.
"""

from __future__ import annotations

from monitoring.feedback_aggregator import (
    MIN_FEEDBACK_FOR_ALERT,
    FeedbackRow,
    FeedbackSummary,
    aggregate_feedback,
    run,
)


def _row(reaction: str, query_id: str | None = "q1", user_id: str = "U1") -> FeedbackRow:
    return FeedbackRow(feedback_id="f", query_id=query_id, user_id=user_id, reaction=reaction)


# ---------------------------------------------------------------------------
# aggregate_feedback() -- pure, deterministic
# ---------------------------------------------------------------------------


def test_empty_rows_has_zero_rate_and_no_alert() -> None:
    summary = aggregate_feedback([], week_ending="2026-08-08")
    assert summary.total == 0
    assert summary.down_rate == 0.0
    assert summary.should_alert is False


def test_counts_up_and_down_correctly() -> None:
    rows = [_row("up"), _row("up"), _row("down")]
    summary = aggregate_feedback(rows, week_ending="2026-08-08")
    assert summary.total == 3
    assert summary.up == 2
    assert summary.down == 1
    assert summary.down_rate == round(1 / 3, 4)


def test_should_alert_requires_both_volume_and_rate() -> None:
    # High down-rate, but below MIN_FEEDBACK_FOR_ALERT -- small-sample noise, no alert.
    few_rows = [_row("down"), _row("down")]
    assert len(few_rows) < MIN_FEEDBACK_FOR_ALERT
    summary = aggregate_feedback(few_rows, week_ending="2026-08-08")
    assert summary.down_rate == 1.0
    assert summary.should_alert is False


def test_should_alert_true_when_volume_and_rate_both_trip() -> None:
    rows = [_row("down")] * 4 + [_row("up")] * 1  # 5 total, 80% down
    summary = aggregate_feedback(rows, week_ending="2026-08-08")
    assert summary.total == 5
    assert summary.down_rate == 0.8
    assert summary.should_alert is True


def test_should_alert_false_below_threshold_even_with_enough_volume() -> None:
    rows = [_row("down")] + [_row("up")] * 9  # 10 total, 10% down
    summary = aggregate_feedback(rows, week_ending="2026-08-08")
    assert summary.should_alert is False


def test_worst_queries_ranks_by_down_count_descending() -> None:
    rows = (
        [_row("down", query_id="q_bad")] * 3
        + [_row("down", query_id="q_meh")] * 1
        + [_row("up", query_id="q_good")] * 5
    )
    summary = aggregate_feedback(rows, week_ending="2026-08-08")
    assert summary.worst_queries[0].query_id == "q_bad"
    assert summary.worst_queries[0].down_count == 3
    assert summary.worst_queries[1].query_id == "q_meh"
    assert all(q.query_id != "q_good" for q in summary.worst_queries)


def test_worst_queries_ignores_rows_with_no_query_id() -> None:
    rows = [_row("down", query_id=None)] * 3
    summary = aggregate_feedback(rows, week_ending="2026-08-08")
    assert summary.worst_queries == []


def test_worst_queries_capped_at_top_n() -> None:
    rows = [_row("down", query_id=f"q{i}") for i in range(10)]
    summary = aggregate_feedback(rows, week_ending="2026-08-08", top_n_worst=5)
    assert len(summary.worst_queries) == 5


# ---------------------------------------------------------------------------
# run() -- injected fetch/persist/alert
# ---------------------------------------------------------------------------


def test_run_calls_persist_fn_with_summary() -> None:
    persisted: list[FeedbackSummary] = []
    rows = [_row("up"), _row("down")]

    result = run(
        fetch_rows_fn=lambda: rows,
        persist_fn=persisted.append,
        week_ending="2026-08-08",
    )
    assert persisted == [result]
    assert result.total == 2


def test_run_calls_alert_fn_only_when_should_alert() -> None:
    alerts: list[FeedbackSummary] = []
    quiet_rows = [_row("up"), _row("down")]

    run(fetch_rows_fn=lambda: quiet_rows, alert_fn=alerts.append, week_ending="2026-08-08")
    assert alerts == []

    loud_rows = [_row("down")] * 5
    run(fetch_rows_fn=lambda: loud_rows, alert_fn=alerts.append, week_ending="2026-08-08")
    assert len(alerts) == 1
    assert alerts[0].should_alert is True


def test_run_defaults_week_ending_to_today_when_not_given() -> None:
    result = run(fetch_rows_fn=lambda: [])
    assert result.week_ending  # non-empty, format not asserted (deterministic date not injected here)
