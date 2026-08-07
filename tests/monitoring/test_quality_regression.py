"""Unit tests for monitoring/quality_regression.py.

Per AGENTS.md ("Deterministic functions: strict TDD") and this file's
"injectable RAGAS runner" requirement, `consecutive_decline_days`,
`detect_regression`, and `evaluate_regression` are pure functions tested
directly with fabricated DailyScore histories -- no RAGAS, no LLM, no I/O.
`run()` is tested with injected fake fetch/sample/ragas-runner/persist/alert
callables, mirroring tests/monitoring/test_drift_detector.py's
constructor-injection convention.
"""

from __future__ import annotations

import pytest

from evaluation.relevance_labeller import BenchmarkQuery
from monitoring.quality_regression import (
    DailyScore,
    MetricRegression,
    QualityRegressionResult,
    consecutive_decline_days,
    detect_regression,
    evaluate_regression,
    run,
)


def _day(date: str, **scores: float) -> DailyScore:
    return DailyScore(date=date, scores=scores, num_queries=20)


# ---------------------------------------------------------------------------
# consecutive_decline_days() -- pure, deterministic
# ---------------------------------------------------------------------------


def test_no_decline_for_empty_list() -> None:
    assert consecutive_decline_days([]) == 0


def test_no_decline_for_single_score() -> None:
    assert consecutive_decline_days([0.8]) == 0


def test_no_decline_for_flat_series() -> None:
    assert consecutive_decline_days([0.8, 0.8, 0.8]) == 0


def test_no_decline_for_rising_series() -> None:
    assert consecutive_decline_days([0.5, 0.6, 0.7]) == 0


def test_counts_two_day_decline() -> None:
    assert consecutive_decline_days([0.9, 0.8, 0.7]) == 2


def test_counts_three_day_decline() -> None:
    assert consecutive_decline_days([0.95, 0.9, 0.8, 0.7]) == 3


def test_streak_stops_at_first_non_decline_scanning_back() -> None:
    # Rises then declines twice: only the trailing decline streak counts.
    assert consecutive_decline_days([0.5, 0.9, 0.8, 0.7]) == 2


def test_eps_requires_a_minimum_drop() -> None:
    # Drops of exactly 0.01 shouldn't count as a "real" decline at eps=0.02.
    assert consecutive_decline_days([0.80, 0.79, 0.78], eps=0.02) == 0
    assert consecutive_decline_days([0.80, 0.75, 0.70], eps=0.02) == 2


def test_is_deterministic_given_same_inputs() -> None:
    series = [0.9, 0.8, 0.7, 0.6]
    assert consecutive_decline_days(series) == consecutive_decline_days(series)


# ---------------------------------------------------------------------------
# detect_regression()
# ---------------------------------------------------------------------------


def test_detect_regression_not_regressing_below_streak() -> None:
    history = [_day("d1", faithfulness=0.9), _day("d2", faithfulness=0.85)]

    result = detect_regression(history, "faithfulness")

    assert isinstance(result, MetricRegression)
    assert result.streak == 1
    assert result.is_regressing is False


def test_detect_regression_flags_at_alert_streak() -> None:
    history = [
        _day("d1", faithfulness=0.95),
        _day("d2", faithfulness=0.90),
        _day("d3", faithfulness=0.85),
        _day("d4", faithfulness=0.80),
    ]

    result = detect_regression(history, "faithfulness")

    assert result.streak == 3
    assert result.is_regressing is True
    assert result.history == [0.95, 0.90, 0.85, 0.80]


def test_detect_regression_custom_alert_streak() -> None:
    history = [_day("d1", faithfulness=0.9), _day("d2", faithfulness=0.8)]

    result = detect_regression(history, "faithfulness", alert_streak=1)

    assert result.is_regressing is True


def test_detect_regression_skips_days_missing_the_metric() -> None:
    history = [
        _day("d1", faithfulness=0.9, answer_relevancy=0.5),
        _day("d2", answer_relevancy=0.4),  # no faithfulness recorded
        _day("d3", faithfulness=0.8),
    ]

    result = detect_regression(history, "faithfulness")

    assert result.history == [0.9, 0.8]


# ---------------------------------------------------------------------------
# evaluate_regression()
# ---------------------------------------------------------------------------


def test_evaluate_regression_raises_on_empty_history() -> None:
    with pytest.raises(ValueError):
        evaluate_regression([])


def test_evaluate_regression_no_alert_when_stable() -> None:
    history = [_day("d1", faithfulness=0.8, answer_relevancy=0.5), _day("d2", faithfulness=0.82, answer_relevancy=0.5)]

    result = evaluate_regression(history)

    assert isinstance(result, QualityRegressionResult)
    assert result.is_regressing is False
    assert result.alert_message is None
    assert {r.metric for r in result.regressions} == {"faithfulness", "answer_relevancy"}


def test_evaluate_regression_alerts_when_one_metric_degrades() -> None:
    history = [
        _day("d1", faithfulness=0.95, answer_relevancy=0.5),
        _day("d2", faithfulness=0.90, answer_relevancy=0.5),
        _day("d3", faithfulness=0.85, answer_relevancy=0.5),
        _day("d4", faithfulness=0.80, answer_relevancy=0.5),
    ]

    result = evaluate_regression(history)

    assert result.is_regressing is True
    assert result.alert_message is not None
    assert "faithfulness" in result.alert_message
    assert "answer_relevancy" not in result.alert_message


def test_evaluate_regression_today_is_last_history_entry() -> None:
    history = [_day("d1", faithfulness=0.9), _day("d2", faithfulness=0.8)]

    result = evaluate_regression(history)

    assert result.today.date == "d2"


# ---------------------------------------------------------------------------
# run() -- Airflow-task-shaped entry point, injected fakes
# ---------------------------------------------------------------------------


def _fake_queries(n: int) -> list[BenchmarkQuery]:
    return [BenchmarkQuery(query_id=f"q{i}", query=f"query {i}") for i in range(n)]


def test_run_persists_todays_score() -> None:
    persisted: list[DailyScore] = []

    result = run(
        fetch_history_fn=lambda: [],
        sample_queries_fn=_fake_queries,
        ragas_runner_fn=lambda qs: {"faithfulness": 0.8, "answer_relevancy": 0.5},
        persist_fn=persisted.append,
        sample_size=5,
        today="2026-08-07",
    )

    assert len(persisted) == 1
    assert persisted[0].date == "2026-08-07"
    assert persisted[0].num_queries == 5
    assert result.today.scores == {"faithfulness": 0.8, "answer_relevancy": 0.5}


def test_run_samples_the_requested_query_count() -> None:
    calls: list[int] = []

    def sample_fn(n: int) -> list[BenchmarkQuery]:
        calls.append(n)
        return _fake_queries(n)

    run(
        fetch_history_fn=lambda: [],
        sample_queries_fn=sample_fn,
        ragas_runner_fn=lambda qs: {"faithfulness": 0.8},
        sample_size=20,
        today="2026-08-07",
    )

    assert calls == [20]


def test_run_no_alert_when_history_is_stable() -> None:
    alerts: list[str] = []
    history = [_day("2026-08-05", faithfulness=0.8), _day("2026-08-06", faithfulness=0.8)]

    result = run(
        fetch_history_fn=lambda: history,
        sample_queries_fn=_fake_queries,
        ragas_runner_fn=lambda qs: {"faithfulness": 0.81},
        alert_fn=alerts.append,
        today="2026-08-07",
    )

    assert result.is_regressing is False
    assert alerts == []


def test_run_fires_alert_fn_on_three_day_degradation() -> None:
    alerts: list[str] = []
    history = [
        _day("2026-08-04", faithfulness=0.95),
        _day("2026-08-05", faithfulness=0.90),
        _day("2026-08-06", faithfulness=0.85),
    ]

    result = run(
        fetch_history_fn=lambda: history,
        sample_queries_fn=_fake_queries,
        ragas_runner_fn=lambda qs: {"faithfulness": 0.80},
        alert_fn=alerts.append,
        today="2026-08-07",
    )

    assert result.is_regressing is True
    assert len(alerts) == 1
    assert "faithfulness" in alerts[0]


def test_run_defaults_today_to_current_date_when_not_supplied() -> None:
    import datetime

    result = run(
        fetch_history_fn=lambda: [],
        sample_queries_fn=_fake_queries,
        ragas_runner_fn=lambda qs: {"faithfulness": 0.8},
    )

    assert result.today.date == datetime.date.today().isoformat()


def test_run_works_without_persist_or_alert_fn() -> None:
    result = run(
        fetch_history_fn=lambda: [],
        sample_queries_fn=_fake_queries,
        ragas_runner_fn=lambda qs: {"faithfulness": 0.8},
        today="2026-08-07",
    )

    assert result.today.scores == {"faithfulness": 0.8}
