"""Layer 8 quality regression monitor: daily RAGAS sample over
evaluation/relevance_labeller.py's BENCHMARK_QUERIES, alerting when a
tracked metric has declined for `ALERT_STREAK` consecutive days.

Mirrors monitoring/drift_detector.py's shape: a pure, deterministic core
(`consecutive_decline_days` / `detect_regression` / `evaluate_regression`)
fully unit-tested with no I/O, model, or LLM call, plus an Airflow-task-shaped
`run()` that wires a real day's RAGAS sample via injected callables.
`ragas_runner_fn` is the "injectable RAGAS runner" this file's spec calls
for — a plain `list[BenchmarkQuery] -> dict[str, float]` callable, so unit
tests supply a fake and never invoke evaluation/ragas_suite.py or a live
Claude/embedding model, per AGENTS.md's mock-everything-in-tests rule.
`build_default_ragas_runner()` wires the real thing (agents/qa_agent.py +
evaluation/ragas_suite.py) lazily inside its closure, mirroring
ragas_suite.py's own build_default_llm/build_default_embeddings split, so
importing this module never requires langchain_anthropic or
sentence-transformers.

A "3-day degradation" alert is defined as: a tracked metric's daily score
has strictly declined for `ALERT_STREAK` (default 3) consecutive days,
scanning back from today. This deliberately mirrors a day-over-day trend,
not a fixed floor -- a metric that's merely low but flat doesn't alert here
(that's a threshold-gate concern, e.g. the CI NDCG gate), only one that's
actively getting worse.
"""

from __future__ import annotations

import datetime
from typing import Callable

import structlog
from pydantic import BaseModel, Field

from evaluation.relevance_labeller import BENCHMARK_QUERIES, BenchmarkQuery

log = structlog.get_logger()

DAILY_SAMPLE_SIZE = 20
DEFAULT_METRICS: tuple[str, ...] = ("faithfulness", "answer_relevancy")
ALERT_STREAK = 3  # consecutive daily declines that trigger an alert


class DailyScore(BaseModel):
    date: str  # ISO date, e.g. "2026-08-07"
    scores: dict[str, float]
    num_queries: int = DAILY_SAMPLE_SIZE


class MetricRegression(BaseModel):
    metric: str
    streak: int
    is_regressing: bool
    history: list[float] = Field(default_factory=list)  # oldest -> newest


class QualityRegressionResult(BaseModel):
    today: DailyScore
    regressions: list[MetricRegression]
    is_regressing: bool
    alert_message: str | None = None


# ---------------------------------------------------------------------------
# Pure, deterministic core
# ---------------------------------------------------------------------------


def consecutive_decline_days(scores: list[float], eps: float = 0.0) -> int:
    """Count consecutive strict declines scanning back from the most recent
    score: scores[-1] < scores[-2] < ... Stops at the first non-decline (or
    the start of the list). Fewer than 2 scores has nothing to compare, so
    the streak is 0 by construction. `eps` is a minimum drop to count as a
    "real" decline rather than float noise (default 0.0: any drop counts).
    """
    streak = 0
    for i in range(len(scores) - 1, 0, -1):
        if scores[i] < scores[i - 1] - eps:
            streak += 1
        else:
            break
    return streak


def detect_regression(
    history: list[DailyScore], metric: str, alert_streak: int = ALERT_STREAK, eps: float = 0.0
) -> MetricRegression:
    series = [day.scores[metric] for day in history if metric in day.scores]
    streak = consecutive_decline_days(series, eps=eps)
    return MetricRegression(metric=metric, streak=streak, is_regressing=streak >= alert_streak, history=series)


def _build_alert_message(today: DailyScore, degraded: list[MetricRegression]) -> str:
    lines = [f"{r.metric}: {r.streak} consecutive days declining (latest={r.history[-1]:.4f})" for r in degraded]
    return f"Quality regression detected on {today.date}: " + "; ".join(lines)


def evaluate_regression(
    history: list[DailyScore],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    alert_streak: int = ALERT_STREAK,
) -> QualityRegressionResult:
    if not history:
        raise ValueError("history must be non-empty")

    regressions = [detect_regression(history, m, alert_streak=alert_streak) for m in metrics]
    degraded = [r for r in regressions if r.is_regressing]
    is_regressing = bool(degraded)
    alert_message = _build_alert_message(history[-1], degraded) if is_regressing else None

    return QualityRegressionResult(
        today=history[-1], regressions=regressions, is_regressing=is_regressing, alert_message=alert_message
    )


# ---------------------------------------------------------------------------
# Airflow-task-shaped entry point -- injected fetch/sample/score/persist/alert
# ---------------------------------------------------------------------------

RagasRunnerFn = Callable[[list[BenchmarkQuery]], dict[str, float]]


def run(
    fetch_history_fn: Callable[[], list[DailyScore]],
    sample_queries_fn: Callable[[int], list[BenchmarkQuery]],
    ragas_runner_fn: RagasRunnerFn,
    persist_fn: Callable[[DailyScore], None] | None = None,
    alert_fn: Callable[[str], None] | None = None,
    sample_size: int = DAILY_SAMPLE_SIZE,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    alert_streak: int = ALERT_STREAK,
    today: str | None = None,
) -> QualityRegressionResult:
    """Fetch prior daily scores, sample today's `sample_size` queries, score
    them with the injected RAGAS runner, persist the new DailyScore, and
    alert if the trailing decline streak reaches `alert_streak`. `today` is
    injectable so callers/tests get a deterministic date without patching
    `datetime`."""
    today = today or datetime.date.today().isoformat()
    queries = sample_queries_fn(sample_size)
    log.info("quality_regression_run", stage="quality_regression", date=today, num_queries=len(queries))

    scores = ragas_runner_fn(queries)
    today_score = DailyScore(date=today, scores=scores, num_queries=len(queries))

    history = [*fetch_history_fn(), today_score]
    result = evaluate_regression(history, metrics=metrics, alert_streak=alert_streak)

    if persist_fn is not None:
        persist_fn(today_score)

    if result.is_regressing:
        log.warning(
            "quality_regression_detected", stage="quality_regression", date=today, alert=result.alert_message
        )
        if alert_fn is not None:
            alert_fn(result.alert_message or "quality regression detected")
    else:
        log.info("quality_regression_ok", stage="quality_regression", date=today)

    return result


# ---------------------------------------------------------------------------
# Real wiring (lazy) -- not exercised by unit tests
# ---------------------------------------------------------------------------


def default_sample_queries_fn(n: int) -> list[BenchmarkQuery]:
    """First `n` of the fixed 50-query benchmark set -- same source Day 11's
    10-query RAGAS sample drew from, per evaluation/ragas_suite.py's `main()`."""
    return BENCHMARK_QUERIES[:n]


def build_default_ragas_runner() -> RagasRunnerFn:
    """Wires the real pipeline: agents/qa_agent.py's `run()` over each
    sampled query, then evaluation/ragas_suite.py's `run_ragas_eval` over the
    resulting QAStates. Imported lazily (mirrors evaluation/ragas_suite.py's
    build_default_llm/build_default_embeddings split) so importing this
    module never requires langchain_anthropic, sentence-transformers, or a
    live Claude/BM25/Qdrant connection -- only invoked when the Airflow task
    actually runs."""

    def _run(queries: list[BenchmarkQuery]) -> dict[str, float]:
        from agents import qa_agent
        from evaluation import ragas_suite

        qa_states = [qa_agent.run(bq.query, query_id=bq.query_id) for bq in queries]
        llm = ragas_suite.build_default_llm()
        embeddings = ragas_suite.build_default_embeddings()
        return ragas_suite.run_ragas_eval(qa_states, llm=llm, embeddings=embeddings)

    return _run
