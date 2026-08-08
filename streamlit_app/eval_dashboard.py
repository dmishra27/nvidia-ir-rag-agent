"""Eval dashboard tab: NDCG@10 · faithfulness · answer_relevancy · citation
accuracy, plus a per-query citation-accuracy breakdown and a daily
quality-regression trend.

Day 13: the four headline metrics and the NDCG-vs-gate chart are wired to
real live data -- faithfulness/answer_relevancy from MLflow's `ragas_eval`
experiment, citation_accuracy from MLflow's `citation_judge` experiment,
NDCG@10 from `reranker_benchmark` (same fetchers as benchmark_tab.py, via
streamlit_app/live_data.py) -- falling back to
streamlit_app/mock_data.py's committed Day 9/11 numbers (the *same* real
values) with a visible banner if MLflow isn't reachable, so the tab still
renders in CI and in a fresh clone before `docker-compose up`. The new
per-query citation-accuracy chart reads evaluation/day9_citation_judgments.json
directly (27 real per-claim judgments across 10 queries) rather than MLflow,
since MLflow only has the aggregate metric.

The trend chart is the one piece still synthetic: no daily quality-history
table exists yet (airflow/dags/drift_monitor.py's quality-regression task
has never actually run against a live scheduler -- see that DAG's own
docstring), so there is no real multi-day series to plot. It runs
monitoring/quality_regression.py's actual `evaluate_regression()` over a
synthetic 10-day history built around the real Day 11 baseline, same as
Day 12 shipped -- the regression *logic* is real, the daily history feeding
it isn't, and the caption says so.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from monitoring.quality_regression import DailyScore, QualityRegressionResult, evaluate_regression
from streamlit_app import live_data, theme
from streamlit_app.mock_data import (
    CI_NDCG_GATE,
    DAY9_BENCHMARK_SUMMARIES,
    DAY9_CITATION_ACCURACY,
    DAY11_RAGAS_NUM_QUERIES,
    DAY11_RAGAS_RUN_ID,
    DAY11_RAGAS_SCORES,
    BenchmarkConfigSummary,
    mock_quality_regression_history,
)

log = structlog.get_logger()

HistoryFn = Callable[[], list[DailyScore]]
SummariesFn = Callable[[], list[BenchmarkConfigSummary]]
RagasFn = Callable[[], tuple[dict[str, float], str, int]]
CitationFn = Callable[[], float]


def _load_summaries(summaries_fn: SummariesFn) -> tuple[list[BenchmarkConfigSummary], bool]:
    try:
        return summaries_fn(), True
    except Exception as exc:
        log.warning("eval_dashboard_benchmark_fetch_failed", stage="eval_dashboard", exc=str(exc))
        return DAY9_BENCHMARK_SUMMARIES, False


def _load_ragas(ragas_fn: RagasFn) -> tuple[dict[str, float], str, int, bool]:
    try:
        scores, run_id, num_queries = ragas_fn()
        return scores, run_id, num_queries, True
    except Exception as exc:
        log.warning("eval_dashboard_ragas_fetch_failed", stage="eval_dashboard", exc=str(exc))
        return DAY11_RAGAS_SCORES, DAY11_RAGAS_RUN_ID, DAY11_RAGAS_NUM_QUERIES, False


def _load_citation_accuracy(citation_fn: CitationFn) -> tuple[float, bool]:
    try:
        return citation_fn(), True
    except Exception as exc:
        log.warning("eval_dashboard_citation_fetch_failed", stage="eval_dashboard", exc=str(exc))
        return DAY9_CITATION_ACCURACY, False


def _headline_metrics(
    config_a: BenchmarkConfigSummary,
    ragas_scores: dict[str, float],
    ragas_run_id: str,
    ragas_num_queries: int,
    citation_accuracy: float,
    live: bool,
) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NDCG@10 (Config A)", f"{config_a.ndcg_at_10:.4f}", delta=f"gate {CI_NDCG_GATE:.2f}")
    col2.metric("Faithfulness", f"{ragas_scores['faithfulness']:.4f}")
    col3.metric("Answer relevancy", f"{ragas_scores['answer_relevancy']:.4f}")
    col4.metric("Citation accuracy", f"{citation_accuracy:.4f}")
    source = "📡 Live MLflow query" if live else "📈 Last-known results, MLflow unreachable right now"
    st.caption(
        f"{source}. NDCG@10/MRR: `reranker_benchmark` run `{config_a.run_id}` ({config_a.config}). "
        f"Faithfulness/answer_relevancy: `ragas_eval` run `{ragas_run_id}` ({ragas_num_queries} queries). "
        "Citation accuracy: `citation_judge`."
    )


def _ndcg_gate_chart(summaries: list[BenchmarkConfigSummary]) -> None:
    fig = go.Figure()
    for i, summary in enumerate(summaries):
        fig.add_bar(
            x=[summary.config],
            y=[summary.ndcg_at_10],
            name=summary.config,
            marker_color=theme.CATEGORICAL[i],
            text=[f"{summary.ndcg_at_10:.4f}"],
            textposition="outside",
        )
    fig.add_hline(
        y=CI_NDCG_GATE,
        line_dash="dash",
        line_color=theme.STATUS["critical"],
        annotation_text=f"CI gate ({CI_NDCG_GATE:.2f})",
        annotation_position="bottom right",
    )
    fig.update_layout(
        template=theme.PLOTLY_TEMPLATE,
        title="NDCG@10 vs CI eval gate",
        yaxis_title="NDCG@10",
        showlegend=False,
        height=340,
    )
    st.plotly_chart(fig, width="stretch")


def _citation_accuracy_by_query_chart(df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_bar(
        x=df["query_id"],
        y=df["accuracy"],
        marker_color=theme.CATEGORICAL[0],
        text=[f"{s}/{t}" for s, t in zip(df["supported"], df["total"])],
        textposition="outside",
    )
    fig.update_layout(
        template=theme.PLOTLY_TEMPLATE,
        title=f"Citation accuracy per query ({int(df['total'].sum())} real judged claims, Day 9)",
        yaxis_title="supported / total claims",
        yaxis_range=[0, 1.1],
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def _trend_chart_and_alert(history: list[DailyScore]) -> QualityRegressionResult:
    result = evaluate_regression(history)

    df = pd.DataFrame([{"date": d.date, **d.scores} for d in history])
    fig = go.Figure()
    fig.add_scatter(
        x=df["date"], y=df["faithfulness"], mode="lines+markers", name="faithfulness", line_color=theme.CATEGORICAL[0]
    )
    fig.add_scatter(
        x=df["date"],
        y=df["answer_relevancy"],
        mode="lines+markers",
        name="answer_relevancy",
        line_color=theme.CATEGORICAL[1],
    )
    fig.update_layout(
        template=theme.PLOTLY_TEMPLATE,
        title="Daily RAGAS sample (20 queries/day) — trailing 10 days",
        yaxis_title="score",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")
    return result


def _alert_banner(result: QualityRegressionResult) -> None:
    if result.is_regressing:
        st.error(f"🔴 {result.alert_message}")
    else:
        st.markdown(theme.status_badge("good", "No 3-day quality regression detected."))


def render(
    summaries_fn: SummariesFn = live_data.fetch_live_benchmark_summaries,
    ragas_fn: RagasFn = live_data.fetch_ragas_scores,
    citation_fn: CitationFn = live_data.fetch_citation_accuracy,
    history_fn: HistoryFn = mock_quality_regression_history,
) -> None:
    st.header("📊 Eval Dashboard")

    summaries, summaries_live = _load_summaries(summaries_fn)
    config_a = next(s for s in summaries if s.config == "config_A_ms_marco")
    ragas_scores, ragas_run_id, ragas_num_queries, ragas_live = _load_ragas(ragas_fn)
    citation_accuracy, citation_live = _load_citation_accuracy(citation_fn)
    all_live = summaries_live and ragas_live and citation_live

    st.caption(
        "📡 Live MLflow queries below, per metric — see each chart's own caption."
        if all_live
        else "📈 Some metrics fell back to last-known Day 9/11 results — see each chart's own caption."
    )
    _headline_metrics(config_a, ragas_scores, ragas_run_id, ragas_num_queries, citation_accuracy, all_live)
    st.divider()

    col_ndcg, col_trend = st.columns([1, 2])
    with col_ndcg:
        _ndcg_gate_chart(summaries)
    with col_trend:
        history = history_fn()
        result = _trend_chart_and_alert(history)

    _alert_banner(result)

    st.subheader("Per-query citation accuracy")
    try:
        judgments = live_data.load_citation_judgments()
        _citation_accuracy_by_query_chart(live_data.citation_accuracy_by_query(judgments))
    except Exception as exc:
        log.warning("eval_dashboard_citation_judgments_load_failed", stage="eval_dashboard", exc=str(exc))
        st.info("evaluation/day9_citation_judgments.json not found — per-query citation chart unavailable.")

    st.caption(
        "⚠️ Trend chart above is monitoring/quality_regression.py's real `evaluate_regression()` logic run over a "
        "synthetic 10-day history seeded from the real Day 11 baseline — no live daily-quality Postgres/JSON "
        "history exists yet (airflow/dags/drift_monitor.py's regression task hasn't run against a live scheduler)."
    )
    with st.expander("Per-metric regression detail"):
        detail_df = pd.DataFrame([r.model_dump() for r in result.regressions])
        st.dataframe(detail_df, width="stretch")


if __name__ == "__main__":
    st.set_page_config(page_title="nvidia-ir-rag-agent — Eval Dashboard", layout="wide")
    render()
