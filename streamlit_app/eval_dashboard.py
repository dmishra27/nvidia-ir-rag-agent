"""Eval dashboard tab: NDCG@10 · faithfulness · answer_relevancy · citation
accuracy, plus a daily quality-regression trend.

Per Day 12 Task 3's note, no live MLflow/Postgres query runs here -- the
headline numbers are this project's own real historical results (Day 9's
Config A benchmark, Day 11's live RAGAS run), sourced back to their
run_ids in streamlit_app/mock_data.py. The trend chart *is* wired to real
logic, though: it runs monitoring/quality_regression.py's actual
`evaluate_regression()` over a synthetic 10-day history, so the alert
state shown here is computed the same way the Day 12 Task 1 monitor
computes it, not re-implemented in the UI layer.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitoring.quality_regression import DailyScore, QualityRegressionResult, evaluate_regression
from streamlit_app import theme
from streamlit_app.mock_data import (
    CI_NDCG_GATE,
    DAY9_BENCHMARK_SUMMARIES,
    DAY9_CITATION_ACCURACY,
    DAY11_RAGAS_NUM_QUERIES,
    DAY11_RAGAS_RUN_ID,
    DAY11_RAGAS_SCORES,
    mock_quality_regression_history,
)

HistoryFn = Callable[[], list[DailyScore]]


def _headline_metrics() -> None:
    config_a = next(s for s in DAY9_BENCHMARK_SUMMARIES if s.config == "config_A_ms_marco")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NDCG@10 (Config A)", f"{config_a.ndcg_at_10:.4f}", delta=f"gate {CI_NDCG_GATE:.2f}")
    col2.metric("Faithfulness", f"{DAY11_RAGAS_SCORES['faithfulness']:.4f}")
    col3.metric("Answer relevancy", f"{DAY11_RAGAS_SCORES['answer_relevancy']:.4f}")
    col4.metric("Citation accuracy", f"{DAY9_CITATION_ACCURACY:.4f}")
    st.caption(
        f"NDCG@10/MRR: Day 9 benchmark, run `{config_a.run_id}` ({DAY9_BENCHMARK_SUMMARIES[0].config}). "
        f"Faithfulness/answer_relevancy: Day 11 RAGAS run `{DAY11_RAGAS_RUN_ID}` ({DAY11_RAGAS_NUM_QUERIES} queries). "
        "Citation accuracy: Day 9 citation judge."
    )


def _ndcg_gate_chart() -> None:
    fig = go.Figure()
    for i, summary in enumerate(DAY9_BENCHMARK_SUMMARIES):
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


def render(history_fn: HistoryFn = mock_quality_regression_history) -> None:
    st.header("📊 Eval Dashboard")
    st.caption("📈 Historical results from Day 9/11 live runs, not a live query — see captions per chart.")

    _headline_metrics()
    st.divider()

    col_ndcg, col_trend = st.columns([1, 2])
    with col_ndcg:
        _ndcg_gate_chart()
    with col_trend:
        history = history_fn()
        result = _trend_chart_and_alert(history)

    _alert_banner(result)

    with st.expander("Per-metric regression detail"):
        detail_df = pd.DataFrame([r.model_dump() for r in result.regressions])
        st.dataframe(detail_df, width="stretch")


if __name__ == "__main__":
    st.set_page_config(page_title="nvidia-ir-rag-agent — Eval Dashboard", layout="wide")
    render()
