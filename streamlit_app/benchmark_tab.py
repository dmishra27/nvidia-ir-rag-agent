"""Benchmark tab: Config A (ms-marco) vs Config C (Cohere Rerank v3).

Per Day 12 Task 3's note, this shows Day 9's real live benchmark results
(evaluation/benchmark_runner.py, MLflow experiment `reranker_benchmark`) as
static historical data -- not a live MLflow query. `render()` takes an
injectable `summaries_fn` (default streamlit_app/mock_data.py's
`DAY9_BENCHMARK_SUMMARIES`) so a real `mcp-mlflow`-backed fetch can replace
it later behind the same `() -> list[BenchmarkConfigSummary]` signature.
Config B (bge-reranker-v2-m3) is shown as "not run" per
evaluation/benchmark_runner.py's `CONFIG_B_DEFERRED_REASON` -- not silently
omitted.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evaluation.benchmark_runner import CONFIG_B_DEFERRED_REASON
from streamlit_app import theme
from streamlit_app.mock_data import DAY9_BENCHMARK_NUM_QUERIES, DAY9_BENCHMARK_SUMMARIES, BenchmarkConfigSummary

SummariesFn = Callable[[], list[BenchmarkConfigSummary]]

_METRICS = [("ndcg_at_10", "NDCG@10"), ("mrr", "MRR"), ("prec_at_3", "Prec@3")]


def _quality_chart(summaries: list[BenchmarkConfigSummary]) -> None:
    fig = go.Figure()
    for i, summary in enumerate(summaries):
        values = [getattr(summary, key) for key, _ in _METRICS]
        fig.add_bar(
            x=[label for _, label in _METRICS],
            y=values,
            name=summary.config,
            marker_color=theme.CATEGORICAL[i],
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        )
    fig.update_layout(
        template=theme.PLOTLY_TEMPLATE,
        title="Ranking quality",
        barmode="group",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")


def _latency_cost_chart(summaries: list[BenchmarkConfigSummary]) -> None:
    # Two different units (ms vs USD) -> two charts, never one dual-axis plot.
    fig = go.Figure()
    for i, summary in enumerate(summaries):
        fig.add_bar(x=[summary.config], y=[summary.latency_ms], marker_color=theme.CATEGORICAL[i], showlegend=False)
    fig.update_layout(template=theme.PLOTLY_TEMPLATE, title="Latency (ms/query)", yaxis_title="ms", height=320)
    st.plotly_chart(fig, width="stretch")


def render(summaries_fn: SummariesFn = lambda: DAY9_BENCHMARK_SUMMARIES) -> None:
    st.header("⚖️ Benchmark: Config A vs C")
    st.caption(
        f"📈 Day 9 live benchmark, {DAY9_BENCHMARK_NUM_QUERIES} cached queries × top-3 RRF pool — "
        "historical results, not a live query."
    )

    summaries = summaries_fn()

    cols = st.columns(len(summaries))
    for col, summary in zip(cols, summaries):
        col.metric(summary.config, f"NDCG@10 {summary.ndcg_at_10:.4f}", f"run {summary.run_id}")

    col_quality, col_latency = st.columns([2, 1])
    with col_quality:
        _quality_chart(summaries)
    with col_latency:
        _latency_cost_chart(summaries)

    st.subheader("All metrics")
    st.dataframe(pd.DataFrame([s.model_dump() for s in summaries]), width="stretch")

    st.warning(f"**Config B (bge-reranker-v2-m3): not run.** {CONFIG_B_DEFERRED_REASON}")


if __name__ == "__main__":
    st.set_page_config(page_title="nvidia-ir-rag-agent — Benchmark", layout="wide")
    render()
