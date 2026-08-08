"""Benchmark tab: Config A (ms-marco) vs Config C (Cohere Rerank v3).

Day 13: wired to real live data. `_load_summaries()` queries MLflow's
`reranker_benchmark` experiment (mcp/mcp_mlflow/server.py's same
MlflowClient pattern) for the headline metrics/quality/latency charts, and
`_load_per_query_ndcg()` queries Postgres' `benchmark_results` table (the
same real Day 9 run's 45 rows, via the ORM) for a per-query NDCG
distribution chart -- not just the two aggregate charts Day 12 shipped.
Both fall back to streamlit_app/mock_data.py's committed Day 9 numbers
(the *same* real values, just not live-queried) with a visible banner if
MLflow/Postgres aren't reachable, so the tab still renders in CI (no live
services there, per AGENTS.md's "no live API calls in CI") and in a fresh
clone before `docker-compose up`.

`render()` keeps `summaries_fn`/`per_query_fn` injection points so tests can
supply canned data without a network call, per this project's
mock-everything-in-tests convention -- streamlit_app/live_data.py's real
fetchers are just the new default, not the only path.

Config B (bge-reranker-v2-m3) is shown as "not run" per
evaluation/benchmark_runner.py's `CONFIG_B_DEFERRED_REASON` -- not silently
omitted.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from evaluation.benchmark_runner import CONFIG_B_DEFERRED_REASON
from streamlit_app import live_data, theme
from streamlit_app.mock_data import DAY9_BENCHMARK_NUM_QUERIES, DAY9_BENCHMARK_SUMMARIES, BenchmarkConfigSummary

log = structlog.get_logger()

SummariesFn = Callable[[], list[BenchmarkConfigSummary]]
PerQueryFn = Callable[[], pd.DataFrame]

_METRICS = [("ndcg_at_10", "NDCG@10"), ("mrr", "MRR"), ("prec_at_3", "Prec@3")]


def _load_summaries(summaries_fn: SummariesFn) -> tuple[list[BenchmarkConfigSummary], bool]:
    """(summaries, is_live). Falls back to Day 9's committed numbers -- the
    same real values, just not live-queried -- on any failure (MLflow
    unreachable, experiment missing, etc.)."""
    try:
        return summaries_fn(), True
    except Exception as exc:
        log.warning("benchmark_tab_live_fetch_failed", stage="benchmark_tab", exc=str(exc))
        return DAY9_BENCHMARK_SUMMARIES, False


def _load_per_query_ndcg(per_query_fn: PerQueryFn) -> tuple[pd.DataFrame | None, bool]:
    try:
        return per_query_fn(), True
    except Exception as exc:
        log.warning("benchmark_tab_per_query_fetch_failed", stage="benchmark_tab", exc=str(exc))
        return None, False


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


def _ndcg_distribution_chart(df: pd.DataFrame) -> None:
    fig = go.Figure()
    for i, config in enumerate(sorted(df["config"].unique())):
        fig.add_box(
            y=df.loc[df["config"] == config, "ndcg_at_10"],
            name=config,
            marker_color=theme.CATEGORICAL[i],
            boxmean=True,
        )
    fig.update_layout(
        template=theme.PLOTLY_TEMPLATE,
        title=f"Per-query NDCG@10 distribution ({len(df)} live benchmark_results rows)",
        yaxis_title="NDCG@10",
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render(
    summaries_fn: SummariesFn = live_data.fetch_live_benchmark_summaries,
    per_query_fn: PerQueryFn = live_data.fetch_per_query_ndcg,
) -> None:
    st.header("⚖️ Benchmark: Config A vs C")

    summaries, summaries_live = _load_summaries(summaries_fn)
    if summaries_live:
        st.caption("📡 Live MLflow query — `reranker_benchmark` experiment, most recent run per config.")
    else:
        st.caption(
            f"📈 Day 9 live benchmark, {DAY9_BENCHMARK_NUM_QUERIES} cached queries × top-3 RRF pool — "
            "last-known results, MLflow unreachable right now."
        )
        st.info("MLflow isn't reachable — showing Day 9's committed historical results instead.")

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

    st.subheader("Per-query NDCG distribution")
    per_query_df, per_query_live = _load_per_query_ndcg(per_query_fn)
    if per_query_live and per_query_df is not None:
        _ndcg_distribution_chart(per_query_df)
    else:
        st.info("Postgres isn't reachable — the per-query distribution chart needs a live `benchmark_results` read.")

    st.warning(f"**Config B (bge-reranker-v2-m3): not run.** {CONFIG_B_DEFERRED_REASON}")


if __name__ == "__main__":
    st.set_page_config(page_title="nvidia-ir-rag-agent — Benchmark", layout="wide")
    render()
