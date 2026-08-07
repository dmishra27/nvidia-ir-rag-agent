"""Monitoring tab: per-stage latency (p50/p95) and daily error rate.

Per Day 12 Task 3's note, this is a UI shell only -- no live query against
`request_log`/`error_log` (schema/schema.sql tables 8-9). `render()` takes
injectable `latency_fn`/`errors_fn` (default streamlit_app/mock_data.py's
deterministic mock builders) so a real Postgres/mcp-postgres-backed fetch
can be wired in later behind the same `() -> pd.DataFrame` signature.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_app import theme
from streamlit_app.mock_data import mock_error_timeseries, mock_stage_latency

LatencyFn = Callable[[], pd.DataFrame]
ErrorsFn = Callable[[], pd.DataFrame]


def _latency_chart(df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_bar(x=df["stage"], y=df["p50_ms"], name="p50", marker_color=theme.CATEGORICAL[0])
    fig.add_bar(x=df["stage"], y=df["p95_ms"], name="p95", marker_color=theme.CATEGORICAL[1])
    fig.update_layout(
        template=theme.PLOTLY_TEMPLATE,
        title="Per-stage latency (api/telemetry.py's traced stages)",
        yaxis_title="ms",
        barmode="group",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")


def _error_rate_chart(df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_scatter(
        x=df["date"], y=df["error_rate"], mode="lines+markers", name="error rate", line_color=theme.STATUS["serious"]
    )
    fig.update_layout(
        template=theme.PLOTLY_TEMPLATE,
        title="Daily error rate (errors / requests)",
        yaxis_title="error rate",
        yaxis_tickformat=".1%",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render(latency_fn: LatencyFn = mock_stage_latency, errors_fn: ErrorsFn = mock_error_timeseries) -> None:
    st.header("🩺 Monitoring")
    st.caption("🧪 Mock data — UI shell only, not wired to a live request_log/error_log query (Day 12).")

    errors_df = errors_fn()
    total_requests = int(errors_df["requests"].sum())
    total_errors = int(errors_df["errors"].sum())
    error_rate = total_errors / total_requests if total_requests else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Requests (14d)", f"{total_requests:,}")
    col2.metric("Errors (14d)", f"{total_errors:,}")
    col3.metric("Error rate (14d)", f"{error_rate:.2%}")

    col_latency, col_errors = st.columns(2)
    with col_latency:
        _latency_chart(latency_fn())
    with col_errors:
        _error_rate_chart(errors_df)


if __name__ == "__main__":
    st.set_page_config(page_title="nvidia-ir-rag-agent — Monitoring", layout="wide")
    render()
