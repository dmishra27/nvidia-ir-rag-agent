"""Drift tab: PSI query-embedding drift + BM25 term-frequency shift.

Per Day 12 Task 3's note, this is a UI shell only -- monitoring/drift_detector.py
and monitoring/term_shift_monitor.py haven't been wired to a live Airflow DAG
yet (per Day 11's open items), so `render()` takes injectable
`drift_fn`/`term_shift_fn` (default streamlit_app/mock_data.py's synthetic
builders) over the real `DriftResult`/`TermShiftResult` shapes. Severity
labels and thresholds are read directly from monitoring/drift_detector.py
rather than re-declared here, so this tab can't drift out of sync with the
monitor's actual alerting logic.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitoring.drift_detector import MODERATE_DRIFT_THRESHOLD, SIGNIFICANT_DRIFT_THRESHOLD, DriftResult
from monitoring.term_shift_monitor import TermShiftResult
from streamlit_app import theme
from streamlit_app.mock_data import mock_drift_result, mock_term_shift_result

DriftFn = Callable[[], DriftResult]
TermShiftFn = Callable[[], TermShiftResult]

_SEVERITY_STATUS = {"none": "good", "moderate": "warning", "significant": "critical"}


def _psi_gauge(result: DriftResult) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result.psi,
            number={"valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 0.5]},
                "bar": {"color": theme.CATEGORICAL[0]},
                "steps": [
                    {"range": [0, MODERATE_DRIFT_THRESHOLD], "color": theme.STATUS["good"]},
                    {"range": [MODERATE_DRIFT_THRESHOLD, SIGNIFICANT_DRIFT_THRESHOLD], "color": theme.STATUS["warning"]},
                    {"range": [SIGNIFICANT_DRIFT_THRESHOLD, 0.5], "color": theme.STATUS["critical"]},
                ],
            },
            title={"text": "PSI (query embedding drift)"},
        )
    )
    fig.update_layout(template=theme.PLOTLY_TEMPLATE, height=300)
    st.plotly_chart(fig, width="stretch")


def _term_shift_chart(result: TermShiftResult) -> None:
    df = pd.DataFrame([t.model_dump() for t in result.shifted_terms]).sort_values("delta")
    colors = [theme.STATUS["good"] if d > 0 else theme.STATUS["critical"] for d in df["delta"]]
    fig = go.Figure(go.Bar(x=df["delta"], y=df["term"], orientation="h", marker_color=colors))
    fig.update_layout(
        template=theme.PLOTLY_TEMPLATE,
        title="Top shifted terms (Δ relative frequency)",
        xaxis_title="current − baseline",
        height=340,
    )
    st.plotly_chart(fig, width="stretch")


def render(drift_fn: DriftFn = mock_drift_result, term_shift_fn: TermShiftFn = mock_term_shift_result) -> None:
    st.header("🌊 Drift")
    st.caption(
        "🧪 Mock data — UI shell only; monitoring/drift_detector.py + term_shift_monitor.py "
        "not yet wired to a live Airflow DAG (Day 11 open item)."
    )

    drift = drift_fn()
    term_shift = term_shift_fn()

    status = _SEVERITY_STATUS[drift.severity]
    st.markdown(theme.status_badge(status, f"Drift severity: {drift.severity}"))

    col_gauge, col_meta = st.columns([1, 1])
    with col_gauge:
        _psi_gauge(drift)
    with col_meta:
        st.metric("Baseline sample size", drift.baseline_size)
        st.metric("Current sample size", drift.current_size)
        st.metric("Is drifted", "Yes" if drift.is_drifted else "No")

    st.divider()
    st.subheader("Term-frequency shift")
    col_chart, col_lists = st.columns([2, 1])
    with col_chart:
        _term_shift_chart(term_shift)
    with col_lists:
        st.markdown(f"**New terms** ({len(term_shift.new_terms)})")
        st.write(", ".join(term_shift.new_terms) or "none")
        st.markdown(f"**Dropped terms** ({len(term_shift.dropped_terms)})")
        st.write(", ".join(term_shift.dropped_terms) or "none")


if __name__ == "__main__":
    st.set_page_config(page_title="nvidia-ir-rag-agent — Drift", layout="wide")
    render()
