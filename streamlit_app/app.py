"""Streamlit UI entry point: composes the 5 Day 12 tabs into one app.

Run with `streamlit run streamlit_app/app.py`. Per Day 12 Task 3's note,
every tab is mock/historical data only -- no live retrieval, LLM, or model
load happens from this process, so the app is safe to run on this host's
memory-constrained setup (see AGENTS.md's environment isolation note /
[[user-host-memory-constraint]]) without touching BM25/Qdrant/torch.

Each tab module owns its own `render()` (and its own injectable data-source
default), so this file only imports and places them inside `st.tabs()` --
none of the 5 tab files call `render()` at import time, which is what lets
them compose here instead of racing to draw outside any tab's container.
"""

from __future__ import annotations

import streamlit as st

from streamlit_app import benchmark_tab, drift_tab, eval_dashboard, monitoring_tab, search_tab

st.set_page_config(page_title="nvidia-ir-rag-agent", page_icon="🟩", layout="wide")

st.title("🟩 nvidia-ir-rag-agent")
st.caption(
    "Hybrid IR + RAG over NVIDIA technical docs — Day 12 UI shell. "
    "Search/Monitoring/Drift tabs are mock data; Eval/Benchmark tabs show real Day 9/11 historical results."
)

TAB_LABELS = ["🔍 Search", "📊 Eval Dashboard", "🩺 Monitoring", "⚖️ Benchmark", "🌊 Drift"]
tab_search, tab_eval, tab_monitoring, tab_benchmark, tab_drift = st.tabs(TAB_LABELS)

with tab_search:
    search_tab.render()
with tab_eval:
    eval_dashboard.render()
with tab_monitoring:
    monitoring_tab.render()
with tab_benchmark:
    benchmark_tab.render()
with tab_drift:
    drift_tab.render()
