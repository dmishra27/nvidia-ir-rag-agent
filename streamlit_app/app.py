"""Streamlit UI entry point: composes the 5 Day 12 tabs into one app.

Run with `streamlit run streamlit_app/app.py`. Per Day 12 Task 3's note,
every tab is mock/historical data only -- no live retrieval, LLM, or model
load happens from this process.

Lazy tabs: by default `st.tabs()` sends every tab's content to the frontend
on *every* rerun regardless of which tab is selected -- tab switching is
purely client-side CSS show/hide, so there's no server round-trip for
Streamlit to defer work into. That made every page load pay for importing
all 5 tab modules' transitive dependency chains (`benchmark_tab`/
`eval_dashboard` pull in `evaluation.benchmark_runner`/`streamlit_app.live_data`
-> mlflow; `search_tab` pulls in `agents.qa_agent` -> langgraph/anthropic),
even for tabs the user never opens -- several minutes of import cost on a
memory-constrained machine before anything painted.

`on_change="rerun"` + each tab's `.open` property (both native to this
project's installed Streamlit 1.61.1 -- see `st.tabs`'s own docstring) fix
this without changing the UI at all: switching tabs now triggers a real
script rerun instead of a client-side-only swap, so `.open` accurately
reflects the selected tab post-rerun, and gating the *import statement*
itself (not just the `render()` call) behind `if tab.open:` means an
unselected tab's module -- and everything it imports -- is never touched.
Each module's own import stays cached in `sys.modules` for the rest of the
session once a tab has been opened once, so this only defers cost, it
doesn't repeat it.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="nvidia-ir-rag-agent", page_icon="🟩", layout="wide")

st.title("🟩 nvidia-ir-rag-agent")
st.caption(
    "Hybrid IR + RAG over NVIDIA technical docs — Day 12 UI shell. "
    "Search/Monitoring/Drift tabs are mock data; Eval/Benchmark tabs show real Day 9/11 historical results."
)

TAB_LABELS = ["🔍 Search", "📊 Eval Dashboard", "🩺 Monitoring", "⚖️ Benchmark", "🌊 Drift"]
tab_search, tab_eval, tab_monitoring, tab_benchmark, tab_drift = st.tabs(
    TAB_LABELS, on_change="rerun", key="active_tab"
)

if tab_search.open:
    with tab_search:
        from streamlit_app import search_tab

        search_tab.render()

if tab_eval.open:
    with tab_eval:
        from streamlit_app import eval_dashboard

        eval_dashboard.render()

if tab_monitoring.open:
    with tab_monitoring:
        from streamlit_app import monitoring_tab

        monitoring_tab.render()

if tab_benchmark.open:
    with tab_benchmark:
        from streamlit_app import benchmark_tab

        benchmark_tab.render()

if tab_drift.open:
    with tab_drift:
        from streamlit_app import drift_tab

        drift_tab.render()
