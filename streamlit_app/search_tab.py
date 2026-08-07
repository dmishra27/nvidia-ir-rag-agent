"""Search tab: query -> retrieved passages -> cited answer.

Per Day 12 Task 3's note, this is a UI shell only -- no call into
agents/qa_agent.py's `run()` (which would load BM25/dense indexes and call
a live Claude model). `render()` takes an injectable `qa_state_fn` (default
streamlit_app/mock_data.py's `mock_qa_state`) so the real pipeline can be
wired in later behind the same `str -> QAState` signature, mirroring this
project's constructor-injection convention (e.g. agents/retrieval_agent.py's
node factories) without any change to the rendering code below.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from agents.qa_agent import QAState
from streamlit_app.mock_data import mock_qa_state

QAStateFn = Callable[[str], QAState]


def _render_passages(state: QAState) -> None:
    passages = state.reranked_results or state.fused_results
    st.subheader(f"Retrieved passages ({len(passages)})")
    for c in passages:
        with st.expander(f"[{c.chunk_id}] score={c.score:.2f} · rank {c.rank}"):
            st.write(c.text)


def _render_answer(state: QAState) -> None:
    st.subheader("Answer")
    if state.error:
        st.error(state.error)
        return
    if not state.answer:
        st.info("No answer yet -- run a search above.")
        return
    st.markdown(state.answer)

    if state.citations:
        st.caption("Citations")
        for c in state.citations:
            st.markdown(f"- *{c.claim}* — `{', '.join(c.chunk_ids) or 'none'}`")


def render(qa_state_fn: QAStateFn = mock_qa_state) -> None:
    st.header("🔍 Search")
    st.caption("🧪 Mock data — UI shell only, no live retrieval or LLM call (Day 12).")

    query = st.text_input("Query", value="cudaMalloc function parameters", key="search_tab_query")
    top_k = st.slider("Top-k passages", min_value=1, max_value=10, value=3, key="search_tab_top_k")

    if st.button("Search", key="search_tab_button") or query:
        state = qa_state_fn(query)
        state = state.model_copy(update={"reranked_results": state.reranked_results[:top_k]})
        col_passages, col_answer = st.columns(2)
        with col_passages:
            _render_passages(state)
        with col_answer:
            _render_answer(state)


if __name__ == "__main__":
    st.set_page_config(page_title="nvidia-ir-rag-agent — Search", layout="wide")
    render()
