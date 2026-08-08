"""Unit tests for benchmark_tab.py/eval_dashboard.py's live-fetch-with-
fallback helpers (`_load_summaries`, `_load_per_query_ndcg`, `_load_ragas`,
`_load_citation_accuracy`) -- called directly with injected fake functions,
no Streamlit runtime and no real MLflow/Postgres connection needed, per
AGENTS.md's mock-everything-external convention.
"""

from __future__ import annotations

import pandas as pd
import pytest

from streamlit_app import benchmark_tab, eval_dashboard
from streamlit_app.mock_data import (
    DAY9_BENCHMARK_SUMMARIES,
    DAY9_CITATION_ACCURACY,
    DAY11_RAGAS_NUM_QUERIES,
    DAY11_RAGAS_RUN_ID,
    DAY11_RAGAS_SCORES,
)


def _raises() -> None:
    raise RuntimeError("service unreachable")


# ---------------------------------------------------------------------------
# benchmark_tab.py
# ---------------------------------------------------------------------------


def test_load_summaries_returns_live_data_on_success() -> None:
    live = [*DAY9_BENCHMARK_SUMMARIES]
    summaries, is_live = benchmark_tab._load_summaries(lambda: live)
    assert summaries is live
    assert is_live is True


def test_load_summaries_falls_back_on_failure() -> None:
    summaries, is_live = benchmark_tab._load_summaries(_raises)
    assert summaries == DAY9_BENCHMARK_SUMMARIES
    assert is_live is False


def test_load_per_query_ndcg_returns_live_data_on_success() -> None:
    df = pd.DataFrame({"config": ["config_A_ms_marco"], "ndcg_at_10": [0.5]})
    result, is_live = benchmark_tab._load_per_query_ndcg(lambda: df)
    assert result is df
    assert is_live is True


def test_load_per_query_ndcg_falls_back_on_failure() -> None:
    result, is_live = benchmark_tab._load_per_query_ndcg(_raises)
    assert result is None
    assert is_live is False


# ---------------------------------------------------------------------------
# eval_dashboard.py
# ---------------------------------------------------------------------------


def test_load_summaries_eval_dashboard_falls_back_on_failure() -> None:
    summaries, is_live = eval_dashboard._load_summaries(_raises)
    assert summaries == DAY9_BENCHMARK_SUMMARIES
    assert is_live is False


def test_load_ragas_returns_live_data_on_success() -> None:
    scores, run_id, num_queries, is_live = eval_dashboard._load_ragas(
        lambda: ({"faithfulness": 0.9, "answer_relevancy": 0.8}, "abc12345", 42)
    )
    assert scores == {"faithfulness": 0.9, "answer_relevancy": 0.8}
    assert run_id == "abc12345"
    assert num_queries == 42
    assert is_live is True


def test_load_ragas_falls_back_on_failure() -> None:
    scores, run_id, num_queries, is_live = eval_dashboard._load_ragas(_raises)  # type: ignore[arg-type]
    assert scores == DAY11_RAGAS_SCORES
    assert run_id == DAY11_RAGAS_RUN_ID
    assert num_queries == DAY11_RAGAS_NUM_QUERIES
    assert is_live is False


def test_load_citation_accuracy_returns_live_data_on_success() -> None:
    accuracy, is_live = eval_dashboard._load_citation_accuracy(lambda: 0.9999)
    assert accuracy == 0.9999
    assert is_live is True


def test_load_citation_accuracy_falls_back_on_failure() -> None:
    accuracy, is_live = eval_dashboard._load_citation_accuracy(_raises)  # type: ignore[arg-type]
    assert accuracy == DAY9_CITATION_ACCURACY
    assert is_live is False


@pytest.mark.parametrize("fn", [benchmark_tab._load_summaries, eval_dashboard._load_summaries])
def test_load_summaries_helpers_never_raise(fn: object) -> None:
    # Both tabs' _load_summaries must swallow the failure, not propagate it --
    # a live-service outage should degrade the tab, not crash it.
    result = fn(_raises)  # type: ignore[operator]
    assert result[1] is False
