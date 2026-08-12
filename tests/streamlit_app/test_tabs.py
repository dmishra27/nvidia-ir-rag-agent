"""Smoke tests for streamlit_app/*.py using Streamlit's AppTest harness
(streamlit.testing.v1) -- runs each script in Streamlit's real script-runner
without a browser or server, so these catch genuine Streamlit API errors
(bad widget kwargs, missing imports, exceptions during render) that a plain
`import` would miss.

Per Day 12 Task 3's note ("Streamlit components only -- no live data calls,
no model loading"), every tab's default data source is one of
streamlit_app/mock_data.py's deterministic builders, so these tests never
touch Postgres/Qdrant/MLflow or a live model -- consistent with
AGENTS.md's mock-everything-in-tests rule.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves relative paths against the *calling file's*
# directory, not the pytest working directory (pytest.ini sets
# `pythonpath = .` at the repo root, which AppTest doesn't use) -- so every
# path here is anchored to the repo root explicitly instead.
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_live_mlflow_or_postgres() -> None:
    """DEF-04: benchmark_tab.py's and eval_dashboard.py's `render()` default
    to streamlit_app/live_data.py's live fetchers, which reach for a real
    MLflow/Postgres. Against an unreachable MLflow, live_data._mlflow_client()'s
    own 15s timeout (set deliberately generous there for a real deployment --
    see its docstring) turns each of the ~5 affected `render()` calls in this
    file into a 15s stall, stretching a ~2min suite past 7 minutes, and
    non-deterministically depending on what's running locally.

    Patching _mlflow_client/get_engine (not the fetch_* functions themselves)
    works even though benchmark_tab.render()/eval_dashboard.render() already
    bound live_data.fetch_* as default argument values at import time -- each
    fetch_* function still looks up _mlflow_client/get_engine through
    live_data's module namespace on every call, so the patched version is
    what actually gets invoked. Every affected test only asserts on tab
    content that's identical whether it came from a live query or the
    mock_data fallback (e.g. config names, metric labels), so failing fast
    into that already-tested fallback path is a correctness-neutral, and
    much faster + deterministic, substitute for a live service.
    """
    with (
        patch("streamlit_app.live_data._mlflow_client", side_effect=RuntimeError("mocked: no live MLflow in tests")),
        patch("streamlit_app.live_data.get_engine", side_effect=RuntimeError("mocked: no live Postgres in tests")),
    ):
        yield


def _run(relative_path: str) -> AppTest:
    at = AppTest.from_file(str(REPO_ROOT / relative_path), default_timeout=30)
    at.run()
    return at


def test_app_composes_all_five_tabs_without_exceptions() -> None:
    at = _run("streamlit_app/app.py")

    assert not at.exception
    assert len(at.tabs) == 5


def test_search_tab_renders_query_passages_and_answer() -> None:
    at = _run("streamlit_app/search_tab.py")

    assert not at.exception
    headers = [h.value for h in at.header]
    assert any("Search" in h for h in headers)
    subheaders = [s.value for s in at.subheader]
    assert any("Retrieved passages" in s for s in subheaders)
    assert any("Answer" in s for s in subheaders)


def test_search_tab_answer_contains_a_citation() -> None:
    at = _run("streamlit_app/search_tab.py")

    markdown_text = " ".join(m.value for m in at.markdown)
    assert "c_cuda_malloc_01" in markdown_text


def test_eval_dashboard_renders_headline_metrics() -> None:
    at = _run("streamlit_app/eval_dashboard.py")

    assert not at.exception
    metric_labels = {m.label for m in at.metric}
    assert "Faithfulness" in metric_labels
    assert "Answer relevancy" in metric_labels
    assert "Citation accuracy" in metric_labels


def test_eval_dashboard_no_regression_by_default() -> None:
    at = _run("streamlit_app/eval_dashboard.py")

    # mock_data.mock_quality_regression_history() bakes in a deliberate
    # 3-day decline, so the real evaluate_regression() call should flag it.
    assert any(e for e in at.error)


def test_monitoring_tab_renders_request_and_error_metrics() -> None:
    at = _run("streamlit_app/monitoring_tab.py")

    assert not at.exception
    metric_labels = {m.label for m in at.metric}
    assert "Requests (14d)" in metric_labels
    assert "Errors (14d)" in metric_labels
    assert "Error rate (14d)" in metric_labels


def test_benchmark_tab_shows_config_a_and_c() -> None:
    at = _run("streamlit_app/benchmark_tab.py")

    assert not at.exception
    metric_labels = {m.label for m in at.metric}
    assert "config_A_ms_marco" in metric_labels
    assert "config_C_cohere_rerank" in metric_labels


def test_benchmark_tab_documents_config_b_as_not_run() -> None:
    at = _run("streamlit_app/benchmark_tab.py")

    warnings = [w.value for w in at.warning]
    assert any("Config B" in w and "not run" in w for w in warnings)


def test_drift_tab_renders_psi_and_term_shift() -> None:
    at = _run("streamlit_app/drift_tab.py")

    assert not at.exception
    metric_labels = {m.label for m in at.metric}
    assert "Baseline sample size" in metric_labels
    assert "Current sample size" in metric_labels
    subheaders = [s.value for s in at.subheader]
    assert any("Term-frequency shift" in s for s in subheaders)


def test_drift_tab_severity_badge_matches_mock_result() -> None:
    from streamlit_app.mock_data import mock_drift_result

    at = _run("streamlit_app/drift_tab.py")

    expected_severity = mock_drift_result().severity
    markdown_text = " ".join(m.value for m in at.markdown)
    assert expected_severity in markdown_text
