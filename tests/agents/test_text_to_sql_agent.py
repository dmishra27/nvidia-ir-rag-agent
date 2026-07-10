from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.text_to_sql_agent import (
    TextToSQLState,
    _dispatch,
    execute_query,
    format_answer,
    plan_query,
    run,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_response(name: str, input: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input
    resp = MagicMock()
    resp.content = [block]
    return resp


def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_db_patches():
    """Return patches for get_engine and get_session_factory as a context manager stack."""
    patch_engine = patch("agents.text_to_sql_agent.get_engine")
    patch_sf = patch("agents.text_to_sql_agent.get_session_factory")
    return patch_engine, patch_sf


def _configure_session(mock_sf: MagicMock) -> MagicMock:
    """Wire mock_sf so that SessionFactory() works as a context manager returning a session."""
    mock_session = MagicMock()
    mock_sf.return_value.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sf.return_value.return_value.__exit__ = MagicMock(return_value=False)
    return mock_session


# ── plan_query ────────────────────────────────────────────────────────────────

class TestPlanQuery:
    def test_sets_tool_name_from_response(self):
        state = TextToSQLState(question="How many docs per GPU family?")
        mock_resp = _tool_response("count_docs_by_gpu_family", {})

        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = plan_query(state)

        assert result.tool_name == "count_docs_by_gpu_family"

    def test_sets_tool_params_from_response(self):
        state = TextToSQLState(question="Show chunks below 0.4 quality.")
        mock_resp = _tool_response("chunks_below_quality_threshold", {"threshold": 0.4})

        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = plan_query(state)

        assert result.tool_params == {"threshold": 0.4}

    def test_preserves_query_id(self):
        state = TextToSQLState(question="List docs.", query_id="abc12345")
        mock_resp = _tool_response("list_indexed_documents", {})

        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = plan_query(state)

        assert result.query_id == "abc12345"

    def test_sets_error_when_no_tool_block(self):
        state = TextToSQLState(question="What?")
        # response with only a text block — no tool_use
        text_block = MagicMock()
        text_block.type = "text"
        mock_resp = MagicMock()
        mock_resp.content = [text_block]

        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = plan_query(state)

        assert result.error is not None
        assert result.tool_name is None

    def test_calls_anthropic_with_any_tool_choice(self):
        state = TextToSQLState(question="avg quality?")
        mock_resp = _tool_response("avg_quality_per_doc", {})

        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            plan_query(state)
            _, kwargs = MockClient.return_value.messages.create.call_args
            assert kwargs.get("tool_choice") == {"type": "any"}


# ── execute_query ─────────────────────────────────────────────────────────────

class TestExecuteQuery:
    def test_skips_when_state_has_error(self):
        state = TextToSQLState(question="q", tool_name="list_indexed_documents", error="oops")

        with patch("agents.text_to_sql_agent._dispatch") as mock_dispatch:
            result = execute_query(state)

        mock_dispatch.assert_not_called()
        assert result.error == "oops"

    def test_skips_when_tool_name_is_none(self):
        state = TextToSQLState(question="q", tool_name=None)

        with patch("agents.text_to_sql_agent._dispatch") as mock_dispatch:
            result = execute_query(state)

        mock_dispatch.assert_not_called()

    def test_stores_dispatch_result_in_sql_result(self):
        state = TextToSQLState(
            question="q", tool_name="count_docs_by_gpu_family", tool_params={}
        )
        expected = [{"gpu_family": "Hopper", "count": 3}]

        p_engine, p_sf = _make_db_patches()
        with p_engine, p_sf as mock_sf, \
             patch("agents.text_to_sql_agent._dispatch", return_value=expected):
            _configure_session(mock_sf)
            result = execute_query(state)

        assert result.sql_result == expected

    def test_passes_tool_params_to_dispatch(self):
        state = TextToSQLState(
            question="q",
            tool_name="chunks_below_quality_threshold",
            tool_params={"threshold": 0.5},
        )

        p_engine, p_sf = _make_db_patches()
        with p_engine, p_sf as mock_sf, \
             patch("agents.text_to_sql_agent._dispatch", return_value=[]) as mock_dispatch:
            mock_session = _configure_session(mock_sf)
            execute_query(state)
            _, kwargs = mock_dispatch.call_args
            assert mock_dispatch.call_args[0][1] == "chunks_below_quality_threshold"
            assert mock_dispatch.call_args[0][2] == {"threshold": 0.5}

    def test_sets_error_on_db_exception(self):
        state = TextToSQLState(
            question="q", tool_name="list_indexed_documents", tool_params={}
        )

        p_engine, p_sf = _make_db_patches()
        with p_engine, p_sf as mock_sf, \
             patch("agents.text_to_sql_agent._dispatch", side_effect=RuntimeError("db down")):
            _configure_session(mock_sf)
            result = execute_query(state)

        assert result.error == "db down"
        assert result.sql_result is None


# ── _dispatch ─────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_routes_count_docs_by_gpu_family(self):
        session = MagicMock()
        with patch("agents.text_to_sql_agent._count_docs_by_gpu_family", return_value=[]) as fn:
            _dispatch(session, "count_docs_by_gpu_family", {})
            fn.assert_called_once_with(session)

    def test_routes_chunks_below_threshold_with_float(self):
        session = MagicMock()
        with patch("agents.text_to_sql_agent._chunks_below_threshold", return_value=[]) as fn:
            _dispatch(session, "chunks_below_quality_threshold", {"threshold": 0.3})
            fn.assert_called_once_with(session, 0.3)

    def test_routes_avg_quality_per_doc(self):
        session = MagicMock()
        with patch("agents.text_to_sql_agent._avg_quality_per_doc", return_value=[]) as fn:
            _dispatch(session, "avg_quality_per_doc", {})
            fn.assert_called_once_with(session)

    def test_routes_chunk_count_for_doc_with_fragment(self):
        session = MagicMock()
        with patch("agents.text_to_sql_agent._chunk_count_for_doc", return_value=[]) as fn:
            _dispatch(session, "chunk_count_for_doc", {"title_fragment": "CUDA"})
            fn.assert_called_once_with(session, "CUDA")

    def test_routes_list_indexed_documents(self):
        session = MagicMock()
        with patch("agents.text_to_sql_agent._list_indexed_documents", return_value=[]) as fn:
            _dispatch(session, "list_indexed_documents", {})
            fn.assert_called_once_with(session)

    def test_raises_on_unknown_tool(self):
        session = MagicMock()
        with pytest.raises(ValueError, match="Unknown tool"):
            _dispatch(session, "nonexistent_tool", {})


# ── format_answer ─────────────────────────────────────────────────────────────

class TestFormatAnswer:
    def test_returns_llm_text_as_answer(self):
        state = TextToSQLState(
            question="How many docs?",
            sql_result=[{"gpu_family": "Hopper", "count": 2}],
        )
        mock_resp = _text_response("There are 2 Hopper documents.")

        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = format_answer(state)

        assert result.answer == "There are 2 Hopper documents."

    def test_skips_llm_and_returns_error_message_when_error(self):
        state = TextToSQLState(question="q", error="db down")

        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient:
            result = format_answer(state)
            MockClient.return_value.messages.create.assert_not_called()

        assert result.answer == "Error: db down"

    def test_preserves_sql_result(self):
        rows = [{"title": "CUDA Guide", "avg_quality": 0.87, "chunk_count": 42}]
        state = TextToSQLState(question="q", sql_result=rows)
        mock_resp = _text_response("Summary here.")

        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = format_answer(state)

        assert result.sql_result == rows


# ── run (end-to-end with mocks) ───────────────────────────────────────────────

class TestRun:
    def test_run_returns_text_to_sql_state(self):
        plan_resp = _tool_response("list_indexed_documents", {})
        format_resp = _text_response("3 documents are indexed.")
        db_rows = [{"doc_id": "d1", "title": "CUDA Guide", "gpu_family": "Hopper", "doc_type": "guide", "page_count": 120}]

        p_engine, p_sf = _make_db_patches()
        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient, \
             p_engine, p_sf as mock_sf, \
             patch("agents.text_to_sql_agent._dispatch", return_value=db_rows):
            _configure_session(mock_sf)
            # First call returns plan_resp, second returns format_resp
            MockClient.return_value.messages.create.side_effect = [plan_resp, format_resp]
            result = run("How many documents are indexed?", query_id="test0001")

        assert isinstance(result, TextToSQLState)
        assert result.answer == "3 documents are indexed."
        assert result.tool_name == "list_indexed_documents"
        assert result.sql_result == db_rows
        assert result.error is None

    def test_run_propagates_query_id(self):
        plan_resp = _tool_response("avg_quality_per_doc", {})
        format_resp = _text_response("Average quality is 0.85.")

        p_engine, p_sf = _make_db_patches()
        with patch("agents.text_to_sql_agent.anthropic.Anthropic") as MockClient, \
             p_engine, p_sf as mock_sf, \
             patch("agents.text_to_sql_agent._dispatch", return_value=[]):
            _configure_session(mock_sf)
            MockClient.return_value.messages.create.side_effect = [plan_resp, format_resp]
            result = run("What is the average quality?", query_id="myqid001")

        assert result.query_id == "myqid001"
