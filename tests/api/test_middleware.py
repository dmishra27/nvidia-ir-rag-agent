"""Unit tests for api/middleware.py's RequestLatencyMiddleware.

Uses a bare Starlette app (not the full FastAPI app) with an injected fake
session_factory, per AGENTS.md's rule to mock all embedding/LLM calls and
never require a live Postgres connection in unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from api.middleware import RequestLatencyMiddleware


def _configure_session(mock_session_factory: MagicMock) -> MagicMock:
    mock_session = MagicMock()
    mock_session_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_session_factory.return_value.__exit__ = MagicMock(return_value=False)
    return mock_session


def _make_app(session_factory) -> Starlette:
    async def endpoint(request):
        request.state.query_id = "fixed0001"
        request.state.reranker_config = "live_fast"
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/search", endpoint, methods=["GET"])])
    app.add_middleware(RequestLatencyMiddleware, session_factory=session_factory)
    return app


class TestRequestLatencyMiddleware:
    def test_writes_request_log_row_with_expected_fields(self) -> None:
        mock_sf = MagicMock()
        mock_session = _configure_session(mock_sf)
        client = TestClient(_make_app(mock_sf))

        resp = client.get("/search")

        assert resp.status_code == 200
        mock_session.add.assert_called_once()
        row = mock_session.add.call_args[0][0]
        assert row.query_id == "fixed0001"
        assert row.reranker_config == "live_fast"
        assert row.endpoint == "/search"
        assert row.stage == "search"
        assert row.status_code == 200
        assert row.duration_ms >= 0
        mock_session.commit.assert_called_once()

    def test_response_unaffected_when_db_write_fails(self) -> None:
        mock_sf = MagicMock(side_effect=RuntimeError("db down"))
        client = TestClient(_make_app(mock_sf))

        resp = client.get("/search")

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_defaults_stage_to_root_for_bare_path(self) -> None:
        async def endpoint(request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/", endpoint, methods=["GET"])])
        mock_sf = MagicMock()
        mock_session = _configure_session(mock_sf)
        app.add_middleware(RequestLatencyMiddleware, session_factory=mock_sf)
        client = TestClient(app)

        client.get("/")

        row = mock_session.add.call_args[0][0]
        assert row.stage == "root"

    def test_defaults_session_factory_when_none_given(self) -> None:
        with patch("api.middleware.get_engine"), patch(
            "api.middleware.get_session_factory"
        ) as mock_get_sf:
            RequestLatencyMiddleware(app=MagicMock())

        mock_get_sf.assert_called_once()
