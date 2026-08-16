"""Endpoint tests for /feedback.

Mocks api.routers.feedback.get_feedback_session_factory via
app.dependency_overrides -- per AGENTS.md's rule to never require a live
Postgres connection in unit tests. The injected session_factory is a
MagicMock configured as a context manager, mirroring
tests/api/test_middleware.py's _configure_session helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.main import create_app
from api.routers.feedback import get_feedback_session_factory
from schema.models import QueryLog


def _configure_session(mock_session_factory: MagicMock, *, query_log_row_exists: bool = True) -> MagicMock:
    mock_session = MagicMock()
    mock_session_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_session_factory.return_value.__exit__ = MagicMock(return_value=False)
    # session.get(QueryLog, query_id) -- explicit per test rather than
    # relying on MagicMock()'s default truthy return value, since the route
    # branches on `is None`. True means "a matching query_log row exists".
    mock_session.get.return_value = MagicMock() if query_log_row_exists else None
    return mock_session


def _make_client(mock_session_factory: MagicMock) -> TestClient:
    app = create_app(session_factory=MagicMock())
    app.dependency_overrides[get_feedback_session_factory] = lambda: mock_session_factory
    return TestClient(app)


class TestFeedback:
    def test_writes_feedback_log_row_with_source_web(self) -> None:
        mock_sf = MagicMock()
        mock_session = _configure_session(mock_sf)
        client = _make_client(mock_sf)

        resp = client.post(
            "/feedback",
            json={"query_id": "q1", "chunk_id": "c1", "reaction": "up"},
        )

        assert resp.status_code == 200
        mock_session.get.assert_called_once_with(QueryLog, "q1")
        mock_session.add.assert_called_once()
        row = mock_session.add.call_args[0][0]
        assert row.query_id == "q1"  # a matching query_log row exists -> kept
        assert row.reaction == "up"
        assert row.source == "web"  # not the schema default "slackbot"
        assert row.user_id == "anonymous"  # no user_id sent -> defaulted
        mock_session.commit.assert_called_once()
        assert resp.json()["feedback_id"]

    def test_nulls_out_query_id_when_no_matching_query_log_row_exists(self) -> None:
        # The gap this closes: FeedbackLog.query_id is a nullable FK to
        # query_log.query_id, but nothing in the live /search or /ask path
        # writes a query_log row -- so a real Postgres would reject this
        # insert with a foreign-key violation if query_id were written
        # through unchecked. Storing None instead is what makes the write
        # succeed regardless.
        mock_sf = MagicMock()
        mock_session = _configure_session(mock_sf, query_log_row_exists=False)
        client = _make_client(mock_sf)

        resp = client.post(
            "/feedback",
            json={"query_id": "orphan-id", "reaction": "up"},
        )

        assert resp.status_code == 200
        mock_session.get.assert_called_once_with(QueryLog, "orphan-id")
        row = mock_session.add.call_args[0][0]
        assert row.query_id is None
        mock_session.commit.assert_called_once()

    def test_allows_a_null_query_id_without_looking_up_a_parent(self) -> None:
        mock_sf = MagicMock()
        mock_session = _configure_session(mock_sf)
        client = _make_client(mock_sf)

        resp = client.post("/feedback", json={"reaction": "down"})

        assert resp.status_code == 200
        mock_session.get.assert_not_called()  # nothing to look up for a None query_id
        row = mock_session.add.call_args[0][0]
        assert row.query_id is None

    def test_accepts_an_explicit_user_id(self) -> None:
        mock_sf = MagicMock()
        mock_session = _configure_session(mock_sf)
        client = _make_client(mock_sf)

        resp = client.post(
            "/feedback",
            json={"query_id": "q1", "reaction": "down", "user_id": "browser-abc123"},
        )

        assert resp.status_code == 200
        row = mock_session.add.call_args[0][0]
        assert row.user_id == "browser-abc123"
        assert row.reaction == "down"

    def test_rejects_a_reaction_outside_up_or_down(self) -> None:
        mock_sf = MagicMock()
        _configure_session(mock_sf)
        client = _make_client(mock_sf)

        resp = client.post("/feedback", json={"query_id": "q1", "reaction": "sideways"})

        assert resp.status_code == 422

    def test_returns_500_when_the_db_write_fails(self) -> None:
        mock_sf = MagicMock(side_effect=RuntimeError("db down"))
        client = _make_client(mock_sf)

        resp = client.post("/feedback", json={"query_id": "q1", "reaction": "up"})

        assert resp.status_code == 500

    def test_feedback_route_is_included_in_the_openapi_schema(self) -> None:
        client = _make_client(MagicMock())

        schema = client.get("/openapi.json").json()

        assert "/feedback" in schema["paths"]
