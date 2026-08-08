"""Contract tests for slackbot/feedback_handler.py.

Uses an in-memory SQLite session factory (real SQLAlchemy ORM writes,
per AGENTS.md's "SQLAlchemy ORM for all database writes" -- no raw SQL,
no real Postgres) and a fake Slack WebClient (no real network call).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from schema.models import Base, FeedbackLog
from slackbot.feedback_handler import extract_query_id, record_feedback


def _in_memory_session_factory() -> sessionmaker[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[FeedbackLog.__table__])
    return sessionmaker(bind=engine)


class _FakeWebClient:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        self.calls: list[tuple[str, str]] = []

    def conversations_history(self, *, channel: str, latest: str, inclusive: bool, limit: int) -> dict[str, Any]:
        self.calls.append((channel, latest))
        return {"messages": self._messages}


def test_extract_query_id_finds_footer_pattern() -> None:
    text = "*Q:* something\n\nan answer\n\nquery_id: `abcd1234` · react 👍 / 👎 for feedback"
    assert extract_query_id(text) == "abcd1234"


def test_extract_query_id_returns_none_when_absent() -> None:
    assert extract_query_id("just a plain message, no footer") is None


def test_record_feedback_writes_one_row_via_orm() -> None:
    factory = _in_memory_session_factory()
    feedback_id = record_feedback(factory, query_id="q1", user_id="U123", reaction="up")

    with factory() as session:
        rows = session.query(FeedbackLog).all()
    assert len(rows) == 1
    assert rows[0].feedback_id == feedback_id
    assert rows[0].query_id == "q1"
    assert rows[0].user_id == "U123"
    assert rows[0].reaction == "up"
    assert rows[0].source == "slackbot"


def test_record_feedback_allows_null_query_id_when_unresolved() -> None:
    factory = _in_memory_session_factory()
    record_feedback(factory, query_id=None, user_id="U123", reaction="down")

    with factory() as session:
        row = session.query(FeedbackLog).one()
    assert row.query_id is None
    assert row.reaction == "down"


def test_message_text_joins_context_block_elements() -> None:
    from slackbot.feedback_handler import _message_text

    messages = [
        {
            "text": "fallback text",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "the answer body"}},
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": "query_id: `xyz98765` · react 👍 / 👎 for feedback"}],
                },
            ],
        }
    ]
    client = _FakeWebClient(messages)
    text = _message_text(client, channel="C1", ts="123.456")  # type: ignore[arg-type]
    assert client.calls == [("C1", "123.456")]
    assert extract_query_id(text) == "xyz98765"


def test_message_text_returns_empty_string_when_no_messages() -> None:
    from slackbot.feedback_handler import _message_text

    client = _FakeWebClient([])
    assert _message_text(client, channel="C1", ts="123.456") == ""  # type: ignore[arg-type]
