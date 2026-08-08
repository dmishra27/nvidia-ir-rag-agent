"""HITL feedback: 👍/👎 (`thumbsup`/`thumbsdown`) reactions on the bot's own
`/nvidia-search` answers -> `feedback_log` (Layer 8 per AGENTS.md).

Slack reactions attach to a *message*, not to our query_id, so this looks
the query_id back up from the reacted-to message's own text: every answer
handlers.py posts carries a `query_id: \\`abcd1234\\`` context footer, and
`extract_query_id` parses that back out via `conversations.history`. Writes
go through the SQLAlchemy ORM only (schema/models.py's FeedbackLog) --
AGENTS.md's "Never raw SQL strings."
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import structlog
from slack_bolt import App
from slack_sdk import WebClient
from sqlalchemy.orm import Session, sessionmaker

from schema.models import FeedbackLog, get_engine, get_session_factory

log = structlog.get_logger()

THUMBS_UP = "thumbsup"
THUMBS_DOWN = "thumbsdown"
# Slack reaction name -> feedback_log.reaction value.
TRACKED_REACTIONS = {THUMBS_UP: "up", THUMBS_DOWN: "down"}

_QUERY_ID_PATTERN = re.compile(r"query_id:\s*`([A-Za-z0-9]+)`")


def extract_query_id(message_text: str) -> str | None:
    """Pull the query_id back out of handlers.py's context-block footer."""
    match = _QUERY_ID_PATTERN.search(message_text)
    return match.group(1) if match else None


def record_feedback(
    session_factory: sessionmaker[Session],
    query_id: str | None,
    user_id: str,
    reaction: str,
    source: str = "slackbot",
) -> str:
    """Write one feedback_log row via the ORM. Returns the new feedback_id."""
    feedback_id = str(uuid.uuid4())
    with session_factory() as session:
        session.add(
            FeedbackLog(
                feedback_id=feedback_id,
                query_id=query_id,
                user_id=user_id,
                reaction=reaction,
                source=source,
            )
        )
        session.commit()
    log.info(
        "feedback_recorded",
        stage="feedback_handler",
        query_id=query_id,
        user_id=user_id,
        reaction=reaction,
        feedback_id=feedback_id,
    )
    return feedback_id


def _message_text(client: WebClient, channel: str, ts: str) -> str:
    """Join every text field of the reacted-to message (top-level `text` plus
    each block's own text/context elements) so the query_id footer is found
    regardless of which block it landed in."""
    response = client.conversations_history(channel=channel, latest=ts, inclusive=True, limit=1)
    messages: list[dict[str, Any]] = response.get("messages", [])
    if not messages:
        return ""
    message = messages[0]
    texts = [message.get("text", "")]
    for block in message.get("blocks", []):
        block_text = block.get("text", {}).get("text")
        if block_text:
            texts.append(block_text)
        for element in block.get("elements", []):
            element_text = element.get("text")
            if element_text:
                texts.append(element_text)
    return "\n".join(texts)


def register(app: App, session_factory: sessionmaker[Session] | None = None) -> None:
    """Wire `reaction_added` (thumbsup/thumbsdown only) onto a slack_bolt App."""
    factory = session_factory or get_session_factory(get_engine())

    @app.event("reaction_added")
    def _handle_reaction(event: dict[str, Any], client: WebClient) -> None:
        reaction = event.get("reaction", "")
        if reaction not in TRACKED_REACTIONS:
            return
        item = event.get("item", {})
        if item.get("type") != "message":
            return

        text = _message_text(client, item["channel"], item["ts"])
        query_id = extract_query_id(text)
        record_feedback(
            factory,
            query_id=query_id,
            user_id=event.get("user", "unknown"),
            reaction=TRACKED_REACTIONS[reaction],
        )
