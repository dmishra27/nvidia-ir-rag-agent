"""Slash command handler: `/nvidia-search <question>` -> FastAPI POST /ask ->
Slack Block Kit blocks (answer + top-3 citations).

Per AGENTS.md's conventions: structlog logging with query_id/stage,
api/schemas' Pydantic v2 AskResponse is reused directly (no shadow schema),
and the HTTP call goes through AskClient so unit tests inject a fake client
instead of opening a real connection -- same mock-everything-external
pattern as every other module (AGENTS.md: "Mock all embedding and LLM calls
in unit tests").

`handle_search_text` is split out from `register()` so the parse -> call ->
format logic is unit-testable without a real slack_bolt App or network call.
"""

from __future__ import annotations

import os

import httpx
import structlog
from dotenv import load_dotenv
from slack_bolt import Ack, App, Respond

from api.schemas import AskResponse

load_dotenv()
log = structlog.get_logger()

DEFAULT_FASTAPI_BASE_URL = "http://localhost:8000"
MAX_CITATIONS_SHOWN = 3
SlackBlock = dict[str, object]


class AskClient:
    """Thin wrapper over FastAPI's POST /ask. `base_url` and `client` are
    both injectable -- tests pass a fake httpx transport instead of hitting
    a live server."""

    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None) -> None:
        self.base_url = base_url or os.environ.get("FASTAPI_BASE_URL", DEFAULT_FASTAPI_BASE_URL)
        self._client = client or httpx.Client(timeout=30.0)

    def ask(self, query: str, top_k: int = 10) -> AskResponse:
        response = self._client.post(f"{self.base_url}/ask", json={"query": query, "top_k": top_k})
        response.raise_for_status()
        return AskResponse.model_validate(response.json())


def format_answer_blocks(result: AskResponse) -> list[SlackBlock]:
    """Answer text, then up to MAX_CITATIONS_SHOWN citations, then a context
    footer carrying query_id -- feedback_handler.py parses that footer back
    out of the message to attribute a 👍/👎 reaction to this query."""
    if result.error:
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": f":warning: *Error:* {result.error}"}},
        ]

    blocks: list[SlackBlock] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Q:* {result.query}\n\n{result.answer or '_No answer produced._'}"},
        }
    ]

    top_citations = result.citations[:MAX_CITATIONS_SHOWN]
    if top_citations:
        citation_lines = [
            f"*{i}.* {c.claim}\n   `{', '.join(c.chunk_ids) or 'no chunk_id'}`"
            for i, c in enumerate(top_citations, start=1)
        ]
        blocks.append({"type": "divider"})
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": "*Top citations:*\n" + "\n".join(citation_lines)}}
        )

    blocks.append(
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"query_id: `{result.query_id}` · react 👍 / 👎 for feedback"}],
        }
    )
    return blocks


def handle_search_text(text: str, ask_client: AskClient) -> list[SlackBlock]:
    """Parse the slash-command text, call FastAPI /ask, return Slack blocks."""
    query = text.strip()
    if not query:
        return [{"type": "section", "text": {"type": "mrkdwn", "text": "Usage: `/nvidia-search <your question>`"}}]

    log.info("slack_search_command", stage="slackbot_handler", query=query)
    try:
        result = ask_client.ask(query)
    except httpx.HTTPError as exc:
        log.error("slack_search_command_failed", stage="slackbot_handler", query=query, exc=str(exc))
        return [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f":warning: Couldn't reach the search service: {exc}"},
            }
        ]

    return format_answer_blocks(result)


def register(app: App, ask_client: AskClient | None = None) -> None:
    """Wire `/nvidia-search` onto a slack_bolt App."""
    client = ask_client or AskClient()

    @app.command("/nvidia-search")
    def _handle(ack: Ack, respond: Respond, command: dict[str, str]) -> None:
        ack()
        blocks = handle_search_text(command.get("text", ""), client)
        respond(blocks=blocks, text="nvidia-ir-rag-agent search result")
