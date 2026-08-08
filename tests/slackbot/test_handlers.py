"""Contract tests for slackbot/handlers.py.

Per AGENTS.md's mock-everything-external convention, AskClient.ask is faked
here (no real httpx call, no real slack_bolt App) -- these test the parse ->
call -> Slack-blocks pipeline in isolation.
"""

from __future__ import annotations

import httpx
import pytest

from api.schemas import AskResponse, CitationOut
from slackbot.handlers import format_answer_blocks, handle_search_text


class _FakeAskClient:
    def __init__(self, result: AskResponse | None = None, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[str] = []

    def ask(self, query: str, top_k: int = 10) -> AskResponse:
        self.calls.append(query)
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result


def _ask_response(**overrides: object) -> AskResponse:
    data: dict[str, object] = dict(
        query_id="q1234567",
        query="How does cudaMalloc work?",
        reranker_mode="live_fast",
        answer="cudaMalloc allocates device memory [c_1].",
        citations=[CitationOut(claim="cudaMalloc allocates device memory.", chunk_ids=["c_1"])],
        error=None,
    )
    data.update(overrides)
    return AskResponse.model_validate(data)


def test_handle_search_text_empty_query_returns_usage_hint() -> None:
    client = _FakeAskClient()
    blocks = handle_search_text("   ", client)
    assert client.calls == []
    assert "Usage:" in str(blocks[0])


def test_handle_search_text_calls_ask_client_with_stripped_query() -> None:
    client = _FakeAskClient(result=_ask_response())
    handle_search_text("  How does cudaMalloc work?  ", client)
    assert client.calls == ["How does cudaMalloc work?"]


def test_handle_search_text_returns_answer_and_citation_blocks() -> None:
    client = _FakeAskClient(result=_ask_response())
    blocks = handle_search_text("How does cudaMalloc work?", client)
    joined = str(blocks)
    assert "cudaMalloc allocates device memory" in joined
    assert "c_1" in joined
    assert "q1234567" in joined


def test_handle_search_text_surfaces_http_errors_without_raising() -> None:
    client = _FakeAskClient(raises=httpx.ConnectError("connection refused"))
    blocks = handle_search_text("How does cudaMalloc work?", client)
    assert "Couldn't reach the search service" in str(blocks)


def test_format_answer_blocks_shows_error_instead_of_answer() -> None:
    result = _ask_response(answer=None, citations=[], error="index unavailable")
    blocks = format_answer_blocks(result)
    assert len(blocks) == 1
    assert "index unavailable" in str(blocks[0])


def test_format_answer_blocks_caps_citations_at_three() -> None:
    citations = [CitationOut(claim=f"claim {i}", chunk_ids=[f"c_{i}"]) for i in range(5)]
    result = _ask_response(citations=citations)
    blocks = format_answer_blocks(result)
    joined = str(blocks)
    assert "claim 0" in joined and "claim 2" in joined
    assert "claim 3" not in joined and "claim 4" not in joined


def test_format_answer_blocks_footer_carries_query_id_for_feedback_lookup() -> None:
    result = _ask_response(query_id="abcd1234")
    blocks = format_answer_blocks(result)
    footer = blocks[-1]
    assert footer["type"] == "context"
    assert "query_id: `abcd1234`" in str(footer)


@pytest.mark.parametrize("chunk_ids", [[], ["c1", "c2"]])
def test_format_answer_blocks_handles_citations_with_and_without_chunk_ids(chunk_ids: list[str]) -> None:
    result = _ask_response(citations=[CitationOut(claim="a claim", chunk_ids=chunk_ids)])
    blocks = format_answer_blocks(result)
    assert "a claim" in str(blocks)
