"""Contract tests for slackbot/handlers.py.

Per AGENTS.md's mock-everything-external convention, AskClient.ask is faked
in most tests here (no real httpx call, no real slack_bolt App) -- these
test the parse -> call -> Slack-blocks pipeline in isolation.

The TestAskClientHttpHandling tests are the exception: they drive the real
AskClient over an in-process transport (a real httpx.Response, real status
handling) so that AskClient's own logic -- specifically how it treats a
503 from /ask -- is actually exercised. The fake client cannot catch a
regression there because it never runs that code.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_bm25_index, get_dense_index, get_msmarco_reranker
from api.main import create_app
from api.schemas import AskResponse, CitationOut
from slackbot.handlers import AskClient, format_answer_blocks, handle_search_text


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


class _RaisingBM25:
    def search(self, query: str, top_k: int) -> list:
        raise RuntimeError("bm25 index unavailable")


class TestAskClientHttpHandling:
    """Drive the real AskClient over an in-process transport, not the fake."""

    def test_503_from_ask_surfaces_the_error_body_not_a_generic_outage(self) -> None:
        """Regression: api/routers/ask.py returns HTTP 503 with
        {"error": ...} when retrieval fails outright. AskClient must parse
        that body so Slack shows the real reason; before this fix
        raise_for_status() turned every 503 into "Couldn't reach the search
        service", hiding it. The fake client can't cover this -- it never
        makes an HTTP call."""
        app = create_app(session_factory=MagicMock())
        app.dependency_overrides[get_bm25_index] = lambda: _RaisingBM25()
        app.dependency_overrides[get_dense_index] = lambda: None
        app.dependency_overrides[get_msmarco_reranker] = lambda: None

        ask_client = AskClient(base_url="http://testserver", client=TestClient(app))
        blocks = handle_search_text("How does cudaMalloc work?", ask_client)

        joined = str(blocks)
        assert "bm25 index unavailable" in joined
        assert "Couldn't reach the search service" not in joined

    def test_5xx_without_an_error_field_still_raises_and_is_handled_as_outage(self) -> None:
        """A 5xx whose body is not a valid AskResponse (a bare 500, a proxy
        error page) has no `error` to show -- it must still raise so
        handle_search_text reports an outage rather than crashing on
        model_validate."""

        def _bare_500(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "Internal Server Error"})

        http = httpx.Client(transport=httpx.MockTransport(_bare_500), base_url="http://testserver")
        ask_client = AskClient(base_url="http://testserver", client=http)

        blocks = handle_search_text("q", ask_client)

        assert "Couldn't reach the search service" in str(blocks)

    def test_200_success_over_real_transport_still_parses(self) -> None:
        def _ok(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "query_id": "abc12345",
                    "query": "q",
                    "reranker_mode": "live_fast",
                    "answer": "real answer [c_1]",
                    "citations": [{"claim": "real answer.", "chunk_ids": ["c_1"]}],
                    "error": None,
                },
            )

        http = httpx.Client(transport=httpx.MockTransport(_ok), base_url="http://testserver")
        ask_client = AskClient(base_url="http://testserver", client=http)

        blocks = handle_search_text("q", ask_client)

        assert "real answer" in str(blocks)


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
