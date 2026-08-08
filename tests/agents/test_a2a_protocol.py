"""Contract tests for agents/a2a_protocol.py.

Per AGENTS.md ("LangGraph nodes: contract tests on state schema only" /
"Mock all embedding and LLM calls in unit tests"), these verify the
handoff envelope's construction/consumption and `run_handoff`'s end-to-end
wiring using fake BM25/dense/router objects and a mocked Anthropic client
-- never a real index, Qdrant connection, cross-encoder, or live API call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.a2a_protocol import (
    AgentMessage,
    ErrorPayload,
    RetrievalHandoffPayload,
    apply_retrieval_handoff,
    build_error_handoff,
    build_retrieval_handoff,
    run_handoff,
)
from agents.qa_agent import QAState
from agents.retrieval_agent import AgentState
from retrieval.candidates import Candidate


def _c(chunk_id: str, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=f"text {chunk_id}", score=1.0, rank=rank)


class _FakeBM25:
    def __init__(self, results: list[Candidate]) -> None:
        self._results = results

    def search(self, query: str, top_k: int) -> list[Candidate]:
        return self._results


class _FakeDense:
    def __init__(self, results: list[Candidate]) -> None:
        self._results = results

    def search(self, query: str, top_k: int) -> list[Candidate]:
        return self._results


class _FakeRouter:
    def __init__(self, results: list[Candidate]) -> None:
        self._results = results

    def rerank(self, query: str, candidates: list[Candidate], top_k: int, query_id: str) -> list[Candidate]:
        return self._results


class _RaisingBM25:
    def search(self, query: str, top_k: int) -> list[Candidate]:
        raise RuntimeError("bm25 index unavailable")


def _tool_response(answer: str, citations: list[dict]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "answer_with_citations"
    block.input = {"answer": answer, "citations": citations}
    resp = MagicMock()
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# build_retrieval_handoff / build_error_handoff
# ---------------------------------------------------------------------------


def test_build_retrieval_handoff_packages_results_and_metadata() -> None:
    state = AgentState(
        query_id="q1", query="cudaMalloc parameters", top_k=5, results=[_c("r1", 1), _c("r2", 2)]
    )

    message = build_retrieval_handoff(state)

    assert message.query_id == "q1"
    assert message.sender == "retrieval_agent"
    assert message.recipient == "qa_agent"
    assert message.message_type == "retrieval_handoff"
    assert isinstance(message.payload, RetrievalHandoffPayload)
    assert message.payload.query == "cudaMalloc parameters"
    assert message.payload.top_k == 5
    assert message.payload.results == [_c("r1", 1), _c("r2", 2)]


def test_build_retrieval_handoff_returns_error_message_when_state_errored() -> None:
    state = AgentState(query_id="q1", query="q", error="qdrant unreachable")

    message = build_retrieval_handoff(state)

    assert message.message_type == "error"
    assert isinstance(message.payload, ErrorPayload)
    assert message.payload.message == "qdrant unreachable"


def test_build_retrieval_handoff_respects_custom_recipient() -> None:
    state = AgentState(query_id="q1", query="q", results=[])
    message = build_retrieval_handoff(state, recipient="eval_agent")
    assert message.recipient == "eval_agent"


def test_build_error_handoff_shape() -> None:
    message = build_error_handoff("q1", sender="a", recipient="b", message="boom")
    assert message.message_type == "error"
    assert message.sender == "a"
    assert message.recipient == "b"
    assert isinstance(message.payload, ErrorPayload)
    assert message.payload.message == "boom"


def test_agent_message_has_a_message_id_and_timestamp() -> None:
    message = build_error_handoff("q1", sender="a", recipient="b", message="boom")
    assert message.message_id
    assert message.created_at is not None


# ---------------------------------------------------------------------------
# apply_retrieval_handoff
# ---------------------------------------------------------------------------


def test_apply_retrieval_handoff_populates_reranked_results() -> None:
    state = AgentState(query_id="q1", query="cudaMalloc parameters", top_k=3, results=[_c("r1", 1)])
    message = build_retrieval_handoff(state)

    qa_state = apply_retrieval_handoff(message)

    assert isinstance(qa_state, QAState)
    assert qa_state.query_id == "q1"
    assert qa_state.query == "cudaMalloc parameters"
    assert qa_state.top_k == 3
    assert qa_state.reranked_results == [_c("r1", 1)]
    assert qa_state.error is None


def test_apply_retrieval_handoff_sets_error_on_error_message() -> None:
    message = build_error_handoff("q1", sender="retrieval_agent", recipient="qa_agent", message="index unavailable")

    qa_state = apply_retrieval_handoff(message)

    assert qa_state.error == "index unavailable"
    assert qa_state.reranked_results == []


def test_apply_retrieval_handoff_merges_into_an_existing_qa_state() -> None:
    existing = QAState(query_id="q1", query="stale query", top_k=99)
    state = AgentState(query_id="q1", query="fresh query", top_k=7, results=[_c("r1", 1)])
    message = build_retrieval_handoff(state)

    qa_state = apply_retrieval_handoff(message, qa_state=existing)

    assert qa_state.query == "fresh query"
    assert qa_state.top_k == 7
    assert qa_state.reranked_results == [_c("r1", 1)]


def test_apply_retrieval_handoff_rejects_mismatched_payload_type() -> None:
    # Constructing an inconsistent envelope directly (bypassing the builders)
    # to exercise apply_retrieval_handoff's own defensive check.
    message = AgentMessage(
        query_id="q1",
        sender="retrieval_agent",
        recipient="qa_agent",
        message_type="retrieval_handoff",
        payload=ErrorPayload(message="wrong payload for this message_type"),
    )
    try:
        apply_retrieval_handoff(message)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# run_handoff — end to end, fakes + mocked Anthropic client
# ---------------------------------------------------------------------------


def test_run_handoff_generates_an_answer_from_handed_off_results() -> None:
    bm25 = _FakeBM25([_c("f1", 1)])
    dense = _FakeDense([_c("f1", 1)])
    router = _FakeRouter([_c("r1", 1)])
    mock_resp = _tool_response("cudaMalloc allocates device memory.", [{"claim": "cudaMalloc allocates device memory.", "chunk_ids": ["r1"]}])

    with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_resp
        result = run_handoff(
            "cudaMalloc parameters", query_id="q1", bm25_index=bm25, dense_index=dense, router=router
        )

    assert result.answer == "cudaMalloc allocates device memory."
    assert result.citations[0].chunk_ids == ["r1"]
    assert result.error is None


def test_run_handoff_short_circuits_to_error_without_calling_the_llm() -> None:
    with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
        result = run_handoff(
            "cudaMalloc parameters",
            query_id="q1",
            bm25_index=_RaisingBM25(),
            dense_index=_FakeDense([]),
            router=_FakeRouter([]),
        )
        MockClient.return_value.messages.create.assert_not_called()

    assert result.error is not None
    assert "bm25 index unavailable" in result.error
    assert result.answer is None
