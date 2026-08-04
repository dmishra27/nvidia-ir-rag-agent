"""Contract tests for agents/eval_agent.py.

Per AGENTS.md ("LangGraph nodes: contract tests on state schema only"), these
tests verify how the evaluate node transforms EvalState using a mocked
Anthropic client — mirrors tests/evaluation/test_citation_judge.py's
MagicMock tool_use response convention. Never a real LLM call.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agents.eval_agent import EvalState, make_evaluate_node
from agents.qa_agent import Citation
from retrieval.candidates import Candidate


def _c(chunk_id: str, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=f"text {chunk_id}", score=1.0, rank=rank)


def _judge_response(supported: bool, rationale: str = "r") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "judge_citation"
    block.input = {"supported": supported, "rationale": rationale}
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_eval_state_defaults() -> None:
    state = EvalState(query="q")

    assert state.citation_judgments == []
    assert state.citation_accuracy is None


def test_evaluate_populates_citation_judgments_and_accuracy() -> None:
    client = MagicMock()
    client.messages.create.return_value = _judge_response(True)
    node = make_evaluate_node(client)
    state = EvalState(
        query="q",
        answer="a",
        citations=[Citation(claim="claim A", chunk_ids=["c1"])],
        reranked_results=[_c("c1", 1)],
    )

    result = node(state)

    assert len(result.citation_judgments) == 1
    assert result.citation_judgments[0].supported is True
    assert result.citation_accuracy == 1.0


def test_evaluate_computes_partial_accuracy() -> None:
    client = MagicMock()
    client.messages.create.side_effect = [_judge_response(True), _judge_response(False)]
    node = make_evaluate_node(client)
    state = EvalState(
        query="q",
        answer="a",
        citations=[Citation(claim="claim A", chunk_ids=["c1", "c2"])],
        reranked_results=[_c("c1", 1), _c("c2", 2)],
    )

    result = node(state)

    assert result.citation_accuracy == 0.5


def test_evaluate_no_citations_yields_zero_accuracy() -> None:
    client = MagicMock()
    node = make_evaluate_node(client)
    state = EvalState(query="q", answer="a", citations=[])

    result = node(state)

    assert client.messages.create.call_count == 0
    assert result.citation_judgments == []
    assert result.citation_accuracy == 0.0


def test_evaluate_skips_and_preserves_state_when_error_already_set() -> None:
    client = MagicMock()
    node = make_evaluate_node(client)
    state = EvalState(query="q", error="upstream failure")

    result = node(state)

    assert client.messages.create.call_count == 0
    assert result.error == "upstream failure"
    assert result.citation_judgments == []
    assert result.citation_accuracy is None


def test_evaluate_preserves_query_id_and_answer() -> None:
    client = MagicMock()
    client.messages.create.return_value = _judge_response(True)
    node = make_evaluate_node(client)
    state = EvalState(
        query="q",
        query_id="fixed0001",
        answer="the answer",
        citations=[Citation(claim="claim A", chunk_ids=["c1"])],
        reranked_results=[_c("c1", 1)],
    )

    result = node(state)

    assert result.query_id == "fixed0001"
    assert result.answer == "the answer"
