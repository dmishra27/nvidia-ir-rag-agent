"""Unit tests for evaluation/citation_judge.py.

Per AGENTS.md ("Mock all embedding and LLM calls in unit tests"), these
tests never call a real Anthropic client — mirrors
tests/agents/test_qa_agent.py's MagicMock tool_use response convention.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agents.qa_agent import Citation, QAState
from evaluation.citation_judge import CitationJudgment, citation_accuracy, judge_claim, judge_qa_output
from retrieval.candidates import Candidate


def _judge_response(supported: bool, rationale: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "judge_citation"
    block.input = {"supported": supported, "rationale": rationale}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _text_only_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# judge_claim() — single (claim, chunk) judgment
# ---------------------------------------------------------------------------


def test_judge_claim_supported() -> None:
    client = MagicMock()
    client.messages.create.return_value = _judge_response(True, "chunk states this directly")

    result = judge_claim(
        query_id="q1", claim="cudaMalloc allocates device memory", chunk_id="c1",
        chunk_text="cudaMalloc allocates memory on the device.", client=client,
    )

    assert result == CitationJudgment(
        query_id="q1", claim="cudaMalloc allocates device memory", chunk_id="c1",
        supported=True, rationale="chunk states this directly",
    )


def test_judge_claim_not_supported() -> None:
    client = MagicMock()
    client.messages.create.return_value = _judge_response(False, "chunk is about something else")

    result = judge_claim(
        query_id="q1", claim="claim text", chunk_id="c1", chunk_text="unrelated text", client=client,
    )

    assert result.supported is False


def test_judge_claim_forces_judge_citation_tool_choice() -> None:
    client = MagicMock()
    client.messages.create.return_value = _judge_response(True, "r")

    judge_claim(query_id="q1", claim="c", chunk_id="c1", chunk_text="t", client=client)

    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "judge_citation"}
    assert kwargs["tools"][0]["name"] == "judge_citation"


def test_judge_claim_returns_none_on_missing_tool_use_block() -> None:
    client = MagicMock()
    client.messages.create.return_value = _text_only_response("no structured output")

    result = judge_claim(query_id="q1", claim="c", chunk_id="c1", chunk_text="t", client=client)

    assert result is None


def test_judge_claim_returns_none_on_client_error() -> None:
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("api error")

    result = judge_claim(query_id="q1", claim="c", chunk_id="c1", chunk_text="t", client=client)

    assert result is None


# ---------------------------------------------------------------------------
# judge_qa_output() — every (claim, cited chunk_id) pair in a QAState
# ---------------------------------------------------------------------------


def _state_with_citations(citations: list[Citation], passages: list[Candidate]) -> QAState:
    return QAState(
        query="q", query_id="q1", answer="answer text", citations=citations, reranked_results=passages
    )


def test_judge_qa_output_judges_every_claim_chunk_pair() -> None:
    state = _state_with_citations(
        citations=[Citation(claim="claim A", chunk_ids=["c1", "c2"])],
        passages=[Candidate(chunk_id="c1", text="text 1", score=1.0, rank=1), Candidate(chunk_id="c2", text="text 2", score=1.0, rank=2)],
    )
    client = MagicMock()
    client.messages.create.return_value = _judge_response(True, "r")

    judgments = judge_qa_output(state, client)

    assert client.messages.create.call_count == 2
    assert [j.chunk_id for j in judgments] == ["c1", "c2"]
    assert all(j.claim == "claim A" for j in judgments)


def test_judge_qa_output_flags_hallucinated_citation_without_calling_llm() -> None:
    state = _state_with_citations(
        citations=[Citation(claim="claim A", chunk_ids=["missing_chunk"])],
        passages=[Candidate(chunk_id="c1", text="text 1", score=1.0, rank=1)],
    )
    client = MagicMock()

    judgments = judge_qa_output(state, client)

    assert client.messages.create.call_count == 0
    assert judgments == [
        CitationJudgment(
            query_id="q1", claim="claim A", chunk_id="missing_chunk", supported=False,
            rationale="cited chunk_id not found in retrieved passages",
        )
    ]


def test_judge_qa_output_falls_back_to_fused_results_when_reranked_empty() -> None:
    state = QAState(
        query="q", query_id="q1", answer="a",
        citations=[Citation(claim="claim A", chunk_ids=["c1"])],
        fused_results=[Candidate(chunk_id="c1", text="fused text", score=1.0, rank=1)],
        reranked_results=[],
    )
    client = MagicMock()
    client.messages.create.return_value = _judge_response(True, "r")

    judge_qa_output(state, client)

    _, kwargs = client.messages.create.call_args
    assert "fused text" in kwargs["messages"][0]["content"]


def test_judge_qa_output_no_citations_returns_empty() -> None:
    state = _state_with_citations(citations=[], passages=[])
    client = MagicMock()

    assert judge_qa_output(state, client) == []
    assert client.messages.create.call_count == 0


# ---------------------------------------------------------------------------
# citation_accuracy() — aggregate fraction supported
# ---------------------------------------------------------------------------


def test_citation_accuracy_all_supported_is_one() -> None:
    judgments = [
        CitationJudgment(query_id="q1", claim="a", chunk_id="c1", supported=True, rationale="r"),
        CitationJudgment(query_id="q1", claim="b", chunk_id="c2", supported=True, rationale="r"),
    ]

    assert citation_accuracy(judgments) == 1.0


def test_citation_accuracy_partial() -> None:
    judgments = [
        CitationJudgment(query_id="q1", claim="a", chunk_id="c1", supported=True, rationale="r"),
        CitationJudgment(query_id="q1", claim="b", chunk_id="c2", supported=False, rationale="r"),
    ]

    assert citation_accuracy(judgments) == 0.5


def test_citation_accuracy_empty_is_zero() -> None:
    assert citation_accuracy([]) == 0.0
