"""Contract tests for agents/qa_agent.py.

Per AGENTS.md ("LangGraph nodes: contract tests on state schema only"),
these tests verify how each node transforms the QAState schema using fake
BM25/dense/router objects and a mocked Anthropic client — never a real
index, Qdrant connection, cross-encoder, or live API call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.qa_agent import (
    Citation,
    QAState,
    build_graph,
    make_generate_node,
    make_rerank_node,
    make_retrieve_node,
    run,
)
from retrieval.candidates import Candidate


def _c(chunk_id: str, rank: int) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=f"text {chunk_id}", score=1.0, rank=rank)


class _FakeBM25:
    def __init__(self, results: list[Candidate], calls: list) -> None:
        self._results = results
        self.calls = calls

    def search(self, query: str, top_k: int) -> list[Candidate]:
        self.calls.append(("bm25", query, top_k))
        return self._results


class _RaisingBM25:
    def search(self, query: str, top_k: int) -> list[Candidate]:
        raise RuntimeError("bm25 index unavailable")


class _FakeDense:
    def __init__(self, results: list[Candidate], calls: list) -> None:
        self._results = results
        self.calls = calls

    def search(self, query: str, top_k: int) -> list[Candidate]:
        self.calls.append(("dense", query, top_k))
        return self._results


class _FakeRouter:
    def __init__(self, results: list[Candidate], calls: list) -> None:
        self._results = results
        self.calls = calls

    def rerank(self, query: str, candidates: list[Candidate], top_k: int, query_id: str) -> list[Candidate]:
        self.calls.append(("rerank", query, candidates, top_k, query_id))
        return self._results


class _RaisingRouter:
    def rerank(self, query: str, candidates: list[Candidate], top_k: int, query_id: str) -> list[Candidate]:
        raise RuntimeError("reranker misconfigured")


def _tool_response(answer: str, citations: list[dict]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "answer_with_citations"
    block.input = {"answer": answer, "citations": citations}
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
# QAState — defaults
# ---------------------------------------------------------------------------


def test_qa_state_defaults() -> None:
    state = QAState(query="cudaDeviceSynchronize return value")

    assert state.top_k == 10
    assert state.candidate_pool_size == 100
    assert state.fused_results == []
    assert state.reranked_results == []
    assert state.answer is None
    assert state.citations == []
    assert state.error is None
    assert isinstance(state.query_id, str) and state.query_id


def test_qa_state_query_id_is_overridable() -> None:
    state = QAState(query="q", query_id="fixed0001")

    assert state.query_id == "fixed0001"


# ---------------------------------------------------------------------------
# retrieve node
# ---------------------------------------------------------------------------


class TestRetrieveNode:
    def test_populates_fused_results(self) -> None:
        calls: list = []
        bm25_results = [_c("b1", 1)]
        dense_results = [_c("d1", 1)]
        node = make_retrieve_node(_FakeBM25(bm25_results, calls), _FakeDense(dense_results, calls))
        state = QAState(query="cudaMalloc parameters")

        result = node(state)

        assert {r.chunk_id for r in result.fused_results} == {"b1", "d1"}

    def test_preserves_query_id_and_query(self) -> None:
        node = make_retrieve_node(_FakeBM25([], []), _FakeDense([], []))
        state = QAState(query="cudaMalloc parameters", query_id="fixed0001")

        result = node(state)

        assert result.query_id == "fixed0001"
        assert result.query == "cudaMalloc parameters"

    def test_sets_error_when_bm25_raises(self) -> None:
        node = make_retrieve_node(_RaisingBM25(), _FakeDense([], []))
        state = QAState(query="q")

        result = node(state)

        assert result.error == "bm25 index unavailable"
        assert result.fused_results == []


# ---------------------------------------------------------------------------
# rerank node
# ---------------------------------------------------------------------------


class TestRerankNode:
    def test_populates_reranked_results_from_router(self) -> None:
        reranked = [_c("r1", 1)]
        calls: list = []
        node = make_rerank_node(_FakeRouter(reranked, calls))
        state = QAState(query="q", fused_results=[_c("f1", 1)])

        result = node(state)

        assert result.reranked_results == reranked

    def test_skips_router_call_and_preserves_state_when_error_already_set(self) -> None:
        calls: list = []
        node = make_rerank_node(_FakeRouter([_c("r1", 1)], calls))
        state = QAState(query="q", error="upstream failure")

        result = node(state)

        assert calls == []
        assert result.error == "upstream failure"
        assert result.reranked_results == []

    def test_sets_error_when_router_raises(self) -> None:
        node = make_rerank_node(_RaisingRouter())
        state = QAState(query="q", fused_results=[_c("f1", 1)])

        result = node(state)

        assert result.error == "reranker misconfigured"
        assert result.reranked_results == []


# ---------------------------------------------------------------------------
# generate node
# ---------------------------------------------------------------------------


class TestGenerateNode:
    def test_skips_llm_and_preserves_state_when_error_already_set(self) -> None:
        node = make_generate_node()
        state = QAState(query="q", error="upstream failure")

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            result = node(state)
            MockClient.return_value.messages.create.assert_not_called()

        assert result.error == "upstream failure"
        assert result.answer is None

    def test_returns_empty_answer_when_no_passages(self) -> None:
        node = make_generate_node()
        state = QAState(query="q")

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            result = node(state)
            MockClient.return_value.messages.create.assert_not_called()

        assert result.answer == ""
        assert result.citations == []

    def test_falls_back_to_fused_results_when_reranked_empty(self) -> None:
        node = make_generate_node()
        state = QAState(query="q", fused_results=[_c("f1", 1)])
        mock_resp = _tool_response("An answer.", [{"claim": "An answer.", "chunk_ids": ["f1"]}])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = node(state)

        assert result.answer == "An answer."

    def test_sets_answer_and_citations_from_tool_call(self) -> None:
        node = make_generate_node()
        state = QAState(
            query="cudaDeviceSynchronize return value",
            reranked_results=[_c("r1", 1)],
        )
        mock_resp = _tool_response(
            "cudaDeviceSynchronize returns cudaSuccess or an error code.",
            [
                {
                    "claim": "cudaDeviceSynchronize returns cudaSuccess or an error code.",
                    "chunk_ids": ["r1"],
                }
            ],
        )

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = node(state)

        assert result.answer == "cudaDeviceSynchronize returns cudaSuccess or an error code."
        assert result.citations == [
            Citation(
                claim="cudaDeviceSynchronize returns cudaSuccess or an error code.",
                chunk_ids=["r1"],
            )
        ]

    def test_forces_answer_with_citations_tool_choice(self) -> None:
        node = make_generate_node()
        state = QAState(query="q", reranked_results=[_c("r1", 1)])
        mock_resp = _tool_response("ans", [{"claim": "ans", "chunk_ids": ["r1"]}])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            node(state)
            _, kwargs = MockClient.return_value.messages.create.call_args
            assert kwargs.get("tool_choice") == {"type": "tool", "name": "answer_with_citations"}

    def test_truncates_passages_to_top_k(self) -> None:
        node = make_generate_node()
        passages = [_c(f"r{i}", i) for i in range(1, 6)]
        state = QAState(query="q", reranked_results=passages, top_k=2)
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            node(state)
            _, kwargs = MockClient.return_value.messages.create.call_args
            content = kwargs["messages"][0]["content"]
            assert "r1" in content and "r2" in content
            assert "r3" not in content

    def test_sets_error_when_no_tool_use_block(self) -> None:
        node = make_generate_node()
        state = QAState(query="q", reranked_results=[_c("r1", 1)])
        mock_resp = _text_only_response("plain text, no tool call")

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = node(state)

        assert result.error is not None
        assert result.answer is None

    def test_sets_error_when_anthropic_raises(self) -> None:
        node = make_generate_node()
        state = QAState(query="q", reranked_results=[_c("r1", 1)])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = RuntimeError("api down")
            result = node(state)

        assert result.error == "api down"

    def test_preserves_query_id(self) -> None:
        node = make_generate_node()
        state = QAState(query="q", query_id="fixed0001", reranked_results=[_c("r1", 1)])
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = node(state)

        assert result.query_id == "fixed0001"


# ---------------------------------------------------------------------------
# build_graph / run — end-to-end schema contract
# ---------------------------------------------------------------------------


class TestRunEndToEnd:
    def test_run_returns_qa_state_with_answer_and_citations(self) -> None:
        calls: list = []
        reranked = [_c("r1", 1)]
        mock_resp = _tool_response("Final answer.", [{"claim": "Final answer.", "chunk_ids": ["r1"]}])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = run(
                "cudaMalloc parameters",
                bm25_index=_FakeBM25([_c("b1", 1)], calls),
                dense_index=_FakeDense([_c("d1", 1)], calls),
                router=_FakeRouter(reranked, calls),
            )

        assert isinstance(result, QAState)
        assert result.answer == "Final answer."
        assert result.citations == [Citation(claim="Final answer.", chunk_ids=["r1"])]
        assert result.error is None

    def test_run_calls_signals_in_retrieve_rerank_order(self) -> None:
        calls: list = []
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            run(
                "q",
                bm25_index=_FakeBM25([], calls),
                dense_index=_FakeDense([], calls),
                router=_FakeRouter([], calls),
            )

        call_kinds = [c[0] for c in calls]
        assert call_kinds.index("bm25") < call_kinds.index("rerank")
        assert call_kinds.index("dense") < call_kinds.index("rerank")

    def test_run_propagates_custom_query_id(self) -> None:
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = run(
                "q",
                query_id="fixed0001",
                bm25_index=_FakeBM25([], []),
                dense_index=_FakeDense([], []),
                router=_FakeRouter([], []),
            )

        assert result.query_id == "fixed0001"

    def test_run_propagates_retrieve_error_without_calling_llm(self) -> None:
        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            result = run(
                "q",
                bm25_index=_RaisingBM25(),
                dense_index=_FakeDense([], []),
                router=_FakeRouter([], []),
            )
            MockClient.return_value.messages.create.assert_not_called()

        assert result.error == "bm25 index unavailable"
        assert result.answer is None

    def test_build_graph_compiles_with_fakes(self) -> None:
        graph = build_graph(_FakeBM25([], []), _FakeDense([], []), _FakeRouter([], []))
        state = QAState(query="q")

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = _tool_response("ans", [])
            result = graph.invoke(state)

        final = QAState(**result) if isinstance(result, dict) else result
        assert final.answer == ""
        assert final.citations == []
