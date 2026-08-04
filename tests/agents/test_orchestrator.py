"""Contract tests for agents/orchestrator.py.

Per AGENTS.md ("LangGraph nodes: contract tests on state schema only"), these
tests verify the composed retrieve -> rerank -> generate -> evaluate graph
transforms EvalState correctly, using fake BM25/dense/router objects, a
patched Anthropic client for the reused qa_agent.generate node (mirrors
tests/agents/test_qa_agent.py), and an injected MagicMock client for the
eval_agent.evaluate node (mirrors tests/agents/test_eval_agent.py). Never a
real index, Qdrant connection, cross-encoder, or live API call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.eval_agent import EvalState
from agents.orchestrator import build_graph, run
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


def _tool_response(answer: str, citations: list[dict]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "answer_with_citations"
    block.input = {"answer": answer, "citations": citations}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _judge_response(supported: bool, rationale: str = "r") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "judge_citation"
    block.input = {"supported": supported, "rationale": rationale}
    resp = MagicMock()
    resp.content = [block]
    return resp


class TestRunEndToEnd:
    def test_run_flows_retrieve_rerank_generate_evaluate_in_order(self) -> None:
        calls: list = []
        reranked = [_c("r1", 1)]
        eval_client = MagicMock()
        eval_client.messages.create.return_value = _judge_response(True)

        with patch("agents.qa_agent.anthropic.Anthropic") as MockGenClient:
            MockGenClient.return_value.messages.create.return_value = _tool_response(
                "ans", [{"claim": "ans", "chunk_ids": ["r1"]}]
            )
            result = run(
                "cudaMalloc parameters",
                bm25_index=_FakeBM25([_c("b1", 1)], calls),
                dense_index=_FakeDense([_c("d1", 1)], calls),
                router=_FakeRouter(reranked, calls),
                client=eval_client,
            )

        assert isinstance(result, EvalState)
        call_kinds = [c[0] for c in calls]
        assert call_kinds.index("bm25") < call_kinds.index("rerank")
        assert call_kinds.index("dense") < call_kinds.index("rerank")
        assert result.answer == "ans"
        assert result.citation_accuracy == 1.0
        assert len(result.citation_judgments) == 1

    def test_run_propagates_retrieve_error_without_calling_generate_or_evaluate(self) -> None:
        eval_client = MagicMock()

        with patch("agents.qa_agent.anthropic.Anthropic") as MockGenClient:
            result = run(
                "q",
                bm25_index=_RaisingBM25(),
                dense_index=_FakeDense([], []),
                router=_FakeRouter([], []),
                client=eval_client,
            )
            MockGenClient.return_value.messages.create.assert_not_called()

        assert result.error == "bm25 index unavailable"
        assert result.answer is None
        eval_client.messages.create.assert_not_called()

    def test_run_no_citations_yields_zero_accuracy_without_evaluate_calls(self) -> None:
        eval_client = MagicMock()

        with patch("agents.qa_agent.anthropic.Anthropic") as MockGenClient:
            MockGenClient.return_value.messages.create.return_value = _tool_response("ans", [])
            result = run(
                "q",
                bm25_index=_FakeBM25([_c("b1", 1)], []),
                dense_index=_FakeDense([], []),
                router=_FakeRouter([_c("r1", 1)], []),
                client=eval_client,
            )

        assert result.citation_accuracy == 0.0
        eval_client.messages.create.assert_not_called()

    def test_run_propagates_custom_query_id_through_all_stages(self) -> None:
        eval_client = MagicMock()
        eval_client.messages.create.return_value = _judge_response(True)

        with patch("agents.qa_agent.anthropic.Anthropic") as MockGenClient:
            MockGenClient.return_value.messages.create.return_value = _tool_response(
                "ans", [{"claim": "ans", "chunk_ids": ["r1"]}]
            )
            result = run(
                "q",
                query_id="fixed0001",
                bm25_index=_FakeBM25([], []),
                dense_index=_FakeDense([], []),
                router=_FakeRouter([_c("r1", 1)], []),
                client=eval_client,
            )

        assert result.query_id == "fixed0001"

    def test_build_graph_compiles_with_fakes(self) -> None:
        eval_client = MagicMock()
        eval_client.messages.create.return_value = _judge_response(True)
        graph = build_graph(_FakeBM25([], []), _FakeDense([], []), _FakeRouter([], []), eval_client)
        state = EvalState(query="q")

        with patch("agents.qa_agent.anthropic.Anthropic") as MockGenClient:
            MockGenClient.return_value.messages.create.return_value = _tool_response("ans", [])
            result = graph.invoke(state)

        final = EvalState(**result) if isinstance(result, dict) else result
        assert final.answer == ""
        assert final.citation_accuracy == 0.0
