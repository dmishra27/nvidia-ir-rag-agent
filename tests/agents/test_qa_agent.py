"""Contract tests for agents/qa_agent.py.

Per AGENTS.md ("LangGraph nodes: contract tests on state schema only"),
these tests verify how each node transforms the QAState schema using fake
BM25/dense/router objects and a mocked Anthropic client — never a real
index, Qdrant connection, cross-encoder, or live API call.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.qa_agent import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
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


class _RaisingDense:
    def search(self, query: str, top_k: int) -> list[Candidate]:
        raise RuntimeError("404 Collection 'nvidia_ir_chunks' doesn't exist!")


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

    def test_none_dense_index_skips_dense_and_keeps_bm25_results(self) -> None:
        """RERANKER_MODE=fallback (api/dependencies.py) passes dense_index=None
        on purpose -- that's a deliberate skip, not a failure, so BM25 results
        still come back fused/ranked instead of being discarded via state.error."""
        calls: list = []
        bm25_results = [_c("b1", 1)]
        node = make_retrieve_node(_FakeBM25(bm25_results, calls), None)
        state = QAState(query="q")

        result = node(state)

        assert result.error is None
        assert [c.chunk_id for c in result.fused_results] == ["b1"]

    def test_dense_failure_degrades_to_bm25_only_instead_of_discarding_results(self) -> None:
        """F-14 on the /ask path: a dense-search failure (Qdrant collection not
        yet populated on a clean clone) must NOT discard the BM25 half.
        retrieve degrades to BM25-only -- no error, BM25 results still fused --
        so `generate` gets real context instead of billing an Anthropic call
        for an answer grounded in nothing."""
        calls: list = []
        bm25_results = [_c("b1", 1), _c("b2", 2)]
        node = make_retrieve_node(_FakeBM25(bm25_results, calls), _RaisingDense())
        state = QAState(query="q")

        result = node(state)

        assert result.error is None
        assert [c.chunk_id for c in result.fused_results] == ["b1", "b2"]

    def test_sets_error_when_bm25_raises_even_if_dense_would_succeed(self) -> None:
        """The degradation is one-directional: BM25 is the floor. If BM25 itself
        raises there is nothing to degrade to, so error is set."""
        node = make_retrieve_node(_RaisingBM25(), _FakeDense([_c("d1", 1)], []))
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

    def test_skips_malformed_citation_entries_instead_of_crashing(self) -> None:
        # Live-observed on Day 9: Claude occasionally returns a citations list
        # item as a bare string rather than {claim, chunk_ids}, despite the
        # forced tool_choice schema. This must degrade gracefully, not crash.
        node = make_generate_node()
        state = QAState(query="q", reranked_results=[_c("r1", 1), _c("r2", 2)])
        mock_resp = _tool_response(
            "answer text",
            [
                {"claim": "a well-formed claim", "chunk_ids": ["r1"]},
                "a malformed bare-string citation",
            ],
        )

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = node(state)

        assert result.answer == "answer text"
        assert result.citations == [Citation(claim="a well-formed claim", chunk_ids=["r1"])]
        assert result.error is None

    def test_recovers_citations_sent_as_a_stringified_json_array(self) -> None:
        # Live-observed on Day 9 (query Q10): the whole `citations` value
        # sometimes arrives as a JSON *string* rather than a native array.
        # Iterating a raw string used to walk individual characters; this
        # must instead parse and recover the real citation objects.
        node = make_generate_node()
        state = QAState(query="q", reranked_results=[_c("r1", 1)])
        stringified = json.dumps([{"claim": "a claim", "chunk_ids": ["r1"]}])
        mock_resp = _tool_response("answer text", stringified)

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = node(state)

        assert result.answer == "answer text"
        assert result.citations == [Citation(claim="a claim", chunk_ids=["r1"])]
        assert result.error is None

    def test_unparseable_stringified_citations_falls_back_to_empty_without_crashing(self) -> None:
        node = make_generate_node()
        state = QAState(query="q", reranked_results=[_c("r1", 1)])
        mock_resp = _tool_response("answer text", "not valid json at all {{{")

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = node(state)

        assert result.answer == "answer text"
        assert result.citations == []
        assert result.error is None

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
# generate node — prompt variants (docs/uat/prompt_variant_comparison.md)
# ---------------------------------------------------------------------------


class TestPromptVariants:
    def test_default_prompt_variant_is_baseline_unless_evaluation_changes_it(self) -> None:
        # Guards against silently flipping the shipped default without updating
        # docs/uat/prompt_variant_comparison.md's stated winner to match.
        assert DEFAULT_PROMPT_VARIANT in PROMPT_VARIANTS

    def test_baseline_matches_the_original_unmodified_instruction(self) -> None:
        node = make_generate_node(prompt_variant="baseline")
        state = QAState(query="q", reranked_results=[_c("r1", 1)])
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            node(state)
            _, kwargs = MockClient.return_value.messages.create.call_args
            content = kwargs["messages"][0]["content"]

        assert "Every claim in your answer must cite the chunk_id(s)" in content
        assert "verify" not in content.lower()

    def test_cite_verify_asks_the_model_to_check_claims_before_answering(self) -> None:
        node = make_generate_node(prompt_variant="cite_verify")
        state = QAState(query="q", reranked_results=[_c("r1", 1)])
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            node(state)
            _, kwargs = MockClient.return_value.messages.create.call_args
            content = kwargs["messages"][0]["content"]

        assert "confirm the passage actually states" in content

    def test_both_variants_still_interpolate_query_and_context(self) -> None:
        state = QAState(query="a very specific question", reranked_results=[_c("r1", 1)])
        mock_resp = _tool_response("ans", [])

        for variant in PROMPT_VARIANTS:
            node = make_generate_node(prompt_variant=variant)
            with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
                MockClient.return_value.messages.create.return_value = mock_resp
                node(state)
                _, kwargs = MockClient.return_value.messages.create.call_args
                content = kwargs["messages"][0]["content"]

            assert "a very specific question" in content, variant
            assert "r1" in content, variant

    def test_unknown_prompt_variant_raises_at_construction_not_at_call_time(self) -> None:
        with pytest.raises(KeyError):
            make_generate_node(prompt_variant="does-not-exist")

    def test_build_graph_accepts_prompt_variant(self) -> None:
        # Regression guard: build_graph must thread prompt_variant through to
        # make_generate_node rather than always using the default.
        mock_resp = _tool_response("ans", [])
        graph = build_graph(
            _FakeBM25([], []), _FakeDense([], []), _FakeRouter([_c("r1", 1)], []), prompt_variant="cite_verify"
        )
        state = QAState(query="q")

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            graph.invoke(state)
            _, kwargs = MockClient.return_value.messages.create.call_args
            content = kwargs["messages"][0]["content"]

        assert "confirm the passage actually states" in content


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

    def test_run_answers_from_bm25_context_when_dense_unavailable(self) -> None:
        """F-14 end to end on /ask: with live_fast (both signals wired) but
        Qdrant down, run() still hands `generate` the BM25 context and returns
        a real answer instead of the empty-context path."""
        calls: list = []
        reranked = [_c("b1", 1)]
        mock_resp = _tool_response("Grounded in BM25.", [{"claim": "Grounded in BM25.", "chunk_ids": ["b1"]}])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            result = run(
                "cudaMalloc parameters",
                bm25_index=_FakeBM25([_c("b1", 1), _c("b2", 2)], calls),
                dense_index=_RaisingDense(),
                router=_FakeRouter(reranked, calls),
            )
            MockClient.return_value.messages.create.assert_called_once()

        assert result.error is None
        assert result.answer == "Grounded in BM25."
        assert result.citations == [Citation(claim="Grounded in BM25.", chunk_ids=["b1"])]

    def test_build_graph_compiles_with_fakes(self) -> None:
        graph = build_graph(_FakeBM25([], []), _FakeDense([], []), _FakeRouter([], []))
        state = QAState(query="q")

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = _tool_response("ans", [])
            result = graph.invoke(state)

        final = QAState(**result) if isinstance(result, dict) else result
        assert final.answer == ""
        assert final.citations == []

    def test_run_with_explicit_none_dense_index_skips_connect(self) -> None:
        """Regression test: dense_index or DenseIndex.connect() used to treat an
        explicitly-passed None the same as "omitted", silently reconnecting to
        Qdrant anyway -- which would reintroduce the >512MB OOM RERANKER_MODE=
        fallback (api/dependencies.py) is meant to avoid. Must stay None."""
        with patch("agents.qa_agent.DenseIndex.connect") as mock_connect, patch(
            "agents.qa_agent.anthropic.Anthropic"
        ) as MockClient:
            MockClient.return_value.messages.create.return_value = _tool_response("ans", [])
            run(
                "q",
                bm25_index=_FakeBM25([_c("b1", 1)], []),
                dense_index=None,
                router=_FakeRouter([_c("b1", 1)], []),
            )

        mock_connect.assert_not_called()

    def test_run_without_dense_index_arg_still_connects_for_real(self) -> None:
        """monitoring/quality_regression.py and evaluation/ragas_suite.py call
        qa_agent.run() without a dense_index -- they must keep getting a real
        connected DenseIndex; only an explicit None skips it."""
        with patch("agents.qa_agent.DenseIndex.connect") as mock_connect, patch(
            "agents.qa_agent.anthropic.Anthropic"
        ) as MockClient:
            mock_connect.return_value = _FakeDense([], [])
            MockClient.return_value.messages.create.return_value = _tool_response("ans", [])
            run("q", bm25_index=_FakeBM25([], []), router=_FakeRouter([], []))

        mock_connect.assert_called_once()
