"""Contract tests for agents/retrieval_agent.py.

Per AGENTS.md ("LangGraph nodes: contract tests on state schema only"),
these tests verify how each node transforms the AgentState schema — which
fields get set, which are preserved, how errors propagate — using fake
BM25/dense/router objects. They do not assert anything about retrieval
quality or content, and never load a real index, connect to Qdrant, or
load a cross-encoder model.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.retrieval_agent import (
    AgentState,
    build_graph,
    make_rerank_node,
    make_retrieve_node,
    return_results,
    run,
    run_state,
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
        raise RuntimeError("qdrant unreachable")


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


# ---------------------------------------------------------------------------
# AgentState — defaults
# ---------------------------------------------------------------------------


def test_agent_state_defaults() -> None:
    state = AgentState(query="cudaMalloc parameters")

    assert state.top_k == 10
    assert state.candidate_pool_size == 100
    assert state.bm25_results == []
    assert state.dense_results == []
    assert state.fused_results == []
    assert state.reranked_results == []
    assert state.results == []
    assert state.error is None
    assert isinstance(state.query_id, str) and state.query_id


def test_agent_state_query_id_is_overridable() -> None:
    state = AgentState(query="q", query_id="fixed0001")

    assert state.query_id == "fixed0001"


# ---------------------------------------------------------------------------
# retrieve node
# ---------------------------------------------------------------------------


class TestRetrieveNode:
    def test_populates_bm25_dense_and_fused_results(self) -> None:
        calls: list = []
        bm25_results = [_c("b1", 1)]
        dense_results = [_c("d1", 1)]
        node = make_retrieve_node(_FakeBM25(bm25_results, calls), _FakeDense(dense_results, calls))
        state = AgentState(query="cudaMalloc parameters")

        result = node(state)

        assert result.bm25_results == bm25_results
        assert result.dense_results == dense_results
        assert {r.chunk_id for r in result.fused_results} == {"b1", "d1"}

    def test_preserves_query_id_and_query(self) -> None:
        node = make_retrieve_node(_FakeBM25([], []), _FakeDense([], []))
        state = AgentState(query="cudaMalloc parameters", query_id="fixed0001")

        result = node(state)

        assert result.query_id == "fixed0001"
        assert result.query == "cudaMalloc parameters"

    def test_passes_candidate_pool_size_as_top_k_to_both_signals(self) -> None:
        calls: list = []
        node = make_retrieve_node(_FakeBM25([], calls), _FakeDense([], calls))
        state = AgentState(query="q", candidate_pool_size=42)

        node(state)

        assert ("bm25", "q", 42) in calls
        assert ("dense", "q", 42) in calls

    def test_sets_error_when_bm25_raises(self) -> None:
        node = make_retrieve_node(_RaisingBM25(), _FakeDense([], []))
        state = AgentState(query="q")

        result = node(state)

        assert result.error == "bm25 index unavailable"
        assert result.bm25_results == []
        assert result.fused_results == []

    def test_dense_failure_degrades_to_bm25_only_instead_of_discarding_results(self) -> None:
        """F-14: a dense-search failure (e.g. Qdrant's nvidia_ir_chunks
        collection not yet populated on a clean clone) must NOT discard the
        already-succeeded BM25 half. retrieve degrades to BM25-only: no error,
        BM25 results preserved, dense pool empty, fusion runs over BM25 alone."""
        calls: list = []
        bm25_results = [_c("b1", 1), _c("b2", 2)]
        node = make_retrieve_node(_FakeBM25(bm25_results, calls), _RaisingDense())
        state = AgentState(query="q")

        result = node(state)

        assert result.error is None
        assert result.bm25_results == bm25_results
        assert result.dense_results == []
        assert [c.chunk_id for c in result.fused_results] == ["b1", "b2"]

    def test_sets_error_when_bm25_raises_even_if_dense_would_succeed(self) -> None:
        """The degradation is one-directional: BM25 is the floor. If BM25
        itself raises there is nothing to degrade to, so error is set."""
        node = make_retrieve_node(_RaisingBM25(), _FakeDense([_c("d1", 1)], []))
        state = AgentState(query="q")

        result = node(state)

        assert result.error == "bm25 index unavailable"
        assert result.fused_results == []

    def test_error_path_preserves_query_id(self) -> None:
        node = make_retrieve_node(_RaisingBM25(), _FakeDense([], []))
        state = AgentState(query="q", query_id="fixed0001")

        result = node(state)

        assert result.query_id == "fixed0001"

    def test_none_dense_index_skips_dense_and_keeps_bm25_results(self) -> None:
        """RERANKER_MODE=fallback (api/dependencies.py) passes dense_index=None
        on purpose -- that's a deliberate skip, not a failure, so BM25 results
        still come back fused/ranked instead of being discarded via state.error."""
        calls: list = []
        bm25_results = [_c("b1", 1)]
        node = make_retrieve_node(_FakeBM25(bm25_results, calls), None)
        state = AgentState(query="q")

        result = node(state)

        assert result.error is None
        assert result.bm25_results == bm25_results
        assert result.dense_results == []
        assert [c.chunk_id for c in result.fused_results] == ["b1"]


# ---------------------------------------------------------------------------
# rerank node
# ---------------------------------------------------------------------------


class TestRerankNode:
    def test_populates_reranked_results_from_router(self) -> None:
        reranked = [_c("r1", 1)]
        calls: list = []
        node = make_rerank_node(_FakeRouter(reranked, calls))
        state = AgentState(query="q", fused_results=[_c("f1", 1)])

        result = node(state)

        assert result.reranked_results == reranked

    def test_passes_query_fused_results_top_k_and_query_id_to_router(self) -> None:
        calls: list = []
        fused = [_c("f1", 1)]
        node = make_rerank_node(_FakeRouter([], calls))
        state = AgentState(query="cudaMalloc parameters", fused_results=fused, top_k=5, query_id="fixed0001")

        node(state)

        assert calls == [("rerank", "cudaMalloc parameters", fused, 5, "fixed0001")]

    def test_preserves_fused_results(self) -> None:
        fused = [_c("f1", 1)]
        node = make_rerank_node(_FakeRouter([_c("r1", 1)], []))
        state = AgentState(query="q", fused_results=fused)

        result = node(state)

        assert result.fused_results == fused

    def test_skips_router_call_and_preserves_state_when_error_already_set(self) -> None:
        calls: list = []
        node = make_rerank_node(_FakeRouter([_c("r1", 1)], calls))
        state = AgentState(query="q", error="upstream failure")

        result = node(state)

        assert calls == []
        assert result.error == "upstream failure"
        assert result.reranked_results == []

    def test_sets_error_when_router_raises(self) -> None:
        node = make_rerank_node(_RaisingRouter())
        state = AgentState(query="q", fused_results=[_c("f1", 1)])

        result = node(state)

        assert result.error == "reranker misconfigured"
        assert result.reranked_results == []


# ---------------------------------------------------------------------------
# return_results node
# ---------------------------------------------------------------------------


class TestReturnResults:
    def test_sets_results_from_reranked_results_when_present(self) -> None:
        reranked = [_c("r1", 1)]
        state = AgentState(query="q", fused_results=[_c("f1", 1)], reranked_results=reranked)

        result = return_results(state)

        assert result.results == reranked

    def test_falls_back_to_fused_results_when_reranked_empty(self) -> None:
        fused = [_c("f1", 1), _c("f2", 2)]
        state = AgentState(query="q", fused_results=fused, top_k=10)

        result = return_results(state)

        assert result.results == fused

    def test_fallback_truncates_fused_results_to_top_k(self) -> None:
        fused = [_c(f"f{i}", i) for i in range(1, 6)]
        state = AgentState(query="q", fused_results=fused, top_k=2)

        result = return_results(state)

        assert len(result.results) == 2

    def test_sets_empty_results_when_error_present(self) -> None:
        state = AgentState(query="q", fused_results=[_c("f1", 1)], error="boom")

        result = return_results(state)

        assert result.results == []

    def test_preserves_query_id(self) -> None:
        state = AgentState(query="q", query_id="fixed0001", reranked_results=[_c("r1", 1)])

        result = return_results(state)

        assert result.query_id == "fixed0001"


# ---------------------------------------------------------------------------
# build_graph / run — end-to-end schema contract
# ---------------------------------------------------------------------------


class TestRunEndToEnd:
    def test_run_returns_list_of_candidate(self) -> None:
        calls: list = []
        reranked = [_c("r1", 1)]
        result = run(
            "cudaMalloc parameters",
            bm25_index=_FakeBM25([_c("b1", 1)], calls),
            dense_index=_FakeDense([_c("d1", 1)], calls),
            router=_FakeRouter(reranked, calls),
        )

        assert isinstance(result, list)
        assert all(isinstance(r, Candidate) for r in result)
        assert result == reranked

    def test_run_calls_signals_in_retrieve_then_rerank_order(self) -> None:
        calls: list = []
        run(
            "q",
            bm25_index=_FakeBM25([], calls),
            dense_index=_FakeDense([], calls),
            router=_FakeRouter([], calls),
        )

        call_kinds = [c[0] for c in calls]
        assert call_kinds.index("bm25") < call_kinds.index("rerank")
        assert call_kinds.index("dense") < call_kinds.index("rerank")

    def test_run_propagates_custom_query_id_through_router_call(self) -> None:
        calls: list = []
        run(
            "q",
            query_id="fixed0001",
            bm25_index=_FakeBM25([], calls),
            dense_index=_FakeDense([], calls),
            router=_FakeRouter([], calls),
        )

        rerank_call = next(c for c in calls if c[0] == "rerank")
        assert rerank_call[4] == "fixed0001"

    def test_run_falls_back_to_fused_results_when_router_returns_nothing(self) -> None:
        fused_bm25 = [_c("b1", 1)]
        result = run(
            "q",
            bm25_index=_FakeBM25(fused_bm25, []),
            dense_index=_FakeDense([], []),
            router=_FakeRouter([], []),
        )

        assert [r.chunk_id for r in result] == ["b1"]

    def test_run_returns_bm25_results_when_dense_is_unavailable(self) -> None:
        """F-14 end to end: with live_fast (both signals wired) but Qdrant
        down, run() still returns the BM25-ranked list instead of []."""
        result = run(
            "cudaMalloc parameters",
            bm25_index=_FakeBM25([_c("b1", 1), _c("b2", 2)], []),
            dense_index=_RaisingDense(),
            router=_FakeRouter([], []),  # router returns nothing -> fused fallback
        )

        assert [r.chunk_id for r in result] == ["b1", "b2"]

    def test_run_state_reports_no_error_on_dense_degradation(self) -> None:
        state = run_state(
            "q",
            bm25_index=_FakeBM25([_c("b1", 1)], []),
            dense_index=_RaisingDense(),
            router=_FakeRouter([_c("b1", 1)], []),
        )

        assert state.error is None
        assert [r.chunk_id for r in state.results] == ["b1"]

    def test_run_state_reports_error_when_retrieval_fails_entirely(self) -> None:
        """F-15: run_state surfaces the failure so /search can return an
        error field + 503 rather than an ambiguous empty 200."""
        state = run_state(
            "q",
            bm25_index=_RaisingBM25(),
            dense_index=_FakeDense([_c("d1", 1)], []),
            router=_FakeRouter([], []),
        )

        assert state.error == "bm25 index unavailable"
        assert state.results == []

    def test_build_graph_compiles_with_fakes(self) -> None:
        graph = build_graph(_FakeBM25([], []), _FakeDense([], []), _FakeRouter([], []))
        state = AgentState(query="q")

        result = graph.invoke(state)

        final = AgentState(**result) if isinstance(result, dict) else result
        assert final.results == []

    def test_run_with_explicit_none_dense_index_skips_connect_and_returns_bm25_results(
        self,
    ) -> None:
        """Regression test: dense_index or DenseIndex.connect() used to treat an
        explicitly-passed None the same as "omitted", silently reconnecting to
        Qdrant anyway -- which would reintroduce the >512MB OOM RERANKER_MODE=
        fallback (api/dependencies.py) is meant to avoid. Must stay None."""
        with patch("agents.retrieval_agent.DenseIndex.connect") as mock_connect:
            result = run(
                "q",
                bm25_index=_FakeBM25([_c("b1", 1)], []),
                dense_index=None,
                router=_FakeRouter([], []),
            )

        mock_connect.assert_not_called()
        assert [r.chunk_id for r in result] == ["b1"]

    def test_run_without_dense_index_arg_still_connects_for_real(self) -> None:
        """Callers that omit dense_index entirely (run_hybrid_search.py, ad hoc
        scripts) must keep getting a real connected DenseIndex -- only an
        explicit None (the RERANKER_MODE=fallback path) skips it."""
        with patch("agents.retrieval_agent.DenseIndex.connect") as mock_connect:
            mock_connect.return_value = _FakeDense([], [])
            run(
                "q",
                bm25_index=_FakeBM25([], []),
                router=_FakeRouter([], []),
            )

        mock_connect.assert_called_once()
