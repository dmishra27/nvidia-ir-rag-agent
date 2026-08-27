"""Endpoint tests for /search, /ask, /health.

BM25/dense/reranker components are swapped for fakes via
app.dependency_overrides; Anthropic is patched at the agents.qa_agent
module boundary — per AGENTS.md's rule to mock all embedding/LLM calls in
tests and never load a real index, connect to Qdrant, or load a
cross-encoder. session_factory=MagicMock() keeps the monitoring
middleware from needing a live Postgres connection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_bm25_index, get_dense_index, get_msmarco_reranker
from api.main import create_app
from retrieval.candidates import Candidate
from retrieval.reranker_router import DEFAULT_MODE


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


class _RaisingBM25:
    def search(self, query: str, top_k: int) -> list[Candidate]:
        raise RuntimeError("bm25 index unavailable")


class _RaisingDense:
    def search(self, query: str, top_k: int) -> list[Candidate]:
        raise RuntimeError("404 Collection 'nvidia_ir_chunks' doesn't exist!")


class _FakeMSMarco:
    def __init__(self, results: list[Candidate]) -> None:
        self._results = results

    def rerank(self, query: str, candidates: list[Candidate], top_k: int, query_id: str) -> list[Candidate]:
        return self._results[:top_k]


def _make_client(
    bm25_results: list[Candidate], dense_results: list[Candidate], reranked_results: list[Candidate]
) -> TestClient:
    app = create_app(session_factory=MagicMock())
    app.dependency_overrides[get_bm25_index] = lambda: _FakeBM25(bm25_results)
    app.dependency_overrides[get_dense_index] = lambda: _FakeDense(dense_results)
    app.dependency_overrides[get_msmarco_reranker] = lambda: _FakeMSMarco(reranked_results)
    return TestClient(app)


def _tool_response(answer: str, citations: list[dict]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"answer": answer, "citations": citations}
    resp = MagicMock()
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_ok_status(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "service": "nvidia-ir-rag-agent"}


# ---------------------------------------------------------------------------
# / (HTML search UI)
# ---------------------------------------------------------------------------


class TestUI:
    def test_returns_html_page(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        resp = client.get("/")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_page_has_title_and_subtitle(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        assert "NVIDIA" in html and "Documentation" in html and "Search" in html
        assert "5,389 passages" in html

    def test_page_lists_all_five_example_queries_grouped_by_intent(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        for group_label in [
            "Look up an API",
            "Diagnose an error",
            "Understand a concept",
        ]:
            assert group_label in html

        for example in [
            "cudaMalloc function parameters",
            "cudaHostAlloc pinned memory",
            "cudaMemcpy host to device",
            "cudaErrorInvalidValue error code",
            "problems with threads executing different code paths",
        ]:
            assert example in html

    def test_page_calls_search_with_fallback_reranker_mode(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        assert "/search?RERANKER_MODE=fallback" in html

    def test_page_links_to_docs_and_github(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        assert 'href="/docs"' in html
        assert "github.com/dmishra27/nvidia-ir-rag-agent" in html

    def test_page_has_author_signature_linked_to_github_profile(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        assert 'href="https://github.com/dmishra27"' in html
        assert "Debabrata Mishra" in html
        assert "ACM SIGIR 2024 Co-author" in html
        assert "MSc Data Science, University of Glasgow" in html
        assert "Greenock, Scotland" in html

    def test_footer_tagline_explains_fallback_mode_and_links_to_evaluation(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        assert 'class="tagline"' in html
        assert "BM25 keyword search only" in html
        assert "512" in html and "MB" in html
        assert 'href="/evaluation"' in html

    def test_ui_route_is_excluded_from_the_openapi_schema(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        schema = client.get("/openapi.json").json()

        assert "/" not in schema["paths"]

    def test_docs_still_serves_swagger_ui(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        resp = client.get("/docs")

        assert resp.status_code == 200
        assert "swagger" in resp.text.lower()

    def test_page_has_corpus_transparency_panel(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        assert "<details" in html and 'class="corpus-panel"' in html
        for doc in [
            "CUDA C++ Programming Guide",
            "CUDA C++ Best Practices Guide",
            "CUDA Math API Reference",
            "CUDA Runtime API",
            "Nsight Systems User Guide",
        ]:
            assert doc in html
        for out_of_scope_topic in ["NVLink", "H100", "TensorRT"]:
            assert out_of_scope_topic in html

    def test_page_has_per_result_feedback_controls(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        assert 'class="feedback-row"' in html
        assert 'data-reaction="up"' in html
        assert 'data-reaction="down"' in html
        assert "/feedback" in html  # the fetch() target the click handler posts to


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_reranked_results(self) -> None:
        client = _make_client([_c("b1", 1)], [_c("d1", 1)], [_c("r1", 1)])

        resp = client.post("/search", json={"query": "cudaMalloc parameters"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "cudaMalloc parameters"
        assert data["results"] == [{"chunk_id": "r1", "text": "text r1", "score": 1.0, "rank": 1}]
        assert isinstance(data["query_id"], str) and data["query_id"]

    def test_reranker_mode_query_param_echoed_in_response(self) -> None:
        client = _make_client([], [], [_c("r1", 1)])

        resp = client.post("/search?RERANKER_MODE=fallback", json={"query": "q"})

        assert resp.status_code == 200
        assert resp.json()["reranker_mode"] == "fallback"

    def test_reranker_mode_echoes_env_var_when_not_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEF-03: no RERANKER_MODE query param -- must echo the mode
        retrieval/reranker_router.py actually resolved to (the env var),
        not null."""
        monkeypatch.setenv("RERANKER_MODE", "live_quality")
        client = _make_client([], [], [_c("r1", 1)])

        resp = client.post("/search", json={"query": "q"})

        assert resp.status_code == 200
        assert resp.json()["reranker_mode"] == "live_quality"

    def test_reranker_mode_echoes_default_when_not_passed_and_no_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DEF-03: no RERANKER_MODE query param and no env var -- must echo
        DEFAULT_MODE, not null."""
        monkeypatch.delenv("RERANKER_MODE", raising=False)
        client = _make_client([], [], [_c("r1", 1)])

        resp = client.post("/search", json={"query": "q"})

        assert resp.status_code == 200
        assert resp.json()["reranker_mode"] == DEFAULT_MODE

    def test_top_k_truncates_results(self) -> None:
        reranked = [_c(f"r{i}", i) for i in range(1, 6)]
        client = _make_client([], [], reranked)

        resp = client.post("/search", json={"query": "q", "top_k": 2})

        assert len(resp.json()["results"]) == 2

    def test_missing_query_returns_422(self) -> None:
        client = _make_client([], [], [])

        resp = client.post("/search", json={})

        assert resp.status_code == 422

    def test_fallback_mode_with_no_dense_index_or_reranker_still_returns_200(self) -> None:
        """This is the shape RERANKER_MODE=fallback actually produces on Render
        (api/dependencies.py returns None for both rather than loading torch/
        qdrant_client, to fit the 512MB free tier) -- must not 500."""
        app = create_app(session_factory=MagicMock())
        app.dependency_overrides[get_bm25_index] = lambda: _FakeBM25([_c("b1", 1)])
        app.dependency_overrides[get_dense_index] = lambda: None
        app.dependency_overrides[get_msmarco_reranker] = lambda: None
        client = TestClient(app)

        resp = client.post("/search?RERANKER_MODE=fallback", json={"query": "cudaMalloc parameters"})

        assert resp.status_code == 200
        assert [r["chunk_id"] for r in resp.json()["results"]] == ["b1"]

    def test_dense_unavailable_degrades_to_bm25_and_still_returns_200(self) -> None:
        """F-14: default RERANKER_MODE=live_fast wires both signals, but on a
        clean clone Qdrant's nvidia_ir_chunks collection doesn't exist. /search
        must return the committed BM25 results, not a bare empty list."""
        app = create_app(session_factory=MagicMock())
        app.dependency_overrides[get_bm25_index] = lambda: _FakeBM25([_c("b1", 1), _c("b2", 2)])
        app.dependency_overrides[get_dense_index] = lambda: _RaisingDense()
        app.dependency_overrides[get_msmarco_reranker] = lambda: _FakeMSMarco([])
        client = TestClient(app)

        resp = client.post("/search", json={"query": "cudaMalloc parameters"})

        assert resp.status_code == 200
        data = resp.json()
        assert [r["chunk_id"] for r in data["results"]] == ["b1", "b2"]
        assert data["error"] is None

    def test_total_retrieval_failure_returns_503_with_error_field(self) -> None:
        """F-15: when retrieval fails outright (not just dense), the response
        must be distinguishable from 'no matches' -- 503 and a populated
        error field, not HTTP 200 {"results": []}."""
        app = create_app(session_factory=MagicMock())
        app.dependency_overrides[get_bm25_index] = lambda: _RaisingBM25()
        app.dependency_overrides[get_dense_index] = lambda: _RaisingDense()
        app.dependency_overrides[get_msmarco_reranker] = lambda: _FakeMSMarco([])
        client = TestClient(app)

        resp = client.post("/search", json={"query": "cudaMalloc parameters"})

        assert resp.status_code == 503
        data = resp.json()
        assert data["results"] == []
        assert data["error"] == "bm25 index unavailable"


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------


class TestAsk:
    def test_returns_answer_and_citations(self) -> None:
        client = _make_client([], [], [_c("r1", 1)])
        mock_resp = _tool_response(
            "Final answer.", [{"claim": "Final answer.", "chunk_ids": ["r1"]}]
        )

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post("/ask", json={"query": "cudaDeviceSynchronize return value"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Final answer."
        assert data["citations"] == [{"claim": "Final answer.", "chunk_ids": ["r1"]}]
        assert data["error"] is None

    def test_no_passages_skips_llm_and_returns_empty_answer(self) -> None:
        client = _make_client([], [], [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            resp = client.post("/ask", json={"query": "q"})
            MockClient.return_value.messages.create.assert_not_called()

        assert resp.status_code == 200
        assert resp.json()["answer"] == ""

    def test_surfaces_agent_error(self) -> None:
        client = _make_client([], [], [_c("r1", 1)])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = RuntimeError("api down")
            resp = client.post("/ask", json={"query": "q"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "api down"
        assert data["answer"] is None

    def test_reranker_mode_query_param_echoed_in_response(self) -> None:
        client = _make_client([], [], [_c("r1", 1)])
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post("/ask?RERANKER_MODE=fallback", json={"query": "q"})

        assert resp.json()["reranker_mode"] == "fallback"

    def test_reranker_mode_echoes_env_var_when_not_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEF-03: no RERANKER_MODE query param -- must echo the mode
        retrieval/reranker_router.py actually resolved to (the env var),
        not null."""
        monkeypatch.setenv("RERANKER_MODE", "live_quality")
        client = _make_client([], [], [_c("r1", 1)])
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post("/ask", json={"query": "q"})

        assert resp.json()["reranker_mode"] == "live_quality"

    def test_reranker_mode_echoes_default_when_not_passed_and_no_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DEF-03: no RERANKER_MODE query param and no env var -- must echo
        DEFAULT_MODE, not null."""
        monkeypatch.delenv("RERANKER_MODE", raising=False)
        client = _make_client([], [], [_c("r1", 1)])
        mock_resp = _tool_response("ans", [])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post("/ask", json={"query": "q"})

        assert resp.json()["reranker_mode"] == DEFAULT_MODE

    def test_fallback_mode_with_no_dense_index_or_reranker_still_returns_200(self) -> None:
        """Same real Render shape as TestSearch's equivalent test -- must not 500."""
        app = create_app(session_factory=MagicMock())
        app.dependency_overrides[get_bm25_index] = lambda: _FakeBM25([_c("b1", 1)])
        app.dependency_overrides[get_dense_index] = lambda: None
        app.dependency_overrides[get_msmarco_reranker] = lambda: None
        client = TestClient(app)
        mock_resp = _tool_response("ans", [{"claim": "ans", "chunk_ids": ["b1"]}])

        with patch("agents.qa_agent.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post("/ask?RERANKER_MODE=fallback", json={"query": "cudaMalloc parameters"})

        assert resp.status_code == 200
        assert resp.json()["answer"] == "ans"
