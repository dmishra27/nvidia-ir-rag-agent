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

from fastapi.testclient import TestClient

from api.dependencies import get_bm25_index, get_dense_index, get_msmarco_reranker
from api.main import create_app
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

    def test_top_k_truncates_results(self) -> None:
        reranked = [_c(f"r{i}", i) for i in range(1, 6)]
        client = _make_client([], [], reranked)

        resp = client.post("/search", json={"query": "q", "top_k": 2})

        assert len(resp.json()["results"]) == 2

    def test_missing_query_returns_422(self) -> None:
        client = _make_client([], [], [])

        resp = client.post("/search", json={})

        assert resp.status_code == 422


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
