"""Endpoint tests for /evaluation.

Mirrors tests/api/test_main.py::TestUI's shape for the same reason: static
HTML served straight off disk, no fakes/dependency_overrides needed since
the route reads no index/DB/model, just session_factory=MagicMock() to
keep RequestLatencyMiddleware from needing a live Postgres connection.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.main import create_app


class TestEvaluation:
    def test_returns_html_page(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        resp = client.get("/evaluation")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_page_contains_key_figures(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/evaluation").text

        # Method comparison (Round 2 pool) headline
        assert "BM25" in html and "Dense" in html and "RRF" in html
        # Benchmark + CI gate
        assert "0.5333" in html  # Config A mean NDCG@10
        assert "0.5280" in html  # Config C mean NDCG@10
        assert "a55012a4" in html  # Config A run_id
        assert "c827cd71" in html  # Config C run_id
        assert "0.50" in html  # DEFAULT_THRESHOLD
        assert "Hardware-blocked" in html  # Config B
        # RAGAS faithfulness
        assert "0.7616" in html
        assert "faithfulness" in html.lower()

    def test_architectural_findings_reflect_live_q1_measurement(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/evaluation").text

        # RRF corroboration bias, measured figures
        assert "0.0235" in html  # target chunk's fused RRF score
        assert "0.0317" in html  # fused rank-1 chunk's RRF score
        # Cross-encoder result: independent scoring, but same wrong chunk won
        assert "6.0145" in html  # CE rank-1 score (cudaFreeArray chunk)
        assert "5.4647" in html  # CE rank-2 score (actual cudaMalloc signature)
        assert "Both retrieval strategies failed on" in html
        # The corrected claim no longer asserts immunity to the bias
        assert "should not be subject to this failure mode" not in html
        # New root-cause finding
        assert "corpus-quality" in html.lower()
        assert "cudaStreamAddCallback" in html

    def test_page_links_back_to_search_ui(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/evaluation").text

        assert 'href="/"' in html

    def test_index_links_to_evaluation(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        html = client.get("/").text

        assert 'href="/evaluation"' in html

    def test_evaluation_route_is_excluded_from_the_openapi_schema(self) -> None:
        client = TestClient(create_app(session_factory=MagicMock()))

        schema = client.get("/openapi.json").json()

        assert "/evaluation" not in schema["paths"]
