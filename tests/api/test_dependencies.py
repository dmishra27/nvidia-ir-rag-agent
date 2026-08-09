"""Tests for api/dependencies.py's RERANKER_MODE=fallback gating.

RERANKER_MODE=fallback must skip DenseIndex.connect() and
MSMarcoReranker.load() entirely (both pull in qdrant_client and
sentence-transformers/torch model loads, which exceed Render's free-tier
512MB before a single request is served -- see api/dependencies.py's
module docstring). Every other mode must keep loading them for real.
Never actually calls the real .connect()/.load() -- always patched, per
AGENTS.md's rule to never load a real index or connect to Qdrant in a
unit test.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from api.dependencies import get_dense_index, get_msmarco_reranker


@pytest.fixture(autouse=True)
def _clear_caches_and_env():
    """lru_cache persists across tests unless cleared; RERANKER_MODE must not
    leak between tests either (os.environ is process-global)."""
    get_dense_index.cache_clear()
    get_msmarco_reranker.cache_clear()
    original = os.environ.get("RERANKER_MODE")
    yield
    if original is None:
        os.environ.pop("RERANKER_MODE", None)
    else:
        os.environ["RERANKER_MODE"] = original
    get_dense_index.cache_clear()
    get_msmarco_reranker.cache_clear()


class TestGetDenseIndex:
    def test_fallback_mode_returns_none_without_connecting(self) -> None:
        os.environ["RERANKER_MODE"] = "fallback"

        with patch("api.dependencies.DenseIndex.connect") as mock_connect:
            result = get_dense_index()

        assert result is None
        mock_connect.assert_not_called()

    def test_live_fast_mode_connects_for_real(self) -> None:
        os.environ["RERANKER_MODE"] = "live_fast"

        with patch("api.dependencies.DenseIndex.connect") as mock_connect:
            mock_connect.return_value = "a-real-dense-index"
            result = get_dense_index()

        assert result == "a-real-dense-index"
        mock_connect.assert_called_once()

    def test_unset_env_var_defaults_to_connecting(self) -> None:
        os.environ.pop("RERANKER_MODE", None)

        with patch("api.dependencies.DenseIndex.connect") as mock_connect:
            get_dense_index()

        mock_connect.assert_called_once()


class TestGetMsmarcoReranker:
    def test_fallback_mode_returns_none_without_loading(self) -> None:
        os.environ["RERANKER_MODE"] = "fallback"

        with patch("api.dependencies.MSMarcoReranker.load") as mock_load:
            result = get_msmarco_reranker()

        assert result is None
        mock_load.assert_not_called()

    def test_live_fast_mode_loads_for_real(self) -> None:
        os.environ["RERANKER_MODE"] = "live_fast"

        with patch("api.dependencies.MSMarcoReranker.load") as mock_load:
            mock_load.return_value = "a-real-reranker"
            result = get_msmarco_reranker()

        assert result == "a-real-reranker"
        mock_load.assert_called_once()
