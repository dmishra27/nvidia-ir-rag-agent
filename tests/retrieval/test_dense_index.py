"""Unit tests for retrieval/dense_index.py.

Per AGENTS.md ("Mock all embedding and LLM calls in unit tests"), these tests
never load sentence-transformers/torch or open a network connection. A fake
QueryEncoder and a fake Qdrant client (matching the query_points(...).points
shape) stand in for the real model and the real cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from retrieval.candidates import Candidate
from retrieval.dense_index import DenseIndex

# ---------------------------------------------------------------------------
# Fakes — fixed query -> vector encoder, fixed vector -> scored points client
# ---------------------------------------------------------------------------


@dataclass
class _FakePoint:
    payload: dict | None
    score: float


class _FakeQueryResponse:
    def __init__(self, points: list[_FakePoint]) -> None:
        self.points = points


class _FakeQdrantClient:
    """Records every call and returns a scripted, already-ranked point list."""

    def __init__(self, points: list[_FakePoint]) -> None:
        self._points = points
        self.calls: list[dict] = []

    def query_points(self, collection_name: str, query: list[float], limit: int):
        self.calls.append({"collection_name": collection_name, "query": query, "limit": limit})
        return _FakeQueryResponse(self._points[:limit])


def _fake_encoder_factory(calls: list[str]):
    def encode(query: str) -> list[float]:
        calls.append(query)
        return [1.0, 0.0, 0.0]

    return encode


_POINTS = [
    _FakePoint({"chunk_id": "c1", "text": "NVLink bandwidth reaches 900 GB/s."}, 0.91),
    _FakePoint({"chunk_id": "c5", "text": "Shared memory within a CUDA block."}, 0.77),
    _FakePoint({"chunk_id": "c2", "text": "CUDA kernels are launched with a grid."}, 0.60),
]


# ---------------------------------------------------------------------------
# search() — happy path
# ---------------------------------------------------------------------------


def test_search_returns_candidates_from_qdrant_points() -> None:
    encoder_calls: list[str] = []
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory(encoder_calls), collection_name="nvidia_ir_chunks")

    results = index.search("NVLink bandwidth", top_k=3)

    assert all(isinstance(r, Candidate) for r in results)
    assert [r.chunk_id for r in results] == ["c1", "c5", "c2"]
    assert results[0].score == pytest.approx(0.91)
    assert results[0].text == "NVLink bandwidth reaches 900 GB/s."


def test_ranks_are_sequential_starting_at_one() -> None:
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory([]))

    results = index.search("NVLink bandwidth", top_k=3)

    assert [r.rank for r in results] == [1, 2, 3]


def test_preserves_qdrant_return_order_without_resorting() -> None:
    # Qdrant already returns points sorted by score desc; DenseIndex must not
    # re-sort, so an out-of-order fake response should pass through as-is.
    out_of_order = [
        _FakePoint({"chunk_id": "low"}, 0.10),
        _FakePoint({"chunk_id": "high"}, 0.95),
    ]
    client = _FakeQdrantClient(out_of_order)
    index = DenseIndex(client, _fake_encoder_factory([]))

    results = index.search("query", top_k=2)

    assert [r.chunk_id for r in results] == ["low", "high"]


# ---------------------------------------------------------------------------
# search() — query routed to encoder, top_k routed to Qdrant
# ---------------------------------------------------------------------------


def test_search_encodes_the_raw_query_text() -> None:
    encoder_calls: list[str] = []
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory(encoder_calls))

    index.search("CUDA memory allocation", top_k=5)

    assert encoder_calls == ["CUDA memory allocation"]


def test_search_passes_encoded_vector_and_top_k_and_collection_to_client() -> None:
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory([]), collection_name="my_collection")

    index.search("NVLink bandwidth", top_k=2)

    assert client.calls == [
        {"collection_name": "my_collection", "query": [1.0, 0.0, 0.0], "limit": 2}
    ]


# ---------------------------------------------------------------------------
# search() — top_k boundary
# ---------------------------------------------------------------------------


def test_top_k_limits_result_count() -> None:
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory([]))

    results = index.search("GPU", top_k=1)

    assert len(results) == 1


def test_top_k_zero_returns_empty_without_calling_encoder_or_client() -> None:
    encoder_calls: list[str] = []
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory(encoder_calls))

    assert index.search("GPU", top_k=0) == []
    assert encoder_calls == []
    assert client.calls == []


def test_top_k_negative_returns_empty_without_calling_encoder_or_client() -> None:
    encoder_calls: list[str] = []
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory(encoder_calls))

    assert index.search("GPU", top_k=-1) == []
    assert encoder_calls == []
    assert client.calls == []


# ---------------------------------------------------------------------------
# search() — empty / degenerate queries
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty_without_calling_encoder_or_client() -> None:
    encoder_calls: list[str] = []
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory(encoder_calls))

    assert index.search("", top_k=5) == []
    assert encoder_calls == []
    assert client.calls == []


def test_whitespace_only_query_returns_empty() -> None:
    client = _FakeQdrantClient(_POINTS)
    index = DenseIndex(client, _fake_encoder_factory([]))

    assert index.search("   ", top_k=5) == []


# ---------------------------------------------------------------------------
# search() — missing/partial payload
# ---------------------------------------------------------------------------


def test_missing_payload_fields_default_to_none_and_empty_text() -> None:
    client = _FakeQdrantClient([_FakePoint(None, 0.5)])
    index = DenseIndex(client, _fake_encoder_factory([]))

    results = index.search("GPU", top_k=1)

    assert results[0].chunk_id is None
    assert results[0].text == ""
    assert results[0].score == pytest.approx(0.5)


def test_payload_missing_text_key_defaults_to_empty_string() -> None:
    client = _FakeQdrantClient([_FakePoint({"chunk_id": "c9"}, 0.5)])
    index = DenseIndex(client, _fake_encoder_factory([]))

    results = index.search("GPU", top_k=1)

    assert results[0].chunk_id == "c9"
    assert results[0].text == ""
