"""Unit tests for retrieval/reranker_cohere.py.

Per AGENTS.md ("Mock all embedding and LLM calls in unit tests"), these
tests never construct a real cohere.Client or make a network call. A fake
CohereRerankClientProtocol records the rerank() call it was asked to make
and returns a scripted result list, standing in for the real Cohere API.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from retrieval.candidates import Candidate
from retrieval.reranker_cohere import CohereReranker


@dataclass
class _FakeResult:
    index: int
    relevance_score: float


class _FakeCohereClient:
    """Records every rerank() call and returns a scripted result list."""

    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def rerank(self, query: str, documents: list[str], model: str, top_n: int) -> list[_FakeResult]:
        self.calls.append({"query": query, "documents": documents, "model": model, "top_n": top_n})
        return self._results


def _c(chunk_id: str, text: str, score: float = 1.0, rank: int = 1) -> Candidate:
    return Candidate(chunk_id=chunk_id, text=text, score=score, rank=rank)


_CANDIDATES = [
    _c("c1", "cudaStreamAddCallback note.", score=11.1, rank=1),
    _c("c2", "cudaMalloc allocates device memory.", score=9.5, rank=2),
    _c("c3", "cudaFree releases device memory.", score=8.2, rank=3),
]


# ---------------------------------------------------------------------------
# rerank() — happy path / re-ordering
# ---------------------------------------------------------------------------


def test_rerank_reorders_by_cohere_result_order() -> None:
    # Cohere returns results already sorted by relevance, referencing candidates by index.
    fake = _FakeCohereClient(
        [
            _FakeResult(index=1, relevance_score=0.9),
            _FakeResult(index=2, relevance_score=0.5),
            _FakeResult(index=0, relevance_score=0.2),
        ]
    )
    reranker = CohereReranker(fake)

    results = reranker.rerank("cudaMalloc parameters", _CANDIDATES, top_k=3)

    assert [r.chunk_id for r in results] == ["c2", "c3", "c1"]


def test_rerank_scores_are_cohere_relevance_scores() -> None:
    fake = _FakeCohereClient([_FakeResult(index=1, relevance_score=0.9), _FakeResult(index=0, relevance_score=0.2)])
    reranker = CohereReranker(fake)

    results = reranker.rerank("query", _CANDIDATES, top_k=2)

    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.2)


def test_rerank_ranks_are_sequential_starting_at_one() -> None:
    fake = _FakeCohereClient(
        [_FakeResult(index=0, relevance_score=0.1), _FakeResult(index=1, relevance_score=0.2)]
    )
    reranker = CohereReranker(fake)

    results = reranker.rerank("query", _CANDIDATES, top_k=2)

    assert [r.rank for r in results] == [1, 2]


def test_rerank_preserves_chunk_id_and_text() -> None:
    fake = _FakeCohereClient([_FakeResult(index=0, relevance_score=0.9)])
    reranker = CohereReranker(fake)

    results = reranker.rerank("query", _CANDIDATES, top_k=1)

    assert results[0].chunk_id == "c1"
    assert results[0].text == "cudaStreamAddCallback note."


# ---------------------------------------------------------------------------
# rerank() — request shape sent to Cohere
# ---------------------------------------------------------------------------


def test_rerank_sends_query_documents_model_and_top_n() -> None:
    fake = _FakeCohereClient([_FakeResult(index=0, relevance_score=0.1)])
    reranker = CohereReranker(fake, model_id="rerank-v3.5")

    reranker.rerank("cudaMalloc parameters", _CANDIDATES, top_k=2)

    assert fake.calls == [
        {
            "query": "cudaMalloc parameters",
            "documents": [
                "cudaStreamAddCallback note.",
                "cudaMalloc allocates device memory.",
                "cudaFree releases device memory.",
            ],
            "model": "rerank-v3.5",
            "top_n": 2,
        }
    ]


def test_rerank_top_n_never_exceeds_candidate_pool_size() -> None:
    fake = _FakeCohereClient([_FakeResult(index=0, relevance_score=0.1)])
    reranker = CohereReranker(fake)

    reranker.rerank("query", _CANDIDATES, top_k=100)

    assert fake.calls[0]["top_n"] == 3


# ---------------------------------------------------------------------------
# rerank() — boundary conditions
# ---------------------------------------------------------------------------


def test_rerank_top_k_zero_returns_empty_without_calling_client() -> None:
    fake = _FakeCohereClient([_FakeResult(index=0, relevance_score=0.1)])
    reranker = CohereReranker(fake)

    assert reranker.rerank("query", _CANDIDATES, top_k=0) == []
    assert fake.calls == []


def test_rerank_top_k_negative_returns_empty_without_calling_client() -> None:
    fake = _FakeCohereClient([_FakeResult(index=0, relevance_score=0.1)])
    reranker = CohereReranker(fake)

    assert reranker.rerank("query", _CANDIDATES, top_k=-1) == []
    assert fake.calls == []


def test_rerank_empty_candidates_returns_empty_without_calling_client() -> None:
    fake = _FakeCohereClient([])
    reranker = CohereReranker(fake)

    assert reranker.rerank("query", [], top_k=5) == []
    assert fake.calls == []
