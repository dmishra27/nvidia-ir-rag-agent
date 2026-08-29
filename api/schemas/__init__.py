"""Pydantic v2 request/response schemas for api/routers/*."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from retrieval.candidates import Candidate


class CandidateOut(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> "CandidateOut":
        return cls(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            score=candidate.score,
            rank=candidate.rank,
        )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    candidate_pool_size: int = 100


class SearchResponse(BaseModel):
    query_id: str
    query: str
    reranker_mode: str | None = None
    results: list[CandidateOut]
    error: str | None = None


class AskRequest(BaseModel):
    query: str
    top_k: int = 10
    candidate_pool_size: int = 100


class CitationOut(BaseModel):
    claim: str
    chunk_ids: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    query_id: str
    query: str
    reranker_mode: str | None = None
    answer: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadinessResponse(BaseModel):
    """GET /health/ready — dependency assertion (F-16).

    `status` is "ready" only when every required dependency answers;
    "not ready" (HTTP 503) otherwise. `checks` carries the per-dependency
    result ("ok" or "error: <detail>") so a 503 says *which* dependency is
    down rather than just that one is.
    """

    status: Literal["ready", "not ready"]
    service: str
    checks: dict[str, str]


class FeedbackRequest(BaseModel):
    query_id: str | None = None
    chunk_id: str | None = None
    reaction: Literal["up", "down"]
    user_id: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
