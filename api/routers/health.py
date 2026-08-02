"""GET /health — liveness check.

Intentionally a pure liveness probe (no DB/Qdrant/model calls): those
dependencies are already exercised, with graceful degradation, by /search
and /ask, and a health check that reaches out to them would make this
endpoint's latency and availability track the slowest downstream service
instead of "is this process up."
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="nvidia-ir-rag-agent")
