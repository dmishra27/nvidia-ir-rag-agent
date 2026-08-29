"""GET /health (liveness) and GET /health/ready (readiness).

`/health` is a pure liveness probe — no DB/Qdrant/model calls. It answers
"is this process up", nothing more, so its latency and availability do not
track the slowest downstream service. The compose `api` healthcheck and
`render.yaml` healthCheckPath both point here, so it must stay cheap and
must not start failing when a degradable dependency is down.

`/health/ready` (F-16 in docs/uat/clean_clone_test_findings.md) is the
readiness probe that `/health` deliberately is not: it asserts the
dependencies a request actually needs — Postgres reachable, and the
committed BM25 index loadable — and returns 503 with a per-dependency
`checks` map when one is down. It was added rather than folded into
`/health` precisely so container startup ordering keeps keying off
liveness. Qdrant/dense is not asserted: a dense failure degrades to
BM25-only and still serves (F-14), so it is not a readiness condition.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from api.dependencies import check_bm25, check_postgres
from api.schemas import HealthResponse, ReadinessResponse

router = APIRouter()

_SERVICE = "nvidia-ir-rag-agent"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=_SERVICE)


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(
    response: Response,
    postgres: str = Depends(check_postgres),
    bm25: str = Depends(check_bm25),
) -> ReadinessResponse:
    checks = {"postgres": postgres, "bm25": bm25}
    ready = all(result == "ok" for result in checks.values())
    if not ready:
        response.status_code = 503
    return ReadinessResponse(
        status="ready" if ready else "not ready",
        service=_SERVICE,
        checks=checks,
    )
