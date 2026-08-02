"""ASGI middleware measuring per-request latency and writing it to
request_log (Postgres) via SQLAlchemy ORM — per AGENTS.md's rule to never
use raw SQL strings for database writes.

`stage` is the request_log column that groups these rows for querying
(e.g. "search" vs "ask" vs "health") — this middleware records one
request-level latency row per HTTP call; fine-grained per-LangGraph-node
timing (retrieve/rerank/generate) is deferred to the Layer 6 monitoring/
module, since that requires threading timers through AgentState/QAState
rather than anything observable at the ASGI layer.

The session factory is constructor-injected (not imported globally) so
unit tests can supply a fake and assert on what would have been written,
without a live Postgres connection — mirroring the injection pattern used
for embedding/index components elsewhere in the project. A failed DB write
is logged and swallowed: monitoring must never break the actual response.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from schema.models import RequestLog, get_engine, get_session_factory

log = structlog.get_logger()


class RequestLatencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, session_factory: Callable[[], Any] | None = None) -> None:
        super().__init__(app)
        self._session_factory = session_factory or get_session_factory(get_engine())

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        query_id = getattr(request.state, "query_id", None)
        reranker_config = getattr(request.state, "reranker_config", None)
        stage = request.url.path.strip("/").split("/")[0] or "root"

        log.info(
            "request_latency",
            query_id=query_id or "unknown",
            stage=stage,
            endpoint=request.url.path,
            duration_ms=round(duration_ms, 2),
            status_code=response.status_code,
        )
        self._write_request_log(
            query_id=query_id,
            endpoint=request.url.path,
            reranker_config=reranker_config,
            stage=stage,
            duration_ms=duration_ms,
            status_code=response.status_code,
        )
        return response

    def _write_request_log(
        self,
        query_id: str | None,
        endpoint: str,
        reranker_config: str | None,
        stage: str,
        duration_ms: float,
        status_code: int,
    ) -> None:
        try:
            with self._session_factory() as session:
                session.add(
                    RequestLog(
                        request_id=str(uuid.uuid4()),
                        query_id=query_id,
                        endpoint=endpoint,
                        reranker_config=reranker_config,
                        stage=stage,
                        duration_ms=duration_ms,
                        status_code=status_code,
                    )
                )
                session.commit()
        except Exception as exc:
            log.warning("request_log_write_failed", stage=stage, exc=str(exc))
