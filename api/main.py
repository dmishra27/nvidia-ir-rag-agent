"""FastAPI app: /, /evaluation, /search, /ask, /health.

`create_app` is a factory (rather than only a module-level singleton) so
tests can inject a fake `session_factory` into RequestLatencyMiddleware —
this avoids a live Postgres connection in unit tests while still letting
`uvicorn api.main:app` run the real thing.

`/` serves api/routers/ui.py's guided HTML search page and `/evaluation`
serves api/routers/evaluation.py's static retrieval-quality report; `/docs`
stays at FastAPI's default Swagger UI location (both routes are registered
with include_in_schema=False so neither shows up as an operation there).

The lifespan handler calls api/telemetry.py's `configure_tracing()` once at
startup, gated by ENABLE_TRACING (default off) so `trace.get_tracer()`
keeps returning OTel's no-op tracer -- and every `traced_stage` span in
agents/retrieval_agent.py and agents/qa_agent.py stays inert -- unless a
Jaeger collector is actually expected to be listening. This keeps
TestClient(create_app(...)) and every existing contract test unaffected,
since ENABLE_TRACING is unset in the test environment. OTLP_ENDPOINT
overrides where spans get exported (see api/telemetry.py's
DEFAULT_OTLP_ENDPOINT).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from dotenv import load_dotenv
from fastapi import FastAPI

from api.telemetry import DEFAULT_OTLP_ENDPOINT, configure_tracing
from monitoring.phoenix_config import DEFAULT_PHOENIX_ENDPOINT, configure_phoenix

from api.middleware import RequestLatencyMiddleware
from api.routers import ask, evaluation, health, search, ui

load_dotenv()

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if os.environ.get("ENABLE_TRACING", "").lower() in ("1", "true", "yes"):
        configure_tracing(otlp_endpoint=os.environ.get("OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT))
    if os.environ.get("ENABLE_PHOENIX", "").lower() in ("1", "true", "yes"):
        configure_phoenix(endpoint=os.environ.get("PHOENIX_ENDPOINT", DEFAULT_PHOENIX_ENDPOINT))
    yield


def create_app(session_factory: Callable[[], Any] | None = None) -> FastAPI:
    app = FastAPI(title="nvidia-ir-rag-agent", version="1.0.0", lifespan=_lifespan)
    app.add_middleware(RequestLatencyMiddleware, session_factory=session_factory)
    app.include_router(ui.router)
    app.include_router(evaluation.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(health.router)
    return app


app = create_app()
