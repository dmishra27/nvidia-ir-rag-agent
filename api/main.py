"""FastAPI app: /search, /ask, /health.

`create_app` is a factory (rather than only a module-level singleton) so
tests can inject a fake `session_factory` into RequestLatencyMiddleware —
this avoids a live Postgres connection in unit tests while still letting
`uvicorn api.main:app` run the real thing.
"""

from __future__ import annotations

from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI

from api.middleware import RequestLatencyMiddleware
from api.routers import ask, health, search

load_dotenv()


def create_app(session_factory: Callable[[], Any] | None = None) -> FastAPI:
    app = FastAPI(title="nvidia-ir-rag-agent")
    app.add_middleware(RequestLatencyMiddleware, session_factory=session_factory)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(health.router)
    return app


app = create_app()
