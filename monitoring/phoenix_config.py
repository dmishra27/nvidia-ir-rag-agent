"""Arize Phoenix instrumentation for the LangGraph agents
(agents/orchestrator.py, agents/qa_agent.py, agents/retrieval_agent.py,
agents/eval_agent.py) -- Layer 7's remaining observability tool per
AGENTS.md ("LangSmith, OpenTelemetry+Jaeger, structlog, Arize Phoenix").

Phoenix's value here is complementary to api/telemetry.py's Jaeger spans,
not a duplicate of them: `openinference.instrumentation.langchain`'s
`LangChainInstrumentor` patches langchain-core's `Runnable.invoke`/`ainvoke`
directly, and every agent's `build_graph()` compiles a `StateGraph` into a
langchain-core `Runnable` -- so instrumenting once, at process startup,
gives Phoenix a full per-node run tree (with prompts/completions/token
counts via OpenInference semantic conventions) for every `graph.invoke()`
call already made by `agents/*.py::run()`, with zero changes to agent code.
This mirrors how `api/telemetry.py`'s `traced_stage()` needed no
LangGraph-specific integration either, just injected at the call sites --
Phoenix instruments at the framework layer instead, one level up.

`configure_phoenix()` is the process-startup call (call once, e.g. from
api/main.py alongside `configure_tracing()`): it registers a Phoenix
TracerProvider pointed at the local Phoenix collector (port 6006, see
docker-compose.yml's `phoenix` service) and instruments LangChain globally.
Like `api/telemetry.py`'s `configure_tracing()`, it is never called at
import time or from unit tests -- an uninstrumented process just doesn't
emit spans, so every existing agent contract test keeps passing unchanged.

`instrument_langchain()`/`uninstrument_langchain()` take an injectable
`tracer_provider` so tests wire a real `TracerProvider` bound to an
`InMemorySpanExporter` (opentelemetry-sdk's in-process test double) instead
of a live Phoenix collector, per AGENTS.md's mock-everything-in-tests rule
-- and always uninstrument again afterward, since `LangChainInstrumentor`
patches langchain-core globally (like OTel's TracerProvider, it isn't
per-instance) and a test that forgot to undo it would leak spans into every
later test that happens to invoke a Runnable.
"""

from __future__ import annotations

from typing import Any

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider

SERVICE_NAME = "nvidia-ir-rag-agent"
PHOENIX_PROJECT_NAME = "nvidia-ir-rag-agent"
DEFAULT_PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"

_instrumentor = LangChainInstrumentor()


def configure_phoenix(
    endpoint: str = DEFAULT_PHOENIX_ENDPOINT, project_name: str = PHOENIX_PROJECT_NAME
) -> TracerProvider:
    """Register a Phoenix-bound TracerProvider (OTLP/HTTP to the local
    Phoenix collector) and instrument LangChain/LangGraph globally. Returns
    the provider so callers can also hand it to api/telemetry.py-style
    manual spans if they want Jaeger and Phoenix to share one provider."""
    from phoenix.otel import register

    tracer_provider = register(
        endpoint=endpoint,
        project_name=project_name,
        protocol="http/protobuf",
        batch=True,
        set_global_tracer_provider=True,
    )
    instrument_langchain(tracer_provider)
    return tracer_provider


def instrument_langchain(tracer_provider: TracerProvider | None = None) -> LangChainInstrumentor:
    """Patch langchain-core's Runnable.invoke/ainvoke to emit OpenInference
    spans to `tracer_provider` (the global provider if omitted). No-op if
    already instrumented -- safe to call more than once."""
    if not _instrumentor.is_instrumented_by_opentelemetry:
        _instrumentor.instrument(tracer_provider=tracer_provider)
    return _instrumentor


def uninstrument_langchain() -> None:
    """Undo `instrument_langchain()`. No-op if not currently instrumented."""
    if _instrumentor.is_instrumented_by_opentelemetry:
        _instrumentor.uninstrument()


def is_instrumented() -> bool:
    return bool(_instrumentor.is_instrumented_by_opentelemetry)


def phoenix_span_kind_attributes(kind: str, **extra: Any) -> dict[str, Any]:
    """OpenInference span-kind attribute dict (e.g. `{"openinference.span.kind":
    "RETRIEVER"}`) for manually tagging a span with Phoenix's semantic
    conventions -- useful anywhere a stage isn't already a Runnable
    (e.g. inside a plain function) but should still show up typed correctly
    in Phoenix's UI. `kind` is one of Phoenix's span kinds: CHAIN, RETRIEVER,
    RERANKER, LLM, TOOL, EMBEDDING, AGENT."""
    return {"openinference.span.kind": kind, **extra}
