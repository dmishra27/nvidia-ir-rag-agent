"""OpenTelemetry span instrumentation: one span per retrieval-pipeline
stage — BM25, dense, RRF, rerank, LLM — per SKILLS.md's add-otel-span
pattern.

`configure_tracing()` wires the global TracerProvider to export to Jaeger
over OTLP (call once at process startup, e.g. api/main.py); it is never
called by the agent modules or by tests. Without it, `trace.get_tracer(...)`
returns OTel's default no-op tracer — `traced_stage` spans become inert
(no export, no network call, no Jaeger dependency), which is exactly what
keeps agents/retrieval_agent.py's and agents/qa_agent.py's existing contract
tests passing unchanged after instrumentation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span

SERVICE_NAME = "nvidia-ir-rag-agent"
DEFAULT_OTLP_ENDPOINT = "http://localhost:4317"


def configure_tracing(otlp_endpoint: str = DEFAULT_OTLP_ENDPOINT) -> None:
    """Set the global TracerProvider to export spans to Jaeger over OTLP.

    BatchSpanProcessor exports asynchronously and drops failed batches
    rather than raising, so a Jaeger collector being unreachable degrades
    silently instead of breaking request handling.
    """
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)))
    trace.set_tracer_provider(provider)


@contextmanager
def traced_stage(
    stage: str, query_id: str, tracer: trace.Tracer | None = None, **attributes: Any
) -> Iterator[Span]:
    """One span per pipeline stage (bm25 | dense | rrf | rerank | llm),
    tagged with query_id + stage per AGENTS.md's structlog convention so
    OTel spans and structlog lines correlate on the same fields. Exceptions
    raised inside the block propagate through unchanged — callers keep
    their own try/except around this exactly as before instrumentation.

    `tracer` is injectable (defaults to the global provider's tracer) so
    tests can pass a tracer bound to an InMemorySpanExporter directly —
    OTel's global TracerProvider can only be set once per process, which
    makes it unsuitable for per-test isolation.
    """
    tracer = tracer or trace.get_tracer(SERVICE_NAME)
    with tracer.start_as_current_span(stage) as span:
        span.set_attribute("query_id", query_id)
        span.set_attribute("stage", stage)
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span
