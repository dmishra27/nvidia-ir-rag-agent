"""Unit tests for api/telemetry.py.

`traced_stage` is verified against a real TracerProvider wired to an
InMemorySpanExporter (opentelemetry-sdk's in-process test double), injected
directly via the `tracer` parameter — no Jaeger collector, no network call,
no OTLP export, and no reliance on OTel's process-global TracerProvider
(which can only be set once, making it unsuitable for per-test isolation).
`configure_tracing` only constructs objects and registers a provider; it is
not exercised against a live Jaeger endpoint here per AGENTS.md's
mock-everything-in-tests rule.
"""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from api.telemetry import SERVICE_NAME, traced_stage


def _tracer_with_exporter() -> tuple[object, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(SERVICE_NAME), exporter


def test_traced_stage_emits_one_span_named_for_the_stage() -> None:
    tracer, exporter = _tracer_with_exporter()

    with traced_stage("bm25", "q1", tracer=tracer):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "bm25"


def test_traced_stage_sets_query_id_and_stage_attributes() -> None:
    tracer, exporter = _tracer_with_exporter()

    with traced_stage("dense", "q42", tracer=tracer):
        pass

    span = exporter.get_finished_spans()[0]
    assert span.attributes["query_id"] == "q42"
    assert span.attributes["stage"] == "dense"


def test_traced_stage_sets_extra_attributes() -> None:
    tracer, exporter = _tracer_with_exporter()

    with traced_stage("rerank", "q1", tracer=tracer, top_k=10, reranker_config="live_fast"):
        pass

    span = exporter.get_finished_spans()[0]
    assert span.attributes["top_k"] == 10
    assert span.attributes["reranker_config"] == "live_fast"


def test_traced_stage_yields_the_span() -> None:
    tracer, _ = _tracer_with_exporter()

    with traced_stage("rrf", "q1", tracer=tracer) as span:
        assert span is not None


def test_traced_stage_propagates_exceptions_raised_inside() -> None:
    tracer, exporter = _tracer_with_exporter()

    raised = False
    try:
        with traced_stage("llm", "q1", tracer=tracer):
            raise RuntimeError("boom")
    except RuntimeError as exc:
        raised = True
        assert str(exc) == "boom"

    assert raised
    # the span still gets recorded (as an error span) even though it raised
    assert len(exporter.get_finished_spans()) == 1


def test_traced_stage_defaults_to_global_tracer_when_none_injected() -> None:
    with traced_stage("bm25", "q1") as span:
        assert span is not None
