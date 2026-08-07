"""Unit tests for monitoring/phoenix_config.py.

`instrument_langchain`/`uninstrument_langchain` are verified against a real
`TracerProvider` wired to an `InMemorySpanExporter` (opentelemetry-sdk's
in-process test double), injected directly -- no live Phoenix collector, no
network call, no OTLP export -- mirroring tests/api/test_telemetry.py's
approach for Jaeger. `configure_phoenix` (which calls the real
`phoenix.otel.register()` against a live collector) is not exercised here
per AGENTS.md's mock-everything-in-tests rule, matching how
`api/telemetry.py::configure_tracing()` is also left untested against a live
Jaeger endpoint.

Every test that instruments LangChain uninstruments it again in a `finally`
(or via the `_uninstrumented` fixture) -- `LangChainInstrumentor` patches
langchain-core globally, so a leaked instrumentation would bleed spans into
unrelated tests that happen to invoke a Runnable.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monitoring.phoenix_config import (
    is_instrumented,
    instrument_langchain,
    phoenix_span_kind_attributes,
    uninstrument_langchain,
)


def _provider_with_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture(autouse=True)
def _uninstrumented():
    """Guarantee every test starts and ends with LangChain uninstrumented,
    even if a test body raises before reaching its own cleanup."""
    uninstrument_langchain()
    yield
    uninstrument_langchain()


def test_not_instrumented_by_default() -> None:
    assert is_instrumented() is False


def test_instrument_langchain_marks_instrumented() -> None:
    provider, _ = _provider_with_exporter()

    instrument_langchain(tracer_provider=provider)

    assert is_instrumented() is True


def test_instrument_langchain_captures_runnable_invoke_as_a_span() -> None:
    provider, exporter = _provider_with_exporter()
    instrument_langchain(tracer_provider=provider)

    chain = RunnableLambda(lambda x: x + 1)
    result = chain.invoke(1)

    spans = exporter.get_finished_spans()
    assert result == 2
    assert len(spans) == 1
    assert spans[0].name == "RunnableLambda"


def test_instrument_langchain_is_idempotent() -> None:
    provider, exporter = _provider_with_exporter()

    instrument_langchain(tracer_provider=provider)
    instrument_langchain(tracer_provider=provider)  # second call is a no-op
    RunnableLambda(lambda x: x).invoke(1)

    assert len(exporter.get_finished_spans()) == 1


def test_uninstrument_langchain_stops_emitting_spans() -> None:
    provider, exporter = _provider_with_exporter()
    instrument_langchain(tracer_provider=provider)

    uninstrument_langchain()
    RunnableLambda(lambda x: x).invoke(1)

    assert is_instrumented() is False
    assert len(exporter.get_finished_spans()) == 0


def test_uninstrument_langchain_is_idempotent() -> None:
    uninstrument_langchain()  # already uninstrumented -- must not raise
    assert is_instrumented() is False


def test_phoenix_span_kind_attributes_sets_openinference_kind() -> None:
    attrs = phoenix_span_kind_attributes("RETRIEVER", query_id="q1")

    assert attrs == {"openinference.span.kind": "RETRIEVER", "query_id": "q1"}
