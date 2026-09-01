"""Tests for :mod:`bitfrost.exporter`.

Strategy
--------
The exporter is an OTel ``SpanExporter``: it accepts a batch of
``ReadableSpan`` objects, maps each one through
:func:`bitfrost.attribute_mapper.map_attributes`, stamps agent + session id,
applies the privacy filter, and delegates to its configured backend.

Coverage groups
---------------
- Empty / non-LLM batch handling (skip silently).
- Mapping success path → backend receives event with agentId, sessionId,
  privacyLevel stamped.
- Privacy levels propagate to the dispatched event.
- Per-span errors do NOT prevent later spans from exporting.
- ``shutdown`` and ``force_flush`` delegate to the backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import Status, StatusCode

from bitfrost.exporter import BitfrostExporter
from bitfrost.types import PrivacyLevel

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeBackend:
    """In-memory backend that captures every event passed to ``send``."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.shutdown_calls = 0
        self.force_flush_calls = 0
        self.force_flush_return = True
        self.raise_on_send: Exception | None = None

    def send(self, event: dict[str, Any]) -> None:
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.events.append(event)

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self.force_flush_calls += 1
        return self.force_flush_return


def _make_span(
    *,
    name: str = "openai.chat",
    attributes: dict[str, Any] | None = None,
    start_time_ns: int = 1_000_000_000,
    end_time_ns: int = 2_000_000_000,
    status_code: StatusCode = StatusCode.UNSET,
    status_description: str | None = None,
    events: tuple[Any, ...] = (),
    scope_name: str | None = "opentelemetry.instrumentation.openai.v1",
) -> Any:
    """Build a ``SimpleNamespace`` shaped like an OTel ``ReadableSpan``."""

    return SimpleNamespace(
        name=name,
        attributes=attributes or {},
        start_time=start_time_ns,
        end_time=end_time_ns,
        status=Status(status_code, status_description),
        events=events,
        instrumentation_scope=SimpleNamespace(name=scope_name, version="0.50.1")
        if scope_name
        else None,
    )


def _llm_span_attrs(**overrides: Any) -> dict[str, Any]:
    """Minimal gen_ai.* attribute set so the mapper recognises an LLM span."""

    attrs: dict[str, Any] = {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 5,
    }
    attrs.update(overrides)
    return attrs


# ---------------------------------------------------------------------------
# Empty / non-LLM batches
# ---------------------------------------------------------------------------


def test_export_returns_success_on_empty_batch() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="test-agent")
    result = exporter.export([])
    assert result is SpanExportResult.SUCCESS
    assert backend.events == []


def test_export_silently_skips_non_llm_spans() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="test-agent")
    http_span = _make_span(
        name="GET /api/users",
        attributes={"http.method": "GET", "http.status_code": 200},
    )
    result = exporter.export([http_span])
    assert result is SpanExportResult.SUCCESS
    assert backend.events == []


def test_export_returns_success_even_when_backend_raises() -> None:
    """Per-span errors must not turn the OTel pipeline into a failed export."""
    backend = _FakeBackend()
    backend.raise_on_send = RuntimeError("backend down")
    exporter = BitfrostExporter(backend=backend, agent="test-agent")
    span = _make_span(attributes=_llm_span_attrs())
    result = exporter.export([span])
    # OTel SDK retries on FAILURE; we'd rather drop than retry indefinitely.
    assert result is SpanExportResult.SUCCESS


# ---------------------------------------------------------------------------
# Mapping success path
# ---------------------------------------------------------------------------


def test_export_stamps_agent_id_on_every_event() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="my-app")
    spans = [_make_span(attributes=_llm_span_attrs()), _make_span(attributes=_llm_span_attrs())]
    exporter.export(spans)
    assert len(backend.events) == 2
    for event in backend.events:
        assert event["agentId"] == "my-app"


def test_export_stamps_session_id_under_metadata() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(
        backend=backend,
        agent="my-app",
        session_id="550e8400-e29b-41d4-a716-446655440000",
    )
    span = _make_span(attributes=_llm_span_attrs())
    exporter.export([span])
    assert len(backend.events) == 1
    assert backend.events[0]["metadata"]["sessionId"] == "550e8400-e29b-41d4-a716-446655440000"


def test_export_session_id_is_stable_across_calls() -> None:
    """Without an override, the exporter auto-generates ONE session id reused for every span."""
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="my-app")
    spans = [_make_span(attributes=_llm_span_attrs()) for _ in range(3)]
    exporter.export(spans)
    session_ids = {event["metadata"]["sessionId"] for event in backend.events}
    assert len(session_ids) == 1
    assert len(next(iter(session_ids))) == 36  # UUID v4 length


def test_export_passes_through_mapped_fields_intact() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="my-app", privacy=PrivacyLevel.FULL)
    span = _make_span(
        attributes=_llm_span_attrs(
            **{
                "gen_ai.response.model": "gpt-4o-mini-2024-07-18",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "Hello world",
                "gen_ai.completion.0.finish_reason": "stop",
            }
        ),
        status_code=StatusCode.OK,
    )
    exporter.export([span])
    [event] = backend.events
    assert event["model"] == "gpt-4o-mini-2024-07-18" or event["model"] == "gpt-4o-mini"
    assert event["outcome"] == "success"
    assert event["metadata"]["responseText"] == "Hello world"
    assert event["metadata"]["finishReason"] == "stop"


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_export_applies_standard_privacy_by_default() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="my-app")
    span = _make_span(
        attributes=_llm_span_attrs(
            **{
                "gen_ai.prompt.0.role": "user",
                "gen_ai.prompt.0.content": "email me at a@b.com",
            }
        )
    )
    exporter.export([span])
    [event] = backend.events
    assert "a@b.com" not in event["input"]["messages"][0]["content"]
    assert "[REDACTED-EMAIL]" in event["input"]["messages"][0]["content"]
    assert event["metadata"]["privacyLevel"] == "standard"


def test_export_minimal_privacy_drops_input_content() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="my-app", privacy=PrivacyLevel.MINIMAL)
    span = _make_span(
        attributes=_llm_span_attrs(
            **{
                "gen_ai.prompt.0.role": "user",
                "gen_ai.prompt.0.content": "sensitive content",
            }
        )
    )
    exporter.export([span])
    [event] = backend.events
    assert "input" not in event or not event["input"].get("messages")
    assert event["metadata"]["privacyLevel"] == "minimal"
    # Numeric metadata survives.
    assert event["metadata"]["tokens"]["input"] == 10


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_shutdown_delegates_to_backend() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="my-app")
    exporter.shutdown()
    assert backend.shutdown_calls == 1


def test_shutdown_swallows_backend_errors() -> None:
    class BrokenBackend(_FakeBackend):
        def shutdown(self) -> None:
            raise RuntimeError("shutdown failed")

    backend = BrokenBackend()
    exporter = BitfrostExporter(backend=backend, agent="my-app")
    exporter.shutdown()  # Must NOT raise.


def test_force_flush_delegates_to_backend_and_returns_its_result() -> None:
    backend = _FakeBackend()
    backend.force_flush_return = True
    exporter = BitfrostExporter(backend=backend, agent="my-app")
    assert exporter.force_flush(timeout_millis=1000) is True
    assert backend.force_flush_calls == 1

    backend.force_flush_return = False
    assert exporter.force_flush(timeout_millis=1000) is False


def test_force_flush_returns_false_when_backend_raises() -> None:
    class BrokenBackend(_FakeBackend):
        def force_flush(self, timeout_millis: int = 30000) -> bool:
            raise RuntimeError("flush failed")

    backend = BrokenBackend()
    exporter = BitfrostExporter(backend=backend, agent="my-app")
    assert exporter.force_flush(timeout_millis=1000) is False


# ---------------------------------------------------------------------------
# Error hook
# ---------------------------------------------------------------------------


def test_export_routes_per_span_errors_to_on_error_hook() -> None:
    errors: list[BaseException] = []
    backend = _FakeBackend()
    backend.raise_on_send = ValueError("bad event")
    exporter = BitfrostExporter(
        backend=backend,
        agent="my-app",
        on_error=errors.append,
    )
    span = _make_span(attributes=_llm_span_attrs())
    exporter.export([span])
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


# ---------------------------------------------------------------------------
# OTel trace context stamping
# ---------------------------------------------------------------------------


def test_trace_context_stamped_first_class() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="a")
    span = _make_span(attributes={"gen_ai.system": "openai"})
    span.context = SimpleNamespace(trace_id=0xABC123, span_id=0xDEF456)
    span.parent = SimpleNamespace(span_id=0x99)
    exporter.export([span])
    (event,) = backend.events
    assert event["traceId"] == format(0xABC123, "032x")
    assert event["spanId"] == format(0xDEF456, "016x")
    assert event["parentSpanId"] == format(0x99, "016x")


def test_trace_context_omitted_without_span_context() -> None:
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="a")
    exporter.export([_make_span(attributes={"gen_ai.system": "openai"})])
    (event,) = backend.events
    assert "traceId" not in event
    assert "spanId" not in event
    assert "parentSpanId" not in event


def test_zeroed_trace_context_not_shipped() -> None:
    # A no-op tracer hands out all-zero ids — shipping them would group
    # unrelated events into one fake trace.
    backend = _FakeBackend()
    exporter = BitfrostExporter(backend=backend, agent="a")
    span = _make_span(attributes={"gen_ai.system": "openai"})
    span.context = SimpleNamespace(trace_id=0, span_id=0)
    span.parent = None
    exporter.export([span])
    (event,) = backend.events
    assert "traceId" not in event
