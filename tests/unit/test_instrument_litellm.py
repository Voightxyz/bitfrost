"""Tests for the LiteLLM CustomLogger adapter in :mod:`bitfrost.instrument`.

LiteLLM's own ``OpenTelemetry`` integration only activates inside their
Proxy Server. For SDK use Bitfrost ships ``_BitfrostLiteLLMHandler``, a
``CustomLogger`` subclass that translates LiteLLM callback events into
OTel spans with semconv v1.32+ ``gen_ai.*`` attributes.

These tests target the adapter's pure logic (``_emit_litellm_span``) with
a mock tracer + fake LiteLLM ``ModelResponse``, plus the dynamic
subclass construction and the callback-list dedup. The Stage 4 smoke
covers the real end-to-end path; these pin the unit-level contract so a
refactor can't silently break the attribute mapping.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any
from unittest.mock import MagicMock

from bitfrost import instrument as _mod

# ---------------------------------------------------------------------------
# Fakes mirroring LiteLLM's callback shapes
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChoice:
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason


class _FakeResponse:
    """Mirror of litellm.ModelResponse — only the fields the adapter reads."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        usage: _FakeUsage | None = None,
        choices: list[_FakeChoice] | None = None,
    ) -> None:
        self.model = model
        self.usage = usage
        self.choices = choices or []


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "custom_llm_provider": "openai",
        "messages": [{"role": "user", "content": "ping"}],
    }
    base.update(overrides)
    return base


def _times() -> tuple[_dt.datetime, _dt.datetime]:
    start = _dt.datetime(2026, 5, 28, 12, 0, 0)
    end = _dt.datetime(2026, 5, 28, 12, 0, 1)  # +1s
    return start, end


def _collect_attrs(span: MagicMock) -> dict[str, Any]:
    """Flatten a mock span's set_attribute calls into a dict."""

    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


# ---------------------------------------------------------------------------
# _emit_litellm_span — core mapping
# ---------------------------------------------------------------------------


def test_emit_span_sets_genai_v132_attributes() -> None:
    tracer = MagicMock()
    span = MagicMock()
    tracer.start_span.return_value = span
    start, end = _times()

    _mod._emit_litellm_span(
        tracer=tracer,
        kwargs=_kwargs(),
        response_obj=_FakeResponse(
            usage=_FakeUsage(12, 5),
            choices=[_FakeChoice("stop")],
        ),
        start_time=start,
        end_time=end,
        success=True,
    )

    tracer.start_span.assert_called_once()
    assert tracer.start_span.call_args.args[0] == "litellm.completion"
    attrs = _collect_attrs(span)
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    span.end.assert_called_once()


def test_emit_span_provider_defaults_to_litellm_when_absent() -> None:
    """No ``custom_llm_provider`` → provider falls back to ``"litellm"``."""

    tracer = MagicMock()
    span = MagicMock()
    tracer.start_span.return_value = span
    start, end = _times()

    _mod._emit_litellm_span(
        tracer=tracer,
        kwargs=_kwargs(custom_llm_provider=None),
        response_obj=_FakeResponse(usage=_FakeUsage(1, 1)),
        start_time=start,
        end_time=end,
        success=True,
    )
    attrs = _collect_attrs(span)
    assert attrs["gen_ai.provider.name"] == "litellm"


def test_emit_span_sets_error_status_on_failure() -> None:
    from opentelemetry.trace import StatusCode

    tracer = MagicMock()
    span = MagicMock()
    tracer.start_span.return_value = span
    start, end = _times()

    _mod._emit_litellm_span(
        tracer=tracer,
        kwargs=_kwargs(),
        response_obj=_FakeResponse(),
        start_time=start,
        end_time=end,
        success=False,
    )

    # set_status called once with a Status carrying ERROR code.
    span.set_status.assert_called_once()
    status_arg = span.set_status.call_args.args[0]
    assert status_arg.status_code == StatusCode.ERROR


def test_emit_span_handles_missing_usage_without_crashing() -> None:
    """A response with no ``usage`` (e.g. streaming early-fail) maps cleanly."""

    tracer = MagicMock()
    span = MagicMock()
    tracer.start_span.return_value = span
    start, end = _times()

    _mod._emit_litellm_span(
        tracer=tracer,
        kwargs=_kwargs(),
        response_obj=_FakeResponse(usage=None, choices=[]),
        start_time=start,
        end_time=end,
        success=True,
    )
    attrs = _collect_attrs(span)
    # No token attrs emitted, but the span still ended cleanly.
    assert "gen_ai.usage.input_tokens" not in attrs
    span.end.assert_called_once()


def test_emit_span_falls_back_to_now_when_timestamps_not_datetimes() -> None:
    """Non-datetime start/end (None) must not crash — fall back to wall clock."""

    tracer = MagicMock()
    span = MagicMock()
    tracer.start_span.return_value = span

    _mod._emit_litellm_span(
        tracer=tracer,
        kwargs=_kwargs(),
        response_obj=_FakeResponse(usage=_FakeUsage(1, 1)),
        start_time=None,
        end_time=None,
        success=True,
    )
    # start_span still called with a positive integer start_time.
    start_ns = tracer.start_span.call_args.kwargs["start_time"]
    assert isinstance(start_ns, int) and start_ns > 0
    span.end.assert_called_once()


# ---------------------------------------------------------------------------
# Dynamic CustomLogger subclass + delegation
# ---------------------------------------------------------------------------


def test_handler_construction_builds_customlogger_subclass() -> None:
    """The factory returns a real CustomLogger subclass with the marker."""

    from litellm.integrations.custom_logger import CustomLogger

    handler_cls = _mod._get_litellm_handler_class()
    handler = handler_cls(MagicMock())
    assert isinstance(handler, CustomLogger)
    assert getattr(handler, "_is_bitfrost_handler", False) is True
    for method in (
        "log_success_event",
        "log_failure_event",
        "async_log_success_event",
        "async_log_failure_event",
    ):
        assert hasattr(handler, method)


def test_factory_caches_the_class_across_calls() -> None:
    """``_get_litellm_handler_class`` returns the SAME class object each call.

    Caching matters so handler instances share one type — otherwise the
    marker-based dedup would still work but isinstance debugging would be
    confusing and every call would rebuild the class.
    """

    first = _mod._get_litellm_handler_class()
    second = _mod._get_litellm_handler_class()
    assert first is second


def test_handler_log_success_event_delegates_to_emit(monkeypatch: Any) -> None:
    """The handler's ``log_success_event`` calls ``_emit_litellm_span`` with success=True."""

    captured: dict[str, Any] = {}

    def fake_emit(**kw: Any) -> None:
        captured.update(kw)

    monkeypatch.setattr(_mod, "_emit_litellm_span", fake_emit)
    tracer = MagicMock()
    handler = _mod._get_litellm_handler_class()(tracer)
    start, end = _times()
    handler.log_success_event(_kwargs(), _FakeResponse(), start, end)

    assert captured["success"] is True
    assert captured["tracer"] is tracer


def test_handler_log_failure_event_delegates_with_success_false(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        _mod, "_emit_litellm_span", lambda **kw: captured.update(kw)
    )
    handler = _mod._get_litellm_handler_class()(MagicMock())
    start, end = _times()
    handler.log_failure_event(_kwargs(), _FakeResponse(), start, end)
    assert captured["success"] is False


# ---------------------------------------------------------------------------
# instrument_litellm — callback registration + dedup
# ---------------------------------------------------------------------------


def test_instrument_litellm_registers_handler_in_callbacks() -> None:
    import litellm

    # Reset state so this test is deterministic.
    _mod._BITFROST_OWNED_PROVIDER = None
    _mod._PROCESSORS_REGISTERED.clear()
    litellm.callbacks = []

    from bitfrost.backends.console import ConsoleBackend

    _mod.instrument_litellm(backend=ConsoleBackend(), agent="t")
    handlers = [
        c for c in litellm.callbacks if getattr(c, "_is_bitfrost_handler", False)
    ]
    assert len(handlers) == 1


def test_instrument_litellm_does_not_double_register_on_repeat() -> None:
    """Calling instrument_litellm twice must not append two handlers.

    This is the dedup bug the unit tests caught: the original __new__
    dynamic-subclass approach made isinstance(handler, _BitfrostLiteLLMHandler)
    always False, so the dedup never matched and a second call appended
    a duplicate. The marker-attribute dedup fixes it.
    """

    import litellm

    _mod._BITFROST_OWNED_PROVIDER = None
    _mod._PROCESSORS_REGISTERED.clear()
    litellm.callbacks = []

    from bitfrost.backends.console import ConsoleBackend

    backend = ConsoleBackend()
    _mod.instrument_litellm(backend=backend, agent="t")
    _mod.instrument_litellm(backend=backend, agent="t")
    handlers = [
        c for c in litellm.callbacks if getattr(c, "_is_bitfrost_handler", False)
    ]
    assert len(handlers) == 1
