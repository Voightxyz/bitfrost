"""``instrument_auto`` with zero detected libraries must still install the
tracer (so manual spans flow) and say so — silent no-op was the least
debuggable failure mode: zero events, zero output."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry import trace

from bitfrost.instrument import instrument_auto


class _FakeBackend:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def send(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def test_zero_detected_libraries_bootstraps_provider_and_warns(monkeypatch) -> None:
    monkeypatch.setattr("bitfrost.instrument._library_importable", lambda _name: False)
    backend = _FakeBackend()
    with pytest.warns(UserWarning, match="no supported LLM library"):
        instrumented = instrument_auto(backend=backend, agent="manual-app")
    assert instrumented == ()
    # The provider is live: a manual gen_ai span reaches the backend.
    tracer = trace.get_tracer("bootstrap-test")
    span = tracer.start_span("manual.llm", attributes={"gen_ai.system": "openai"})
    span.end()
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(5_000)
    assert any(e.get("agentId") == "manual-app" for e in backend.events)
