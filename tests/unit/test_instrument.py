"""Tests for :mod:`bitfrost.instrument`.

The auto-instrument helpers are the user-facing surface — these tests
pin the contracts that the launch demo relies on:

- TracerProvider augmentation never clobbers (Sentry/Datadog/etc keep working)
- Re-calling a helper is idempotent (no duplicate processors)
- A second helper in the same process attaches to the SAME provider
- ``instrument_auto`` skips libraries that aren't installed
- Late-import emits a UserWarning, doesn't raise
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from bitfrost import instrument as _instrument_module


@pytest.fixture(autouse=True)
def _reset_otel_state() -> Any:
    """Reset module + OTel state between tests so each starts clean."""

    # OTel's global tracer provider is process-wide; tests must reset it
    # to avoid one test's setup leaking into the next.
    _instrument_module._BITFROST_OWNED_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    yield
    _instrument_module._BITFROST_OWNED_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TracerProvider bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_creates_provider_when_none_exists() -> None:
    """Fresh process: bootstrap installs a new TracerProvider."""

    from bitfrost.backends.console import ConsoleBackend

    _instrument_module._bootstrap_provider(
        backend=ConsoleBackend(), agent="t", session_id=None, privacy="standard", on_error=None
    )
    assert isinstance(trace.get_tracer_provider(), TracerProvider)


def test_bootstrap_reuses_existing_provider_set_by_user() -> None:
    """A user's pre-configured TracerProvider must NOT be clobbered.

    Ecosystem-friendliness: if the host app already wired Sentry / Datadog
    / Langfuse, calling instrument_openai shouldn't replace their provider.
    """

    user_provider = TracerProvider()
    trace.set_tracer_provider(user_provider)
    from bitfrost.backends.console import ConsoleBackend

    _instrument_module._bootstrap_provider(
        backend=ConsoleBackend(), agent="t", session_id=None, privacy="standard", on_error=None
    )
    assert trace.get_tracer_provider() is user_provider


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_two_helpers_share_one_provider() -> None:
    """Calling instrument_openai THEN instrument_anthropic uses the same provider.

    Reduces resource usage + avoids the user surprise of "why do I have
    two TracerProviders now?". The late-import warning is irrelevant
    here and explicitly silenced — its behaviour is covered in the
    dedicated late-import tests below.
    """

    from bitfrost.backends.console import ConsoleBackend

    with patch("opentelemetry.instrumentation.openai.OpenAIInstrumentor") as mock_openai, patch(
        "opentelemetry.instrumentation.anthropic.AnthropicInstrumentor"
    ) as mock_anthropic:
        _instrument_module.instrument_openai(backend=ConsoleBackend())
        first = trace.get_tracer_provider()
        _instrument_module.instrument_anthropic(backend=ConsoleBackend())
        second = trace.get_tracer_provider()

    assert first is second
    mock_openai.return_value.instrument.assert_called_once()
    mock_anthropic.return_value.instrument.assert_called_once()


# ---------------------------------------------------------------------------
# Late-import warning (D5)
# ---------------------------------------------------------------------------


def test_warns_when_target_library_already_imported() -> None:
    """Calling instrument_X after `import X` emits UserWarning, doesn't raise."""

    sys.modules["openai"] = MagicMock()  # simulate "already imported"
    try:
        with patch("opentelemetry.instrumentation.openai.OpenAIInstrumentor") as mock_inst:
            with pytest.warns(UserWarning, match="called after"):
                _instrument_module.instrument_openai()
            mock_inst.return_value.instrument.assert_called_once()
    finally:
        del sys.modules["openai"]


def test_no_warning_when_target_library_not_yet_imported() -> None:
    """The happy path — no warning when imports are in the right order."""

    sys.modules.pop("anthropic", None)
    import warnings

    with patch("opentelemetry.instrumentation.anthropic.AnthropicInstrumentor"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _instrument_module.instrument_anthropic()
        # No UserWarning from our late-import guard.
        late_import_warnings = [w for w in caught if "called after" in str(w.message)]
        assert late_import_warnings == []


# ---------------------------------------------------------------------------
# instrument_auto
# ---------------------------------------------------------------------------


def test_instrument_auto_returns_only_libraries_actually_present() -> None:
    """Quietly skip libraries whose pkg isn't installed."""

    with patch.object(
        _instrument_module,
        "_library_importable",
        side_effect=lambda name: name == "openai",
    ), patch("opentelemetry.instrumentation.openai.OpenAIInstrumentor"):
        installed = _instrument_module.instrument_auto()
    assert installed == ("openai",)


def test_instrument_auto_calls_helpers_in_correct_order() -> None:
    """Providers (openai, anthropic) instrumented before dispatchers (litellm, smolagents).

    Order matters because higher-level frameworks that internally call
    a provider would otherwise risk double-emission if both instrumentors
    patch the same call site.
    """

    call_order: list[str] = []

    def make_helper(name: str) -> Any:
        def _helper(**_kw: Any) -> None:
            call_order.append(name)

        return _helper

    with patch.object(
        _instrument_module, "_library_importable", return_value=True
    ), patch.object(
        _instrument_module, "instrument_openai", make_helper("openai")
    ), patch.object(
        _instrument_module, "instrument_anthropic", make_helper("anthropic")
    ), patch.object(
        _instrument_module, "instrument_litellm", make_helper("litellm")
    ), patch.object(
        _instrument_module, "instrument_smolagents", make_helper("smolagents")
    ):
        _instrument_module.instrument_auto()

    assert call_order == ["openai", "anthropic", "litellm", "smolagents"]


def test_instrument_auto_swallows_missing_companion_instrumentor() -> None:
    """User has ``openai`` but not ``opentelemetry-instrumentation-openai``.

    The ``find_spec`` check passes (target lib installed), but the helper
    raises ImportError on the instrumentor import. ``instrument_auto``
    should skip silently — best-effort semantics.
    """

    def raising_helper(**_kw: Any) -> None:
        raise ImportError("companion instrumentor missing")

    with patch.object(
        _instrument_module, "_library_importable", return_value=True
    ), patch.object(_instrument_module, "instrument_openai", raising_helper), patch.object(
        _instrument_module, "instrument_anthropic", raising_helper
    ), patch.object(
        _instrument_module, "instrument_litellm", raising_helper
    ), patch.object(
        _instrument_module, "instrument_smolagents", raising_helper
    ):
        installed = _instrument_module.instrument_auto()
    assert installed == ()


# ---------------------------------------------------------------------------
# quickstart
# ---------------------------------------------------------------------------


def test_quickstart_delegates_to_instrument_auto() -> None:
    with patch.object(
        _instrument_module, "instrument_auto", return_value=("openai", "anthropic")
    ) as mock_auto:
        result = _instrument_module.quickstart(agent="demo")
    assert result == ("openai", "anthropic")
    mock_auto.assert_called_once()
    kwargs = mock_auto.call_args.kwargs
    assert kwargs["agent"] == "demo"


def test_quickstart_returns_empty_when_no_supported_lib_installed() -> None:
    """Quickstart on a fresh venv with no provider SDKs returns ()."""

    with patch.object(_instrument_module, "_library_importable", return_value=False):
        result = _instrument_module.quickstart(agent="demo")
    assert result == ()
