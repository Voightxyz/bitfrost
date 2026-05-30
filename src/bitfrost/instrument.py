"""Auto-instrumentation helpers — the drop-in Logfire-style API.

Each helper does the OTel boilerplate users would otherwise write by
hand: build (or reuse) a ``TracerProvider``, attach a
:class:`~bitfrost.exporter.BitfrostExporter`, install the relevant
library instrumentor, and return.

::

    import bitfrost
    bitfrost.instrument_openai(agent="my-app")
    # any openai SDK call from here on streams to Bitfrost

The default backend is :class:`~bitfrost.backends.console.ConsoleBackend`
so a developer who just runs ``pip install bitfrost`` and calls
``instrument_openai()`` already sees their LLM activity in their
terminal — no Voight key, no SQLite path, no setup.

Ecosystem-aligned
-----------------
Bitfrost does NOT ship its own monkey-patches for the supported
libraries. Each helper delegates to the canonical OTel / OpenInference
instrumentor for that library:

- ``instrument_openai`` → ``opentelemetry-instrumentation-openai``
- ``instrument_anthropic`` → ``opentelemetry-instrumentation-anthropic``
- ``instrument_litellm`` → ``litellm.integrations.opentelemetry.OpenTelemetry``
- ``instrument_smolagents`` → ``openinference-instrumentation-smolagents``

If those libraries ever change their attribute conventions, Bitfrost's
multi-version mapper already handles both v1.27 + v1.32+ semconv shapes,
and the ecosystem standard fix flows in for free.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from bitfrost.exporter import BitfrostExporter
from bitfrost.types import PrivacyLevel

# Module-level state so multiple ``instrument_*`` calls in one process
# add SpanProcessors to the same TracerProvider, never clobber.
_BITFROST_OWNED_PROVIDER: TracerProvider | None = None

# Per-process registry of (provider_id, backend_id) pairs that already
# have a BatchSpanProcessor attached. Without this, calling
# ``instrument_openai(backend=b)`` + ``instrument_anthropic(backend=b)``
# would add TWO processors for the same backend — every span would be
# emitted twice, producing duplicate events in the destination. The
# registry dedupes by Python object identity so a user who genuinely
# wants two distinct backends (different keys, different destinations)
# passes two different instances and gets two processors as expected.
_PROCESSORS_REGISTERED: dict[int, set[int]] = {}


def instrument_openai(
    *,
    backend: Any = None,
    agent: str | None = None,
    session_id: str | None = None,
    privacy: PrivacyLevel | str = PrivacyLevel.STANDARD,
    on_error: Any = None,
) -> None:
    """Auto-instrument the ``openai`` SDK.

    Sync + async paths are both covered — ``opentelemetry-instrumentation-openai``
    monkey-patches both ``OpenAI`` and ``AsyncOpenAI``.

    Parameters
    ----------
    backend
        Destination backend. Defaults to a :class:`~bitfrost.backends.console.ConsoleBackend`
        so the zero-config path produces visible output in the terminal.
    agent, session_id, privacy, on_error
        Passed straight through to :class:`~bitfrost.exporter.BitfrostExporter`.

    Notes
    -----
    Call this BEFORE the first ``import openai`` for the monkey-patch
    to take effect. If ``openai`` is already imported, a
    :class:`UserWarning` is emitted (the instrumentation will still
    install but won't capture already-active client instances).
    """

    _warn_if_already_imported("openai")
    _bootstrap_provider(
        backend=backend,
        agent=agent,
        session_id=session_id,
        privacy=privacy,
        on_error=on_error,
    )
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()


def instrument_anthropic(
    *,
    backend: Any = None,
    agent: str | None = None,
    session_id: str | None = None,
    privacy: PrivacyLevel | str = PrivacyLevel.STANDARD,
    on_error: Any = None,
) -> None:
    """Auto-instrument the ``anthropic`` SDK. Mirror of :func:`instrument_openai`."""

    _warn_if_already_imported("anthropic")
    _bootstrap_provider(
        backend=backend,
        agent=agent,
        session_id=session_id,
        privacy=privacy,
        on_error=on_error,
    )
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

    AnthropicInstrumentor().instrument()


def instrument_litellm(
    *,
    backend: Any = None,
    agent: str | None = None,
    session_id: str | None = None,
    privacy: PrivacyLevel | str = PrivacyLevel.STANDARD,
    on_error: Any = None,
) -> None:
    """Auto-instrument the ``litellm`` package for SDK use.

    LiteLLM ships ``litellm.integrations.opentelemetry.OpenTelemetry``
    but that path **only activates inside their Proxy Server** — for
    direct SDK callers (``litellm.completion(...)`` in app code) the
    Proxy Server check fails and the OTel hook is skipped silently.

    The canonical SDK-side path is ``litellm.integrations.custom_logger.CustomLogger``:
    subclass it, override ``log_success_event`` / ``log_failure_event``,
    and translate LiteLLM's callback kwargs into OTel spans manually.
    We ship that adapter as :class:`_BitfrostLiteLLMHandler` below.

    The adapter emits one OTel span per ``litellm.completion()`` call,
    annotated with semconv v1.32+ ``gen_ai.*`` attributes so Bitfrost's
    multi-version mapper picks them up identically to the OpenAI /
    Anthropic auto-instrumentor paths.
    """

    provider = _bootstrap_provider(
        backend=backend,
        agent=agent,
        session_id=session_id,
        privacy=privacy,
        on_error=on_error,
    )
    import litellm

    tracer = provider.get_tracer("bitfrost.litellm")
    handler_cls = _get_litellm_handler_class()
    handler = handler_cls(tracer)

    # litellm.callbacks is a list of handlers — append rather than
    # replace so users with other callbacks (Langfuse, Helicone) keep
    # them. Dedup by the ``_is_bitfrost_handler`` marker attribute, NOT
    # isinstance: the handler class is built lazily inside a factory so
    # there's no stable module-level class object to isinstance against.
    existing = getattr(litellm, "callbacks", None)
    if isinstance(existing, list):
        if not any(getattr(c, "_is_bitfrost_handler", False) for c in existing):
            existing.append(handler)
    else:
        litellm.callbacks = [handler]


def instrument_smolagents(
    *,
    backend: Any = None,
    agent: str | None = None,
    session_id: str | None = None,
    privacy: PrivacyLevel | str = PrivacyLevel.STANDARD,
    on_error: Any = None,
) -> None:
    """Auto-instrument ``smolagents`` via OpenInference's instrumentor.

    smolagents has no native OTel integration in its core; the Arize
    OpenInference team maintains ``openinference-instrumentation-smolagents``
    which patches ``smolagents.MultiStepAgent`` and friends. We delegate
    to that instrumentor with our TracerProvider attached.
    """

    _warn_if_already_imported("smolagents")
    provider = _bootstrap_provider(
        backend=backend,
        agent=agent,
        session_id=session_id,
        privacy=privacy,
        on_error=on_error,
    )
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor

    SmolagentsInstrumentor().instrument(tracer_provider=provider)


def instrument_auto(
    *,
    backend: Any = None,
    agent: str | None = None,
    session_id: str | None = None,
    privacy: PrivacyLevel | str = PrivacyLevel.STANDARD,
    on_error: Any = None,
) -> Sequence[str]:
    """Detect which supported libraries are installed and instrument each.

    Returns a tuple of library names that were instrumented. Quietly
    skips libraries whose instrumentor isn't installed — the convenience
    here is "instrument every supported lib I have available", not "fail
    if I'm missing something".

    Order matters: we instrument lower-level providers (openai,
    anthropic) before higher-level dispatchers (litellm, smolagents) so
    that agentic frameworks that internally call the providers don't
    double-emit through both paths in the rare case both instrumentors
    monkey-patch the same call site.
    """

    instrumented: list[str] = []
    for name, fn in (
        ("openai", instrument_openai),
        ("anthropic", instrument_anthropic),
        ("litellm", instrument_litellm),
        ("smolagents", instrument_smolagents),
    ):
        if not _library_importable(name):
            continue
        try:
            fn(
                backend=backend,
                agent=agent,
                session_id=session_id,
                privacy=privacy,
                on_error=on_error,
            )
            instrumented.append(name)
        except ImportError:
            # The companion instrumentor isn't installed (e.g. user has
            # openai but not opentelemetry-instrumentation-openai).
            # Quietly skip — instrument_auto is best-effort.
            continue
    return tuple(instrumented)


def quickstart(
    *,
    agent: str | None = None,
    backend: Any = None,
    privacy: PrivacyLevel | str = PrivacyLevel.STANDARD,
) -> Sequence[str]:
    """One-line setup: default ConsoleBackend + auto-instrument everything.

    The Logfire ``logfire.configure()`` equivalent. Drop this at the
    top of a Python entrypoint and any installed LLM library starts
    streaming to the terminal::

        import bitfrost
        bitfrost.quickstart(agent="my-app")

    Returns the tuple of library names that were instrumented so the
    caller can sanity-check the auto-detection result.
    """

    return instrument_auto(agent=agent, backend=backend, privacy=privacy)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _bootstrap_provider(
    *,
    backend: Any,
    agent: str | None,
    session_id: str | None,
    privacy: PrivacyLevel | str,
    on_error: Any,
) -> TracerProvider:
    """Resolve a TracerProvider and attach a Bitfrost-bound SpanProcessor.

    Behaviour:
    - If the application has already set a TracerProvider, augment it
      (add a SpanProcessor) — never clobber. This is how Bitfrost
      co-exists with Sentry, Datadog, Langfuse, OpenTelemetry collectors.
    - If we already own the TracerProvider from a previous Bitfrost
      ``instrument_*`` call, reuse it so a second helper just adds
      another processor — never two providers.
    - If no provider exists at all, install a fresh one.
    """

    global _BITFROST_OWNED_PROVIDER

    if backend is None:
        from bitfrost.backends.console import ConsoleBackend

        backend = ConsoleBackend()

    exporter = BitfrostExporter(
        backend=backend,
        agent=agent,
        session_id=session_id,
        privacy=privacy,
        on_error=on_error,
    )

    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        provider: TracerProvider = existing
    elif _BITFROST_OWNED_PROVIDER is not None:
        provider = _BITFROST_OWNED_PROVIDER
        trace.set_tracer_provider(provider)
    else:
        provider = TracerProvider()
        _BITFROST_OWNED_PROVIDER = provider
        trace.set_tracer_provider(provider)

    # Dedupe by (provider, backend) identity so calling two helpers with
    # the same backend instance doesn't double-emit every span.
    provider_id = id(provider)
    backend_id = id(backend)
    seen = _PROCESSORS_REGISTERED.setdefault(provider_id, set())
    if backend_id in seen:
        return provider
    seen.add(backend_id)

    # BatchSpanProcessor for production-grade throughput. The
    # underlying ingest path drains in-flight HTTP on shutdown/flush
    # (fix commit f07c83e) so daemon-thread loss is not a concern.
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _warn_if_already_imported(module_name: str) -> None:
    """Emit a UserWarning when the library is already in ``sys.modules``.

    The OTel instrumentors monkey-patch at import time; if a client
    instance was constructed before the patch landed, its bound methods
    keep referencing the un-instrumented originals. This warning lets a
    user know they ran their imports in the wrong order without making
    the helper fail.
    """

    if module_name in sys.modules:
        warnings.warn(
            f"bitfrost.instrument_{module_name}() called after `import {module_name}` — "
            f"already-constructed client instances will NOT be captured. "
            f"Move bitfrost.instrument_{module_name}() above `import {module_name}` "
            f"in your entrypoint.",
            UserWarning,
            stacklevel=3,
        )


def _library_importable(module_name: str) -> bool:
    """Cheap test for "is this library installed?" without importing it."""

    import importlib.util

    return importlib.util.find_spec(module_name) is not None


# ---------------------------------------------------------------------------
# LiteLLM SDK adapter
# ---------------------------------------------------------------------------


# The handler class is built once, lazily, by ``_get_litellm_handler_class``
# and cached here. Lazy construction keeps litellm an optional dependency:
# importing :mod:`bitfrost.instrument` never imports litellm; only calling
# ``instrument_litellm`` does.
_LITELLM_HANDLER_CLASS: type | None = None


def _get_litellm_handler_class() -> type:
    """Build (once) and return the LiteLLM ``CustomLogger`` adapter subclass.

    The subclass translates each LiteLLM ``log_{success,failure}_event``
    callback into an OTel span with semconv v1.32+ ``gen_ai.*``
    attributes, so Bitfrost's attribute_mapper handles LiteLLM calls
    identically to the openai / anthropic auto-instrumentor paths.

    Instances carry an ``_is_bitfrost_handler = True`` marker so
    :func:`instrument_litellm` can dedupe its own handler in
    ``litellm.callbacks`` without relying on an isinstance check against
    a class object that doesn't exist at module import time (litellm is
    imported lazily).
    """

    global _LITELLM_HANDLER_CLASS
    if _LITELLM_HANDLER_CLASS is not None:
        return _LITELLM_HANDLER_CLASS

    from litellm.integrations.custom_logger import CustomLogger

    class _BitfrostLiteLLMHandler(CustomLogger):
        _is_bitfrost_handler = True

        def __init__(self, tracer: Any) -> None:
            super().__init__()
            self._tracer = tracer

        def log_success_event(
            self,
            kwargs: dict[str, Any],
            response_obj: Any,
            start_time: Any,
            end_time: Any,
        ) -> None:
            _emit_litellm_span(
                tracer=self._tracer,
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                success=True,
            )

        def log_failure_event(
            self,
            kwargs: dict[str, Any],
            response_obj: Any,
            start_time: Any,
            end_time: Any,
        ) -> None:
            _emit_litellm_span(
                tracer=self._tracer,
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                success=False,
            )

        async def async_log_success_event(
            self,
            kwargs: dict[str, Any],
            response_obj: Any,
            start_time: Any,
            end_time: Any,
        ) -> None:
            _emit_litellm_span(
                tracer=self._tracer,
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                success=True,
            )

        async def async_log_failure_event(
            self,
            kwargs: dict[str, Any],
            response_obj: Any,
            start_time: Any,
            end_time: Any,
        ) -> None:
            _emit_litellm_span(
                tracer=self._tracer,
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                success=False,
            )

    _LITELLM_HANDLER_CLASS = _BitfrostLiteLLMHandler
    return _BitfrostLiteLLMHandler


def _emit_litellm_span(
    *,
    tracer: Any,
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    success: bool,
) -> None:
    """Build an OTel span carrying gen_ai.* attributes from LiteLLM callback args.

    LiteLLM's callback shape: ``kwargs`` is the full dict of completion
    parameters (``model``, ``messages``, ``custom_llm_provider``, ...),
    ``response_obj`` is the LiteLLM ``ModelResponse`` with
    ``usage`` + ``choices`` populated, and ``start_time`` / ``end_time``
    are wall-clock ``datetime`` instances.
    """

    from opentelemetry.trace import Status, StatusCode

    model = str(kwargs.get("model") or "")
    provider = str(kwargs.get("custom_llm_provider") or "")

    # Derive timing in nanoseconds for OTel span start/end. LiteLLM
    # passes Python ``datetime`` objects — convert via ``.timestamp()``.
    try:
        start_ns = int(start_time.timestamp() * 1_000_000_000)
        end_ns = int(end_time.timestamp() * 1_000_000_000)
    except (AttributeError, TypeError):
        # Fall back to "now" if the callback didn't pass datetimes.
        import time as _time

        end_ns = int(_time.time() * 1_000_000_000)
        start_ns = end_ns

    span = tracer.start_span("litellm.completion", start_time=start_ns)
    try:
        span.set_attribute("gen_ai.provider.name", provider or "litellm")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "chat")

        # Usage — LiteLLM normalises to OpenAI's shape regardless of
        # the actual provider, so the keys are stable.
        usage = getattr(response_obj, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
            span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))

        # Response model + finish reason
        response_model = getattr(response_obj, "model", None)
        if response_model:
            span.set_attribute("gen_ai.response.model", str(response_model))

        choices = getattr(response_obj, "choices", None) or []
        if choices:
            finish = getattr(choices[0], "finish_reason", None)
            if finish:
                span.set_attribute("gen_ai.response.finish_reasons", [str(finish)])

        if not success:
            span.set_status(Status(StatusCode.ERROR))
    finally:
        span.end(end_time=end_ns)


__all__ = [
    "instrument_anthropic",
    "instrument_auto",
    "instrument_litellm",
    "instrument_openai",
    "instrument_smolagents",
    "quickstart",
]
