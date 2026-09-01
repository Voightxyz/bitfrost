"""OTel ``SpanExporter`` implementation that powers Bitfrost.

The exporter is the glue between OpenTelemetry's tracing pipeline and the
Bitfrost backend layer:

.. code-block:: text

    [your app] → OTel TracerProvider → BatchSpanProcessor → BitfrostExporter → ExportBackend

For every span in a batch, :class:`BitfrostExporter`:

1. Extracts primitive fields from the :class:`~opentelemetry.sdk.trace.ReadableSpan`.
2. Calls :func:`~bitfrost.attribute_mapper.map_attributes` to derive an
   :class:`~bitfrost.types.EventPayload`.
3. Returns ``None`` for non-LLM spans (mapper handles this), silently skipping them.
4. Stamps the configured ``agent`` id and a stable ``session_id`` onto the event.
5. Applies the configured :class:`~bitfrost.types.PrivacyLevel` filter.
6. Hands the resulting event to the configured :class:`ExportBackend`.

Errors per-span are routed to an optional ``on_error`` hook; the exporter
ALWAYS returns :attr:`SpanExportResult.SUCCESS` so the OTel SDK doesn't
schedule retries that would re-deliver already-dispatched events.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from bitfrost.attribute_mapper import map_attributes
from bitfrost.backends.base import ExportBackend
from bitfrost.identity import resolve_agent, resolve_session_id
from bitfrost.privacy import apply_privacy
from bitfrost.types import PrivacyLevel


class BitfrostExporter(SpanExporter):
    """``SpanExporter`` that translates LLM spans to Voight events.

    Parameters
    ----------
    backend
        Destination implementing :class:`~bitfrost.backends.base.ExportBackend`.
        :class:`~bitfrost.backends.voight.VoightBackend` is the canonical
        choice for sending to ``api.voight.xyz``; see :mod:`bitfrost.backends`
        for the rest.
    agent
        Stable agent identifier surfaced under :attr:`EventPayload.agentId`.
        When omitted, resolves via the standard fallback chain
        (``VOIGHT_AGENT`` → ``HOSTNAME`` → ``"unknown-agent"``).
    session_id
        Trace-grouping identifier stamped on every event's
        ``metadata.sessionId``. When omitted, the exporter generates one
        UUID v4 at construction time and reuses it for the lifetime of
        the instance, so events emitted by a single exporter share one
        timeline in the dashboard.
    privacy
        Capture aggressiveness applied to every event before dispatch.
        Defaults to :attr:`PrivacyLevel.STANDARD` (local PII scrubbing).
    on_error
        Optional callback invoked with any per-span exception that the
        exporter would otherwise swallow silently. Useful for surfacing
        misconfiguration during development.
    """

    def __init__(
        self,
        backend: ExportBackend,
        *,
        agent: str | None = None,
        session_id: str | None = None,
        privacy: PrivacyLevel | str = PrivacyLevel.STANDARD,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._backend = backend
        self._agent = resolve_agent({"agent": agent})
        self._session_id = resolve_session_id({"sessionId": session_id})
        self._privacy = privacy
        self._on_error = on_error

    # ----- SpanExporter contract -------------------------------------------

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            try:
                event = self._span_to_event(span)
            except BaseException as err:
                self._route_error(err)
                continue
            if event is None:
                continue
            try:
                self._backend.send(event)
            except BaseException as err:
                self._route_error(err)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        with contextlib.suppress(BaseException):
            self._backend.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return bool(self._backend.force_flush(timeout_millis))
        except BaseException:
            return False

    # ----- Internals --------------------------------------------------------

    def _span_to_event(self, span: ReadableSpan) -> dict[str, Any] | None:
        """Map a ``ReadableSpan`` to a stamped, privacy-filtered event dict.

        Returns ``None`` for non-LLM spans (delegated to the mapper).
        """

        status_code = _status_code_name(span.status.status_code)
        scope_name = _scope_name(span.instrumentation_scope)
        events = _events_list(span.events)

        event = map_attributes(
            span_name=span.name,
            attributes=dict(span.attributes or {}),
            start_time_ns=span.start_time or 0,
            end_time_ns=span.end_time or 0,
            status_code=status_code,
            status_description=span.status.description,
            events=events,
            instrumentation_scope_name=scope_name,
        )
        if event is None:
            return None

        stamped: dict[str, Any] = dict(event)
        stamped["agentId"] = self._agent
        metadata = dict(stamped.get("metadata") or {})
        metadata.setdefault("sessionId", self._session_id)
        stamped["metadata"] = metadata

        # OTel trace context, first-class: traceId groups one request's spans
        # into a tree, parentSpanId links it. Zero ids (no-op tracer) are
        # skipped rather than shipped as meaningless zeros.
        ctx = getattr(span, "context", None)
        if ctx is not None and getattr(ctx, "trace_id", 0):
            stamped["traceId"] = format(ctx.trace_id, "032x")
            stamped["spanId"] = format(ctx.span_id, "016x")
        parent = getattr(span, "parent", None)
        if parent is not None and getattr(parent, "span_id", 0):
            stamped["parentSpanId"] = format(parent.span_id, "016x")

        return apply_privacy(stamped, self._privacy)

    def _route_error(self, err: BaseException) -> None:
        """Forward an exception to ``on_error`` without ever raising back."""

        if self._on_error is None:
            return
        with contextlib.suppress(BaseException):
            self._on_error(err)


# ---------------------------------------------------------------------------
# Internal helpers (pure, no side effects)
# ---------------------------------------------------------------------------


def _status_code_name(code: Any) -> str:
    """Normalise an OTel ``StatusCode`` (or stand-in) to its enum-tail name.

    ``StatusCode.UNSET`` → ``"UNSET"``; ``"StatusCode.ERROR"`` → ``"ERROR"``.
    Tolerant of strings and stand-in objects so tests don't need real OTel types.
    """

    name = getattr(code, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(code).rsplit(".", 1)[-1].upper()


def _scope_name(scope: Any) -> str | None:
    if scope is None:
        return None
    name = getattr(scope, "name", None)
    return str(name) if name else None


def _events_list(events: Any) -> list[dict[str, Any]] | None:
    if not events:
        return None
    return [
        {
            "name": getattr(evt, "name", None),
            "attributes": dict(getattr(evt, "attributes", None) or {}),
        }
        for evt in events
    ]


__all__ = ["BitfrostExporter"]
