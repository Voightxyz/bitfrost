"""Generic HTTP backend — POSTs events as JSON to any user-configured endpoint.

This backend is the universal escape hatch. Point it at:

- a custom in-house collector that accepts JSON over HTTP,
- a webhook receiver (Zapier / n8n / Make.com),
- a generic OpenTelemetry collector configured to accept JSON ingest,
- anywhere you can speak HTTP to.

It does **not** speak OTLP protobuf. Backends that require the OTLP wire
format (Langfuse, Phoenix, Datadog APM agent) should be used alongside
Bitfrost via OTel's own ``OTLPSpanExporter`` — register both processors on
the same ``TracerProvider`` and each receives the full span batch.

The name ``OTLPBackend`` is retained for compatibility with the v0.1
roadmap; the docstring + class doc describe the actual contract honestly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from bitfrost.ingest import Transport, create_ingest_client


class OTLPBackend:
    """Generic HTTP backend for any JSON-accepting endpoint.

    Parameters
    ----------
    endpoint
        Full URL to POST events to (e.g. ``"https://collector.example.com/v1/ingest"``).
        Required and non-empty.
    headers
        Extra HTTP headers to attach to every request (auth tokens, tenancy
        identifiers, content-type override, …). Built-in ``content-type:
        application/json`` is set first so user values can override it.
    transport
        Optional override for the network call. Tests inject mocks;
        production callers omit this and the default httpx → urllib
        transport is used.
    on_error
        Callback invoked with any exception or non-2xx response. Defaults
        to ``None`` (silent).
    timeout_seconds
        Per-request timeout, default 5 seconds.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        transport: Transport | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not endpoint:
            msg = "OTLPBackend requires a non-empty endpoint URL."
            raise ValueError(msg)

        # The ingest helper appends ``/v1/events`` to its ``api_base``; for
        # this generic backend the caller passes the full URL, so we route
        # around it via a tiny custom transport wrapper.
        wrapped_transport = _wrap_transport_with_full_url(
            transport or _default_transport_from_ingest(),
            endpoint,
            extra_headers=headers or {},
        )

        # ``create_ingest_client`` requires an api_key; we pass an empty
        # placeholder and override the Authorization header via the
        # ``headers`` argument (which our wrapped transport applies).
        self._client = create_ingest_client(
            api_base="https://placeholder.invalid",
            api_key="placeholder",
            transport=wrapped_transport,
            on_error=on_error,
            timeout_seconds=timeout_seconds,
        )
        self._shutdown_called = False

    def send(self, event: dict[str, Any]) -> None:
        if self._shutdown_called:
            return
        self._client.send(event)

    def shutdown(self) -> None:
        self._shutdown_called = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis  # Reserved for v0.2 retry-buffer support.
        return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wrap_transport_with_full_url(
    inner: Transport,
    full_url: str,
    *,
    extra_headers: dict[str, str],
) -> Transport:
    """Return a transport that ignores the upstream-built URL and uses ``full_url``.

    The :func:`~bitfrost.ingest.create_ingest_client` helper builds its own
    ``{api_base}/v1/events`` URL internally; for the generic-HTTP backend we
    let the user pass the full endpoint directly and route around the
    pre-built URL by overriding it here. User-supplied headers merge on top
    of the ingest-supplied auth header so callers can override defaults
    (e.g. to switch the content type).
    """

    def _send(
        _ignored_url: str,
        *,
        method: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> Any:
        merged: dict[str, str] = dict(headers)
        # User-provided headers win (allows content-type override etc.).
        merged.update(extra_headers)
        # The generic endpoint does not need the placeholder Bearer token.
        merged.pop("authorization", None)
        return inner(full_url, method=method, headers=merged, body=body, timeout=timeout)

    return cast(Transport, _send)


def _default_transport_from_ingest() -> Transport:
    """Lazy import of the default transport from the ingest module."""

    from bitfrost.ingest import _default_transport

    return _default_transport()


__all__ = ["OTLPBackend"]
