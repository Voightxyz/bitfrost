"""Voight ingest backend — POSTs each event to ``api.voight.xyz/v1/events``.

This is the canonical destination for Bitfrost when you have a Voight
account. The backend is multi-tenant by ``VOIGHT_KEY``: each user's events
land in their own dashboard, segregated by API key on the backend side.

Architecture
------------
``VoightBackend`` is a thin wrapper around
:func:`bitfrost.ingest.create_ingest_client`. The ingest module owns the
fire-and-forget HTTP semantics (background thread, error swallowing,
httpx-or-urllib transport selection); the backend just resolves the API
key + base URL and delegates.

Standalone-friendly
-------------------
For local development with no Voight account, use
:class:`~bitfrost.backends.console.ConsoleBackend` instead. The whole
Bitfrost stack stays usable without ever touching ``api.voight.xyz``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from typing import Any

from bitfrost.identity import resolve_api_key
from bitfrost.ingest import Transport, create_ingest_client

DEFAULT_API_BASE = "https://api.voight.xyz"


class VoightBackend:
    """Backend that ships events to the Voight ingest endpoint.

    Parameters
    ----------
    api_key
        Voight API key (``vk_…``). Optional — when omitted, resolves from
        ``options['voightApiKey']`` then ``env['VOIGHT_KEY']``. Raises
        :class:`ValueError` if none of those are set, because a backend
        configured without auth would silently drop every event.
    api_base
        Override for the ingest base URL. Useful for self-hosted
        deployments or for pointing tests at a local mock server.
    transport
        Optional override for the HTTP call. Tests inject mocks; production
        callers omit this and the default httpx → urllib transport is used.
    on_error
        Callback invoked with any exception or non-2xx response that would
        otherwise be silently dropped. Defaults to ``None`` (silent).
    env
        Environment mapping used to resolve ``VOIGHT_KEY`` when ``api_key``
        is not passed explicitly. Defaults to :data:`os.environ` when omitted.
    timeout_seconds
        Per-request timeout. Default 5 seconds keeps a hung backend from
        piling worker threads up forever.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_base: str = DEFAULT_API_BASE,
        transport: Transport | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        resolved_key = resolve_api_key({"voightApiKey": api_key}, env)
        if not resolved_key:
            msg = (
                "VoightBackend requires an API key. Pass api_key=... or set "
                "the VOIGHT_KEY environment variable."
            )
            raise ValueError(msg)

        self._client = create_ingest_client(
            api_base=api_base,
            api_key=resolved_key,
            transport=transport,
            on_error=on_error,
            timeout_seconds=timeout_seconds,
        )
        self._shutdown_called = False

    def send(self, event: dict[str, Any]) -> None:
        """Dispatch one event. Synchronous, never raises, never blocks."""

        if self._shutdown_called:
            return
        self._client.send(event)

    def shutdown(self) -> None:
        """Block new sends and wait for in-flight HTTP requests to complete.

        Idempotent. Delegates the actual drain to
        :meth:`IngestClient.shutdown`, which joins every daemon thread
        spawned by :meth:`send` with a 30-second budget. Without this drain
        the daemon threads would die at interpreter shutdown — exactly the
        bug that caused Stages 1+2 smoke events to never reach the
        backend on 2026-05-25.
        """

        self._shutdown_called = True
        with contextlib.suppress(BaseException):
            self._client.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Wait for in-flight HTTP requests up to ``timeout_millis``.

        Propagates to :meth:`IngestClient.force_flush`. Unlike
        :meth:`shutdown`, callers can still send after a successful flush —
        this matches the OTel ``SpanExporter.force_flush`` contract.
        Returns ``True`` when every dispatch completed in time.
        """

        try:
            return bool(self._client.force_flush(timeout_millis))
        except BaseException:
            return False


__all__ = ["DEFAULT_API_BASE", "VoightBackend"]
