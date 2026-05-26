"""Fire-and-forget HTTP client for the Voight ingest endpoint.

Why "fire and forget"
---------------------
``VoightBackend`` runs inside the OTel ``BatchSpanProcessor`` worker thread,
which itself runs in the user's app process. A failing or slow Voight
backend must NEVER turn into a failing or slow LLM call for the user.
:meth:`IngestClient.send` returns immediately; the actual POST happens on
a background thread, and any error reaches the optional ``on_error`` hook
rather than the caller's stack.

We intentionally do not retry, batch, or buffer in v0.1.0. Those add state
and complexity — we'd rather drop the occasional event than ship a
half-implemented queue. Retry + buffer arrive in 0.2.0 once we have
real-world failure-mode data to design against.

Transport selection
-------------------
``httpx`` is the preferred client when installed. Without it we fall back
to :mod:`urllib.request` from the stdlib — Bitfrost stays usable in
"no extra dependencies" deployments. Tests inject a ``transport`` callable
to assert request shape without going to the network.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol


class _Response(Protocol):
    """Minimal protocol both ``httpx.Response`` and our urllib adapter satisfy.

    Attributes are declared via properties so the protocol matches read-only
    attributes (such as ``httpx.Response.is_success``) structurally.
    """

    @property
    def status_code(self) -> int: ...
    @property
    def is_success(self) -> bool: ...
    @property
    def text(self) -> str: ...


class Transport(Protocol):
    """Callable that performs one HTTP request and returns a :class:`_Response`."""

    def __call__(
        self,
        url: str,
        *,
        method: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> _Response: ...


class IngestClient(Protocol):
    """Public surface returned by :func:`create_ingest_client`.

    Required methods:

    - ``send(event)`` — fire-and-forget dispatch. Returns immediately; the
      HTTP request happens on a daemon thread.
    - ``shutdown(timeout_seconds)`` — block new sends, drain in-flight
      threads up to ``timeout_seconds``. Returns ``True`` when fully drained,
      ``False`` on timeout. Idempotent.
    - ``force_flush(timeout_millis)`` — drain in-flight threads up to the
      OTel-style millisecond budget WITHOUT marking the client shut down.
      Callers can keep sending after a successful flush.
    """

    def send(self, event: dict[str, Any]) -> None: ...
    def shutdown(self, timeout_seconds: float = 30.0) -> bool: ...
    def force_flush(self, timeout_millis: int = 30000) -> bool: ...


def create_ingest_client(
    api_base: str,
    api_key: str,
    *,
    transport: Transport | None = None,
    on_error: Callable[[BaseException], None] | None = None,
    timeout_seconds: float = 5.0,
) -> IngestClient:
    """Build an :class:`IngestClient` bound to a single api_base + api_key pair.

    Parameters
    ----------
    api_base
        Voight backend base URL (e.g. ``"https://api.voight.xyz"``). Trailing
        slashes are stripped so callers can pass it with or without one.
    api_key
        The ``vk_…`` API key. Set on every outgoing ``Authorization`` header.
    transport
        Optional override for the network call. Tests inject a mock; production
        callers omit this and the default httpx → urllib transport is used.
    on_error
        Called with any exception or non-2xx response that would otherwise be
        silently dropped. Useful for surfacing misconfiguration (bad key,
        wrong api_base) during development. Defaults to a no-op so production
        stays quiet.
    timeout_seconds
        Per-request timeout. Set low enough that a hung backend can't pile
        worker threads up forever.

    Returns
    -------
    IngestClient
        Object whose ``send`` method is synchronous (returns immediately) and
        never throws.
    """

    base = api_base.rstrip("/")
    url = f"{base}/v1/events"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    transport_impl = transport or _default_transport()

    def _noop(_err: BaseException) -> None:  # pragma: no cover - replaced below
        return

    error_handler = on_error or _noop

    # In-flight tracking — daemon threads spawned by ``send`` are added here
    # and removed in the dispatch ``finally``. ``shutdown`` / ``force_flush``
    # walk a snapshot of this set with ``thread.join(timeout)`` so callers
    # (``provider.force_flush`` / ``provider.shutdown``) actually wait for
    # the HTTP requests to complete instead of letting daemon threads die
    # silently at interpreter shutdown.
    in_flight: set[threading.Thread] = set()
    in_flight_lock = threading.Lock()
    shutdown_event = threading.Event()

    class _Client:
        def send(self, event: dict[str, Any]) -> None:
            if shutdown_event.is_set():
                # After shutdown we silently drop — sending here would spawn
                # a thread that the shutdown waiter already moved past.
                return

            try:
                body = json.dumps(event, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError) as err:
                # Serialisation can fail on non-JSON values (Decimal, custom
                # classes). Route to ``on_error`` before spawning a thread.
                _safe_call(error_handler, err)
                return

            # The dispatched thread closes over its own reference via
            # ``thread_box`` so it can remove itself from ``in_flight``
            # when the request completes (success or error).
            thread_box: list[threading.Thread] = []

            def _dispatch() -> None:
                try:
                    try:
                        response = transport_impl(
                            url,
                            method="POST",
                            headers=headers,
                            body=body,
                            timeout=timeout_seconds,
                        )
                    except BaseException as err:
                        _safe_call(error_handler, err)
                        return
                    if not _response_ok(response):
                        _safe_call(
                            error_handler,
                            RuntimeError(
                                f"voight ingest failed: {response.status_code} {response.text!r}"
                            ),
                        )
                finally:
                    if thread_box:
                        with in_flight_lock:
                            in_flight.discard(thread_box[0])

            thread = threading.Thread(target=_dispatch, daemon=True)
            thread_box.append(thread)
            with in_flight_lock:
                in_flight.add(thread)
            thread.start()

        def shutdown(self, timeout_seconds: float = 30.0) -> bool:
            """Block new ``send`` calls and drain in-flight threads.

            Idempotent — a second call simply re-drains (which is a no-op if
            the first call already completed). Returns ``True`` when every
            in-flight thread has exited, ``False`` if the timeout elapsed
            with threads still alive.
            """

            shutdown_event.set()
            return _drain(timeout_seconds, in_flight, in_flight_lock)

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            """Drain in-flight threads WITHOUT marking the client shut down.

            Lets the OTel ``provider.force_flush(timeout_millis)`` path wait
            for actual HTTP completion, not just for spans to leave the
            processor. Callers can still ``send`` after a successful flush.
            """

            return _drain(timeout_millis / 1000.0, in_flight, in_flight_lock)

    return _Client()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _drain(
    timeout_seconds: float,
    in_flight: set[threading.Thread],
    lock: threading.Lock,
) -> bool:
    """Wait for every in-flight dispatch thread or until the deadline elapses.

    We take a snapshot under ``lock`` so we don't iterate the set while
    completing threads remove themselves. ``thread.join(timeout)`` honours a
    per-thread remaining budget derived from a monotonic deadline.
    """

    if timeout_seconds < 0:
        timeout_seconds = 0.0
    deadline = time.monotonic() + timeout_seconds
    with lock:
        snapshot = list(in_flight)
    for thread in snapshot:
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)
    with lock:
        still_alive = any(t.is_alive() for t in in_flight)
    return not still_alive


def _response_ok(response: Any) -> bool:
    """Tolerate both ``httpx.Response`` (``is_success``) and adapters (``status_code``)."""

    is_success = getattr(response, "is_success", None)
    if isinstance(is_success, bool):
        return is_success
    status = getattr(response, "status_code", 0)
    return 200 <= int(status) < 300


def _safe_call(handler: Callable[[BaseException], None], err: BaseException) -> None:
    """Invoke the ``on_error`` callback without ever raising.

    A user-supplied callback that itself raises must NOT cascade and bring
    down the dispatch thread silently — we swallow secondary errors.
    """

    with contextlib.suppress(BaseException):
        handler(err)


def _default_transport() -> Transport:
    """Return the first transport whose backing library is importable."""

    try:
        return _httpx_transport()
    except ImportError:
        return _urllib_transport()


def _httpx_transport() -> Transport:
    """Build a transport backed by httpx. Imports lazily."""

    import httpx

    def _send(
        url: str,
        *,
        method: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> _Response:
        return httpx.request(
            method,
            url,
            content=body,
            headers=headers,
            timeout=timeout,
        )

    return _send


def _urllib_transport() -> Transport:
    """Build a transport backed by the stdlib ``urllib.request``."""

    from urllib import error as urllib_error
    from urllib import request as urllib_request

    class _UrllibResponse:
        def __init__(self, status_code: int, text: str) -> None:
            self.status_code = status_code
            self.text = text
            self.is_success = 200 <= status_code < 300

    def _send(
        url: str,
        *,
        method: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> _Response:
        req = urllib_request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                return _UrllibResponse(resp.status, resp.read().decode("utf-8", "replace"))
        except urllib_error.HTTPError as err:
            return _UrllibResponse(err.code, err.read().decode("utf-8", "replace"))

    return _send


__all__ = [
    "IngestClient",
    "Transport",
    "create_ingest_client",
]
