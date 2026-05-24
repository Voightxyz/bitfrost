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
    """Public surface returned by :func:`create_ingest_client`."""

    def send(self, event: dict[str, Any]) -> None: ...


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

    class _Client:
        def send(self, event: dict[str, Any]) -> None:
            try:
                body = json.dumps(event, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError) as err:
                # Serialisation can fail on non-JSON values (Decimal, custom
                # classes). Route to ``on_error`` before spawning a thread.
                _safe_call(error_handler, err)
                return

            def _dispatch() -> None:
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

            thread = threading.Thread(target=_dispatch, daemon=True)
            thread.start()

    return _Client()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
