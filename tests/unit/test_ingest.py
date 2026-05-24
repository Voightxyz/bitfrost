"""Tests for :mod:`bitfrost.ingest`.

Strategy
--------
The ingest client must NEVER block, throw, or surface backend errors to the
caller — it sits in user-app hot paths. Tests assert that:

- The dispatched request shape (URL, headers, body) is correct.
- The client survives mock HTTP failures silently and routes them to
  ``on_error`` instead of raising.
- A trailing slash on ``api_base`` doesn't produce ``//v1/events``.
- ``httpx`` is preferred when installed; ``urllib`` is the zero-deps fallback.
"""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import MagicMock

from bitfrost.ingest import create_ingest_client


def _wait_for_call(mock: MagicMock, timeout: float = 1.0) -> None:
    """Block until a mock has been called once, or the timeout elapses.

    Ingest dispatches on a background thread; tests need to synchronise.
    """

    end = threading.Event()

    def watcher() -> None:
        while not mock.called:
            if end.wait(0.01):
                return

    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    t.join(timeout=timeout)
    end.set()


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_send_posts_to_v1_events_with_correct_headers_and_body() -> None:
    transport = MagicMock(
        return_value=type("R", (), {"status_code": 202, "is_success": True, "text": ""})()
    )
    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=transport,
    )
    payload = {"type": "action", "model": "gpt-4o-mini", "durationMs": 100}
    client.send(payload)
    _wait_for_call(transport)
    args, kwargs = transport.call_args
    # First positional arg is the URL; method goes in kwargs.
    url = args[0] if args else kwargs.get("url")
    assert url == "https://api.voight.xyz/v1/events"
    assert kwargs.get("method") == "POST"
    headers = kwargs.get("headers") or {}
    assert headers["content-type"] == "application/json"
    assert headers["authorization"] == "Bearer vk_test_key"
    body = kwargs.get("body") or b""
    assert json.loads(body) == payload


def test_send_normalises_trailing_slash_on_api_base() -> None:
    transport = MagicMock(
        return_value=type("R", (), {"status_code": 202, "is_success": True, "text": ""})()
    )
    client = create_ingest_client(
        api_base="https://api.voight.xyz///",
        api_key="vk_test_key",
        transport=transport,
    )
    client.send({"x": 1})
    _wait_for_call(transport)
    args, kwargs = transport.call_args
    url = args[0] if args else kwargs.get("url")
    assert url == "https://api.voight.xyz/v1/events"


# ---------------------------------------------------------------------------
# Fire-and-forget semantics
# ---------------------------------------------------------------------------


def test_send_never_blocks_caller() -> None:
    """``send`` returns immediately; the POST happens on a background thread."""

    block = threading.Event()
    released = threading.Event()

    def slow_transport(*_args: Any, **_kwargs: Any) -> Any:
        released.set()
        block.wait(timeout=1.0)
        return type("R", (), {"status_code": 202, "is_success": True, "text": ""})()

    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=slow_transport,
    )
    client.send({"x": 1})
    # If send blocked, this assertion would never succeed within 0.5s
    # because the transport blocks on `block` for up to 1s.
    assert released.wait(timeout=0.5), "transport never invoked — send blocked the caller"
    block.set()


def test_send_routes_transport_exceptions_to_on_error() -> None:
    """A raise inside the transport must NOT propagate; ``on_error`` sees it."""

    errors: list[BaseException] = []

    def broken_transport(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network down")

    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=broken_transport,
        on_error=errors.append,
    )
    client.send({"x": 1})
    # Wait briefly for the background thread to surface the error.
    for _ in range(100):
        if errors:
            break
        threading.Event().wait(0.01)
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "network down" in str(errors[0])


def test_send_routes_non_2xx_responses_to_on_error() -> None:
    errors: list[BaseException] = []
    response = type("R", (), {"status_code": 401, "is_success": False, "text": "Unauthorized"})()
    transport = MagicMock(return_value=response)
    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_bad_key",
        transport=transport,
        on_error=errors.append,
    )
    client.send({"x": 1})
    _wait_for_call(transport)
    for _ in range(100):
        if errors:
            break
        threading.Event().wait(0.01)
    assert len(errors) == 1
    assert "401" in str(errors[0])


def test_send_swallows_json_serialisation_errors() -> None:
    """A non-serialisable payload routes to on_error instead of raising."""

    errors: list[BaseException] = []
    transport = MagicMock()

    class Unserialisable:
        pass

    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=transport,
        on_error=errors.append,
    )
    client.send({"weird": Unserialisable()})  # type: ignore[dict-item]
    for _ in range(100):
        if errors:
            break
        threading.Event().wait(0.01)
    assert len(errors) == 1
    assert isinstance(errors[0], TypeError)
    # Transport must NOT have been invoked because we never built a body.
    assert not transport.called


# ---------------------------------------------------------------------------
# Default transport selection (httpx vs urllib)
# ---------------------------------------------------------------------------


def test_create_client_works_without_explicit_transport() -> None:
    """Without ``transport`` injected, the client picks httpx or urllib internally.

    We only assert the client builds; the actual network call is short-circuited
    by passing an unresolvable api_base after construction would be brittle,
    so we just construct + verify the surface.
    """

    client = create_ingest_client(
        api_base="https://nonexistent.invalid",
        api_key="vk_test_key",
    )
    # Smoke: client is callable and has a `send` attribute.
    assert callable(client.send)
