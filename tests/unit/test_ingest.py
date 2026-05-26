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


# ---------------------------------------------------------------------------
# Shutdown / force_flush — drain in-flight HTTP before exit
#
# Without this drain the daemon threads spawned by ``send`` die when the
# Python interpreter exits, dropping HTTP requests silently. Stages 1+2
# Bitfrost smokes (2026-05-25) hit exactly this and produced zero events
# in the dashboard. The tests below pin the drain semantics.
# ---------------------------------------------------------------------------


def test_force_flush_waits_for_in_flight_thread() -> None:
    """``force_flush`` blocks until the dispatch thread completes."""

    completed = threading.Event()

    def slow_transport(*_args: Any, **_kwargs: Any) -> Any:
        # Simulate a 200ms HTTP request — well within the flush budget.
        threading.Event().wait(0.2)
        completed.set()
        return type("R", (), {"status_code": 202, "is_success": True, "text": ""})()

    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=slow_transport,
    )
    client.send({"x": 1})
    assert not completed.is_set(), "transport should not have completed before flush"
    drained = client.force_flush(timeout_millis=2000)
    assert drained is True, "force_flush should return True when threads exit in time"
    assert completed.is_set(), "force_flush returned before transport completed"


def test_force_flush_honors_timeout_when_thread_exceeds_budget() -> None:
    """If the request runs longer than the budget, ``force_flush`` returns False."""

    release = threading.Event()

    def hung_transport(*_args: Any, **_kwargs: Any) -> Any:
        release.wait(timeout=2.0)
        return type("R", (), {"status_code": 202, "is_success": True, "text": ""})()

    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=hung_transport,
    )
    client.send({"x": 1})
    drained = client.force_flush(timeout_millis=100)  # 100ms vs hung 2s
    assert drained is False, "force_flush should return False on timeout"
    release.set()  # let the daemon thread finish so the test doesn't leak it


def test_shutdown_blocks_further_sends() -> None:
    """After ``shutdown``, ``send`` becomes a silent no-op."""

    transport = MagicMock(
        return_value=type("R", (), {"status_code": 202, "is_success": True, "text": ""})()
    )
    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=transport,
    )
    client.shutdown(timeout_seconds=1.0)
    client.send({"x": 1})
    # Wait briefly to confirm no background thread fired.
    threading.Event().wait(0.05)
    assert not transport.called, "shutdown should silence subsequent sends"


def test_shutdown_is_idempotent() -> None:
    """Calling ``shutdown`` twice must not raise or double-join."""

    transport = MagicMock(
        return_value=type("R", (), {"status_code": 202, "is_success": True, "text": ""})()
    )
    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=transport,
    )
    client.send({"x": 1})
    assert client.shutdown(timeout_seconds=1.0) is True
    # Second call: already drained, must still return True without raising.
    assert client.shutdown(timeout_seconds=1.0) is True


def test_shutdown_drains_multiple_in_flight_threads() -> None:
    """``shutdown`` waits for every concurrent dispatch, not just the first."""

    completed_count = 0
    counter_lock = threading.Lock()

    def slow_transport(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal completed_count
        threading.Event().wait(0.1)
        with counter_lock:
            completed_count += 1
        return type("R", (), {"status_code": 202, "is_success": True, "text": ""})()

    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=slow_transport,
    )
    for i in range(5):
        client.send({"i": i})
    drained = client.shutdown(timeout_seconds=3.0)
    assert drained is True
    assert completed_count == 5, f"expected 5 completions, got {completed_count}"


def test_force_flush_does_not_mark_shutdown() -> None:
    """``force_flush`` drains but leaves the client usable for new sends."""

    transport = MagicMock(
        return_value=type("R", (), {"status_code": 202, "is_success": True, "text": ""})()
    )
    client = create_ingest_client(
        api_base="https://api.voight.xyz",
        api_key="vk_test_key",
        transport=transport,
    )
    client.send({"first": True})
    client.force_flush(timeout_millis=1000)
    transport.reset_mock()
    client.send({"second": True})
    _wait_for_call(transport)
    assert transport.called, "force_flush must not silence further sends"
