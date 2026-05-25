"""Tests for :mod:`bitfrost.backends.voight`.

Strategy
--------
``VoightBackend`` wraps :func:`bitfrost.ingest.create_ingest_client` plus a
configurable transport. Tests inject a fake transport so we can assert the
exact HTTP request shape without going to the network, and so that no real
``VOIGHT_KEY`` is needed for the suite.
"""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from bitfrost.backends.voight import VoightBackend


def _wait_for_call(mock: MagicMock, timeout: float = 1.0) -> None:
    end = threading.Event()

    def watcher() -> None:
        while not mock.called:
            if end.wait(0.01):
                return

    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    t.join(timeout=timeout)
    end.set()


def _ok_response() -> Any:
    return type("R", (), {"status_code": 202, "is_success": True, "text": ""})()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_voight_backend_requires_api_key() -> None:
    """No env, no options, no key → constructor raises (caller misconfigured)."""
    with pytest.raises(ValueError, match="VOIGHT_KEY"):
        VoightBackend(env={})


def test_voight_backend_accepts_explicit_api_key() -> None:
    backend = VoightBackend(api_key="vk_test_key", env={})
    # Smoke check that the public surface is intact.
    assert callable(backend.send)
    assert callable(backend.shutdown)
    assert callable(backend.force_flush)


def test_voight_backend_resolves_api_key_from_env() -> None:
    backend = VoightBackend(env={"VOIGHT_KEY": "vk_from_env"})
    assert callable(backend.send)


# ---------------------------------------------------------------------------
# Send semantics
# ---------------------------------------------------------------------------


def test_send_dispatches_event_to_v1_events_endpoint() -> None:
    transport = MagicMock(return_value=_ok_response())
    backend = VoightBackend(
        api_key="vk_test_key",
        env={},
        transport=transport,
    )
    payload = {"type": "action", "model": "gpt-4o-mini", "durationMs": 100}
    backend.send(payload)
    _wait_for_call(transport)
    args, kwargs = transport.call_args
    url = args[0] if args else kwargs.get("url")
    assert url == "https://api.voight.xyz/v1/events"
    assert kwargs.get("method") == "POST"
    headers = kwargs.get("headers") or {}
    assert headers["authorization"] == "Bearer vk_test_key"
    body = kwargs.get("body") or b""
    assert json.loads(body) == payload


def test_send_respects_custom_api_base() -> None:
    transport = MagicMock(return_value=_ok_response())
    backend = VoightBackend(
        api_key="vk_test_key",
        api_base="https://self-hosted.example.com",
        env={},
        transport=transport,
    )
    backend.send({"x": 1})
    _wait_for_call(transport)
    args, kwargs = transport.call_args
    url = args[0] if args else kwargs.get("url")
    assert url == "https://self-hosted.example.com/v1/events"


def test_send_routes_errors_to_on_error_hook() -> None:
    errors: list[BaseException] = []

    def broken_transport(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("connection refused")

    backend = VoightBackend(
        api_key="vk_test_key",
        env={},
        transport=broken_transport,
        on_error=errors.append,
    )
    backend.send({"x": 1})
    for _ in range(100):
        if errors:
            break
        threading.Event().wait(0.01)
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)


def test_send_never_raises_to_caller() -> None:
    """The hot-path contract: ``send`` is synchronous and silent on all errors."""

    def angry_transport(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("nope")

    backend = VoightBackend(api_key="vk_test_key", env={}, transport=angry_transport)
    # No exception should escape.
    backend.send({"x": 1})


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_shutdown_is_idempotent() -> None:
    backend = VoightBackend(api_key="vk_test_key", env={})
    backend.shutdown()
    backend.shutdown()  # must not raise


def test_force_flush_returns_true_on_clean_state() -> None:
    """v0.1 ships without a real buffer; ``force_flush`` is a contract no-op."""
    backend = VoightBackend(api_key="vk_test_key", env={})
    assert backend.force_flush() is True
    assert backend.force_flush(timeout_millis=1) is True
