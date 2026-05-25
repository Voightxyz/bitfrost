"""Tests for :mod:`bitfrost.backends.otlp`.

``OTLPBackend`` is the generic-HTTP destination — it POSTs each
:class:`~bitfrost.types.EventPayload` as JSON to a user-configured endpoint
with optional custom headers (auth, tenancy). Tests inject the transport
to assert request shape without going to the network.
"""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from bitfrost.backends.otlp import OTLPBackend


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


def _ok() -> Any:
    return type("R", (), {"status_code": 202, "is_success": True, "text": ""})()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_endpoint_is_required() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        OTLPBackend(endpoint="")


def test_accepts_full_endpoint_url() -> None:
    backend = OTLPBackend(endpoint="https://my-collector.example.com/v1/events")
    assert callable(backend.send)


# ---------------------------------------------------------------------------
# Send semantics
# ---------------------------------------------------------------------------


def test_send_posts_event_to_configured_endpoint() -> None:
    transport = MagicMock(return_value=_ok())
    backend = OTLPBackend(
        endpoint="https://collector.example.com/ingest",
        transport=transport,
    )
    payload = {"type": "action", "model": "gpt-4o-mini"}
    backend.send(payload)
    _wait_for_call(transport)
    args, kwargs = transport.call_args
    url = args[0] if args else kwargs.get("url")
    assert url == "https://collector.example.com/ingest"
    assert kwargs.get("method") == "POST"
    headers = kwargs.get("headers") or {}
    assert headers["content-type"] == "application/json"
    body = kwargs.get("body") or b""
    assert json.loads(body) == payload


def test_send_attaches_custom_headers() -> None:
    transport = MagicMock(return_value=_ok())
    backend = OTLPBackend(
        endpoint="https://collector.example.com/ingest",
        headers={"x-api-key": "secret-token", "x-tenant": "acme"},
        transport=transport,
    )
    backend.send({"x": 1})
    _wait_for_call(transport)
    headers = transport.call_args.kwargs.get("headers") or {}
    assert headers["x-api-key"] == "secret-token"
    assert headers["x-tenant"] == "acme"
    # Built-in content-type always set, but does not override user headers.
    assert headers["content-type"] == "application/json"


def test_user_headers_override_default_content_type() -> None:
    """Power users can override defaults (e.g. switch to protobuf)."""
    transport = MagicMock(return_value=_ok())
    backend = OTLPBackend(
        endpoint="https://collector.example.com/ingest",
        headers={"content-type": "application/x-protobuf"},
        transport=transport,
    )
    backend.send({"x": 1})
    _wait_for_call(transport)
    headers = transport.call_args.kwargs.get("headers") or {}
    assert headers["content-type"] == "application/x-protobuf"


def test_send_routes_transport_errors_to_on_error() -> None:
    errors: list[BaseException] = []

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("collector down")

    backend = OTLPBackend(
        endpoint="https://collector.example.com/ingest",
        transport=broken,
        on_error=errors.append,
    )
    backend.send({"x": 1})
    for _ in range(100):
        if errors:
            break
        threading.Event().wait(0.01)
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
