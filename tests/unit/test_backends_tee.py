"""Tests for :mod:`bitfrost.backends.tee`.

TeeBackend's job is fan-out with strict error isolation. The tests
pin the two contracts users rely on:

- One flaky child never silences the others (Voight backend hiccup
  doesn't stop the local SQLite log from being written).
- Lifecycle (``shutdown``, ``force_flush``) reaches every child even
  when intermediate children raise.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from bitfrost.backends.tee import TeeBackend


def _make_event() -> dict[str, Any]:
    return {"agentId": "tee-test", "type": "action", "model": "gpt-4o-mini"}


def test_requires_at_least_one_child() -> None:
    """Empty TeeBackend would silently drop every event — refuse to build."""

    with pytest.raises(ValueError):
        TeeBackend()


def test_send_fans_out_to_every_child() -> None:
    a, b, c = MagicMock(), MagicMock(), MagicMock()
    tee = TeeBackend(a, b, c)
    event = _make_event()
    tee.send(event)
    a.send.assert_called_once_with(event)
    b.send.assert_called_once_with(event)
    c.send.assert_called_once_with(event)


def test_send_continues_after_one_child_raises() -> None:
    """A raise in child A must NOT stop B + C from receiving the event."""

    a = MagicMock()
    a.send.side_effect = RuntimeError("backend A blew up")
    b, c = MagicMock(), MagicMock()
    errors: list[BaseException] = []
    tee = TeeBackend(a, b, c, on_error=errors.append)

    tee.send(_make_event())

    b.send.assert_called_once()  # B still received
    c.send.assert_called_once()  # C still received
    assert len(errors) == 1
    assert "backend A blew up" in str(errors[0])


def test_shutdown_broadcasts_to_every_child_even_when_one_raises() -> None:
    """Lifecycle reaches every child regardless of intermediate failures."""

    a = MagicMock()
    a.shutdown.side_effect = RuntimeError("A shutdown failed")
    b, c = MagicMock(), MagicMock()
    tee = TeeBackend(a, b, c)

    tee.shutdown()  # must not raise

    b.shutdown.assert_called_once()
    c.shutdown.assert_called_once()


def test_force_flush_returns_false_when_any_child_returns_false() -> None:
    """All-must-flush semantics so callers know if a flush window expired."""

    a, b = MagicMock(), MagicMock()
    a.force_flush.return_value = True
    b.force_flush.return_value = False
    tee = TeeBackend(a, b)
    assert tee.force_flush(timeout_millis=1000) is False


def test_force_flush_returns_true_when_all_children_succeed() -> None:
    a, b, c = MagicMock(), MagicMock(), MagicMock()
    a.force_flush.return_value = True
    b.force_flush.return_value = True
    c.force_flush.return_value = True
    tee = TeeBackend(a, b, c)
    assert tee.force_flush() is True


def test_force_flush_routes_raising_child_to_on_error() -> None:
    """A child whose force_flush raises is counted as failed + routed."""

    a = MagicMock()
    a.force_flush.return_value = True
    b = MagicMock()
    b.force_flush.side_effect = RuntimeError("flush exploded")
    errors: list[BaseException] = []
    tee = TeeBackend(a, b, on_error=errors.append)

    assert tee.force_flush() is False
    assert len(errors) == 1
    assert "flush exploded" in str(errors[0])
