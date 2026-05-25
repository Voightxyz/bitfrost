"""Tests for :mod:`bitfrost.backends.jsonl`.

``JSONLBackend`` appends each event as one JSON line to a file. The file
becomes the input for ``bitfrost replay``, so the format must be stable
and one-line-per-event.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bitfrost.backends.jsonl import JSONLBackend


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_jsonl_backend_creates_parent_dir_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "spans.jsonl"
    backend = JSONLBackend(target)
    backend.send({"x": 1})
    backend.shutdown()
    assert target.exists()


def test_jsonl_backend_requires_path() -> None:
    with pytest.raises(ValueError, match="path"):
        JSONLBackend("")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Send semantics
# ---------------------------------------------------------------------------


def test_send_appends_one_event_per_line(tmp_path: Path) -> None:
    target = tmp_path / "spans.jsonl"
    backend = JSONLBackend(target)
    backend.send({"id": 1, "model": "gpt-4o-mini"})
    backend.send({"id": 2, "model": "claude-haiku-4-5"})
    backend.shutdown()
    lines = _read_lines(target)
    assert lines == [
        {"id": 1, "model": "gpt-4o-mini"},
        {"id": 2, "model": "claude-haiku-4-5"},
    ]


def test_send_preserves_unicode_content(tmp_path: Path) -> None:
    """CJK + emoji + accented characters must round-trip without escapes."""
    target = tmp_path / "spans.jsonl"
    backend = JSONLBackend(target)
    backend.send({"prompt": "你好,世界 ☀️ — café résumé"})
    backend.shutdown()
    lines = _read_lines(target)
    assert lines[0]["prompt"] == "你好,世界 ☀️ — café résumé"


def test_send_after_shutdown_is_a_noop(tmp_path: Path) -> None:
    target = tmp_path / "spans.jsonl"
    backend = JSONLBackend(target)
    backend.send({"id": 1})
    backend.shutdown()
    backend.send({"id": 2})  # dropped silently
    assert _read_lines(target) == [{"id": 1}]


def test_send_swallows_serialisation_errors(tmp_path: Path) -> None:
    """Non-JSON values must NOT crash the OTel pipeline; they route to on_error."""
    target = tmp_path / "spans.jsonl"
    errors: list[BaseException] = []
    backend = JSONLBackend(target, on_error=errors.append)

    class Unserialisable:
        pass

    backend.send({"weird": Unserialisable()})  # type: ignore[dict-item]
    backend.shutdown()
    assert len(errors) == 1
    assert isinstance(errors[0], TypeError)
    # File may not exist if the bad payload was the only send call.
    if target.exists():
        assert target.read_text("utf-8").strip() == ""


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_shutdown_is_idempotent(tmp_path: Path) -> None:
    backend = JSONLBackend(tmp_path / "spans.jsonl")
    backend.shutdown()
    backend.shutdown()  # must not raise


def test_force_flush_returns_true(tmp_path: Path) -> None:
    backend = JSONLBackend(tmp_path / "spans.jsonl")
    assert backend.force_flush() is True
