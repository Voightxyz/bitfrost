"""Tests for :mod:`bitfrost.backends.console`."""

from __future__ import annotations

import io
import threading
from typing import Any
from unittest.mock import patch

from bitfrost.backends.console import ConsoleBackend


def _make_event(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "agentId": "console-test",
        "type": "action",
        "model": "gpt-4o-mini",
        "durationMs": 100,
        "outcome": "success",
        "metadata": {"tokens": {"input": 10, "output": 5}},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Basic send / output
# ---------------------------------------------------------------------------


def test_send_writes_one_line_to_configured_stream() -> None:
    sink = io.StringIO()
    backend = ConsoleBackend(stream=sink, colorize=False)
    backend.send(_make_event())

    out = sink.getvalue()
    assert out.count("\n") == 1
    assert "console-test" in out
    assert "gpt-4o-mini" in out
    assert "success" in out


def test_send_appends_trailing_newline_so_lines_dont_merge() -> None:
    sink = io.StringIO()
    backend = ConsoleBackend(stream=sink, colorize=False)
    backend.send(_make_event())
    backend.send(_make_event())
    assert sink.getvalue().count("\n") == 2


def test_send_after_shutdown_is_silent_drop() -> None:
    sink = io.StringIO()
    backend = ConsoleBackend(stream=sink, colorize=False)
    backend.shutdown()
    backend.send(_make_event())
    assert sink.getvalue() == ""


# ---------------------------------------------------------------------------
# Colour auto-detection
# ---------------------------------------------------------------------------


def test_colorize_explicit_true_emits_ansi() -> None:
    sink = io.StringIO()
    backend = ConsoleBackend(stream=sink, colorize=True)
    backend.send(_make_event())
    assert "\x1b[" in sink.getvalue()


def test_colorize_explicit_false_emits_plain() -> None:
    sink = io.StringIO()
    backend = ConsoleBackend(stream=sink, colorize=False)
    backend.send(_make_event())
    assert "\x1b[" not in sink.getvalue()


def test_colorize_auto_off_when_stream_is_not_tty() -> None:
    """``io.StringIO`` reports ``isatty() == False`` — colour stays off."""

    sink = io.StringIO()
    backend = ConsoleBackend(stream=sink, colorize=None)
    backend.send(_make_event())
    assert "\x1b[" not in sink.getvalue()


def test_colorize_auto_on_when_stream_is_tty() -> None:
    """A stream that fakes ``isatty() → True`` opts into colour."""

    class _TTYStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    sink = _TTYStream()
    backend = ConsoleBackend(stream=sink, colorize=None)
    backend.send(_make_event())
    assert "\x1b[" in sink.getvalue()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_shutdown_is_idempotent() -> None:
    backend = ConsoleBackend(stream=io.StringIO(), colorize=False)
    backend.shutdown()
    backend.shutdown()  # must not raise


def test_force_flush_returns_true_and_drains_stream() -> None:
    """``force_flush`` calls ``stream.flush()`` and returns True."""

    flush_count = {"n": 0}

    class _Counting(io.StringIO):
        def flush(self) -> None:
            flush_count["n"] += 1

    backend = ConsoleBackend(stream=_Counting(), colorize=False)
    backend.send(_make_event())  # one flush
    flushed_before = flush_count["n"]
    assert backend.force_flush(timeout_millis=1000) is True
    assert flush_count["n"] > flushed_before


def test_force_flush_after_shutdown_still_returns_true() -> None:
    backend = ConsoleBackend(stream=io.StringIO(), colorize=False)
    backend.shutdown()
    assert backend.force_flush() is True


# ---------------------------------------------------------------------------
# Error routing + thread safety
# ---------------------------------------------------------------------------


def test_renderer_exception_routed_to_on_error_not_raised() -> None:
    """A bug in the renderer must NOT propagate up the BatchProcessor stack."""

    errors: list[BaseException] = []
    backend = ConsoleBackend(stream=io.StringIO(), colorize=False, on_error=errors.append)
    with patch(
        "bitfrost.backends.console.render_event",
        side_effect=RuntimeError("renderer is on fire"),
    ):
        backend.send(_make_event())

    assert len(errors) == 1
    assert "renderer is on fire" in str(errors[0])


def test_stream_write_exception_routed_to_on_error() -> None:
    """A failing stream (closed pipe, full disk) routes the error and survives."""

    class _BrokenStream:
        def write(self, _: str) -> int:
            raise OSError("pipe closed")

        def flush(self) -> None:
            return None

        def isatty(self) -> bool:
            return False

    errors: list[BaseException] = []
    backend = ConsoleBackend(
        stream=_BrokenStream(),  # type: ignore[arg-type]
        colorize=False,
        on_error=errors.append,
    )
    backend.send(_make_event())
    assert len(errors) == 1
    assert isinstance(errors[0], OSError)


def test_send_under_concurrent_threads_serialises_lines_cleanly() -> None:
    """The internal lock prevents interleaved bytes from multiple writers."""

    sink = io.StringIO()
    backend = ConsoleBackend(stream=sink, colorize=False)

    def worker() -> None:
        for _ in range(25):
            backend.send(_make_event())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = sink.getvalue().splitlines()
    # Every line should be a complete render — no partial fragments.
    assert len(lines) == 4 * 25
    for line in lines:
        assert "console-test" in line
        assert "success" in line
