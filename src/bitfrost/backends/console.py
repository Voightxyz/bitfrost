"""Console backend — emit each event as a single tail-f-style line to stdout.

This is the **standalone-friendly** backend: it requires no Voight key,
no network, and no on-disk state. Drop it into ``BitfrostExporter`` and
every captured LLM call surfaces as one styled line on the developer's
terminal, suitable for live demos and quick local debugging.

Usage::

    from bitfrost.exporter import BitfrostExporter
    from bitfrost.backends.console import ConsoleBackend

    exporter = BitfrostExporter(
        backend=ConsoleBackend(),
        agent="my-app",
        privacy="standard",
    )

When ``rich`` is installed (the ``[rich]`` extra) the output is
ANSI-coloured. Without it, the renderer falls back to plain text — so
``ConsoleBackend`` never breaks on a missing optional dep.

The backend is also colour-aware about the runtime: a non-TTY stdout
(piped to a file, captured in CI logs) automatically renders plain
text even when ``rich`` is available. This matches the established
``NO_COLOR`` convention and keeps test output diffable.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Any, TextIO

from bitfrost.console_renderer import render_event


class ConsoleBackend:
    """Standalone, zero-config backend that prints events to a stream.

    Parameters
    ----------
    stream
        Destination text stream. Defaults to :data:`sys.stdout`. Pass
        :data:`sys.stderr` for separation from app output, or any
        writeable file-like for log capture.
    colorize
        ``True`` to emit ANSI colour codes (when ``rich`` is installed),
        ``False`` to force plain text, or ``None`` (default) to
        auto-detect based on TTY status of the stream.
    on_error
        Optional callback for renderer / write failures. The backend
        itself never raises into the caller's hot path.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        colorize: bool | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._colorize = self._resolve_colorize(colorize)
        self._on_error = on_error
        self._lock = threading.Lock()
        self._shutdown = False

    def send(self, event: dict[str, Any]) -> None:
        """Render one event and write it to the configured stream."""

        if self._shutdown:
            return
        try:
            line = render_event(event, colorize=self._colorize)
        except BaseException as err:
            self._route_error(err)
            return

        with self._lock:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except BaseException as err:
                self._route_error(err)

    def shutdown(self) -> None:
        """Flush + mark drained. Idempotent."""

        self._shutdown = True
        with self._lock, suppress(BaseException):
            self._stream.flush()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush the stream so buffered output reaches the terminal."""

        del timeout_millis
        with self._lock, suppress(BaseException):
            self._stream.flush()
        return True

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _resolve_colorize(self, colorize: bool | None) -> bool:
        """Auto-detect colour eligibility unless the caller forced a value.

        Auto behaviour:
        - Explicit ``True`` / ``False`` from the caller wins.
        - ``None`` → True when the stream is a TTY, False otherwise.

        The renderer itself layers on ``NO_COLOR`` env handling, so we
        don't need to duplicate that check here.
        """

        if colorize is not None:
            return colorize
        isatty = getattr(self._stream, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    def _route_error(self, err: BaseException) -> None:
        if self._on_error is None:
            return
        with suppress(BaseException):
            self._on_error(err)


__all__ = ["ConsoleBackend"]
