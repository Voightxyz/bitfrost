"""File-logging backend — one JSON line per event.

``JSONLBackend`` writes every captured event as a single line of JSON to a
local file. It's the canonical input format for ``bitfrost replay``: a run
captured to ``.jsonl`` today can be re-emitted to any backend tomorrow.

Why JSON Lines (not pretty-printed JSON, not a single array)
------------------------------------------------------------
- **Append-only** — concurrent writes don't fight over a closing ``]``.
- **Stream-friendly** — each line is a complete record, so partial files
  are still parseable.
- **Tool-friendly** — ``jq`` / ``rg`` / ``grep`` understand it natively.

Hot-path safety
---------------
Writes happen synchronously on the calling thread (the OTel
``BatchSpanProcessor`` worker). The file is opened in append mode for the
lifetime of the backend, then closed on :meth:`shutdown`. Serialisation
errors route to ``on_error`` and never crash the caller.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any


class JSONLBackend:
    """Append each event to a ``.jsonl`` file as one line of JSON.

    Parameters
    ----------
    path
        Destination file. Parent directories are created on first write.
    on_error
        Callback invoked on serialisation failures. Defaults to ``None``
        (silent).
    encoding
        File encoding. ``utf-8`` is the only sensible choice for JSON,
        exposed as a parameter for completeness.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        on_error: Callable[[BaseException], None] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        if not path:
            msg = "JSONLBackend requires a non-empty file path."
            raise ValueError(msg)
        self._path = Path(path)
        self._encoding = encoding
        self._on_error = on_error
        self._lock = threading.Lock()
        self._file: IO[str] | None = None
        self._shutdown_called = False

    # ----- public API -------------------------------------------------------

    def send(self, event: dict[str, Any]) -> None:
        if self._shutdown_called:
            return
        try:
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as err:
            self._route_error(err)
            return

        with self._lock:
            try:
                fh = self._ensure_open()
                fh.write(line)
                fh.write("\n")
                fh.flush()
            except OSError as err:
                self._route_error(err)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_called = True
            if self._file is not None:
                with contextlib.suppress(Exception):
                    self._file.close()
                self._file = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis  # Reserved for future buffering support.
        with self._lock, contextlib.suppress(Exception):
            if self._file is not None:
                self._file.flush()
        return True

    # ----- internals --------------------------------------------------------

    def _ensure_open(self) -> IO[str]:
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding=self._encoding)
        return self._file

    def _route_error(self, err: BaseException) -> None:
        if self._on_error is None:
            return
        with contextlib.suppress(BaseException):
            self._on_error(err)


__all__ = ["JSONLBackend"]
