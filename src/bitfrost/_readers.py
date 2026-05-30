"""Readers that turn a captured log (JSONL file or SQLite DB) back into events.

The CLI commands (``watch``, ``replay``, ``query``) all need to pull events
out of a file the user captured earlier. Two storage formats ship in v0.1:

- **JSONL** (:class:`~bitfrost.backends.jsonl.JSONLBackend`) — one JSON
  object per line, each line a full :class:`~bitfrost.types.EventPayload`.
- **SQLite** (:class:`~bitfrost.backends.sqlite.SQLiteBackend`) — flat
  columns plus a ``metadata`` JSON column.

Both readers expose the same surface:

- ``read_all()`` → every event as a payload dict, in capture order.
- ``tail(marker)`` → ``(new_events, new_marker)`` since the last poll, so
  ``bitfrost watch`` can incrementally stream new events without re-reading
  the whole file. The marker is opaque to callers: a byte offset for
  JSONL, a rowid for SQLite. Start a watch loop with ``marker=0``.

:class:`SQLiteReader` additionally exposes ``query(sql)`` — a **read-only**
SQL passthrough for ``bitfrost query`` — opened with SQLite's ``mode=ro``
URI so a fat-fingered ``DROP TABLE`` can't damage the user's capture.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any


class JSONLReader:
    """Read events from a ``.jsonl`` capture file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def read_all(self) -> list[dict[str, Any]]:
        """Return every well-formed event in the file, in order.

        Blank lines and malformed JSON are skipped silently — a partially
        written final line (the writer crashed mid-append) shouldn't make
        the whole replay unreadable.
        """

        events: list[dict[str, Any]] = []
        if not self._path.exists():
            return events
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                event = _parse_line(line)
                if event is not None:
                    events.append(event)
        return events

    def tail(self, marker: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Return events appended since byte offset ``marker``.

        Reads only complete lines: if the file currently ends mid-line
        (a write in progress), the partial trailing line is left for the
        next poll by rewinding the returned marker to the start of it.
        """

        if not self._path.exists():
            return [], marker
        events: list[dict[str, Any]] = []
        with self._path.open("rb") as fh:
            fh.seek(marker)
            data = fh.read()
            new_marker = fh.tell()
        if not data:
            return [], marker
        text = data.decode("utf-8", "replace")
        # If the chunk doesn't end on a newline, hold back the partial
        # last line so we re-read it whole next time.
        if not text.endswith("\n"):
            last_nl = text.rfind("\n")
            if last_nl == -1:
                # No complete line yet — wait for more.
                return [], marker
            consumed = last_nl + 1
            new_marker = marker + len(text[:consumed].encode("utf-8"))
            text = text[:consumed]
        for line in text.splitlines():
            event = _parse_line(line)
            if event is not None:
                events.append(event)
        return events, new_marker


class SQLiteReader:
    """Read events from a SQLite capture DB written by ``SQLiteBackend``."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def read_all(self) -> list[dict[str, Any]]:
        """Return every event reconstructed into payload-dict shape, in order."""

        if not self._path.exists():
            return []
        conn = self._connect_ro()
        try:
            cursor = conn.execute(
                "SELECT rowid, agent_id, session_id, timestamp, event_type, "
                "model, duration_ms, outcome, tool_executed, input, metadata "
                "FROM events ORDER BY rowid"
            )
            return [_row_to_event(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def tail(self, marker: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Return events with ``rowid > marker``; new marker is the max rowid.

        rowid is SQLite's monotonic insertion counter, so it gives a stable
        "everything since last poll" cursor even when two events share a
        millisecond timestamp.
        """

        if not self._path.exists():
            return [], marker
        conn = self._connect_ro()
        try:
            cursor = conn.execute(
                "SELECT rowid, agent_id, session_id, timestamp, event_type, "
                "model, duration_ms, outcome, tool_executed, input, metadata "
                "FROM events WHERE rowid > ? ORDER BY rowid",
                (marker,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()
        if not rows:
            return [], marker
        events = [_row_to_event(row) for row in rows]
        new_marker = max(int(row[0]) for row in rows)
        return events, new_marker

    def query(self, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Run a read-only SQL query, returning ``(column_names, rows)``.

        The connection is opened ``mode=ro`` so any mutation (INSERT,
        UPDATE, DELETE, DROP, …) raises ``sqlite3.OperationalError`` rather
        than touching the user's capture. ``bitfrost query`` surfaces that
        as a friendly error.
        """

        conn = self._connect_ro()
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return columns, rows
        finally:
            conn.close()

    def _connect_ro(self) -> sqlite3.Connection:
        """Open a read-only connection via the ``file:…?mode=ro`` URI."""

        uri = f"file:{self._path}?mode=ro"
        return sqlite3.connect(uri, uri=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_line(line: str) -> dict[str, Any] | None:
    """Parse one JSONL line into an event dict, or ``None`` if not usable."""

    stripped = line.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _row_to_event(row: tuple[Any, ...]) -> dict[str, Any]:
    """Reconstruct a render-ready event dict from a SQLite row.

    Column order matches the SELECT in :meth:`SQLiteReader.read_all`:
    ``(rowid, agent_id, session_id, timestamp, event_type, model,
    duration_ms, outcome, tool_executed, input, metadata)``.

    ``metadata`` already carries ``tokens`` / ``provider`` / ``sessionId``
    and ``input`` carries the (privacy-filtered) prompt, so the
    reconstructed dict renders identically to a live event.
    """

    (
        _rowid,
        agent_id,
        _session_id,
        timestamp,
        event_type,
        model,
        duration_ms,
        outcome,
        tool_executed,
        input_raw,
        metadata_raw,
    ) = row
    try:
        metadata = json.loads(metadata_raw) if metadata_raw else {}
    except (json.JSONDecodeError, ValueError, TypeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    event: dict[str, Any] = {
        "agentId": agent_id,
        "type": event_type,
        "model": model,
        "durationMs": duration_ms,
        "outcome": outcome,
        "timestamp": timestamp,
        "metadata": metadata,
    }
    if tool_executed:
        event["toolExecuted"] = tool_executed
    if input_raw:
        with contextlib.suppress(json.JSONDecodeError, ValueError, TypeError):
            event["input"] = json.loads(input_raw)
    return event


def make_reader(path: str | Path, fmt: str | None = None) -> JSONLReader | SQLiteReader:
    """Return the right reader for ``path``, auto-detecting by extension.

    ``fmt`` overrides detection: ``"jsonl"`` or ``"sqlite"`` / ``"db"``.
    Detection falls back to SQLite for ``.db`` / ``.sqlite`` / ``.sqlite3``
    and JSONL for ``.jsonl`` / ``.ndjson``; anything else raises so the CLI
    can tell the user to pass an explicit flag.
    """

    if fmt is not None:
        fmt_l = fmt.lower()
        if fmt_l == "jsonl":
            return JSONLReader(path)
        if fmt_l in ("sqlite", "db"):
            return SQLiteReader(path)
        msg = f"unknown reader format: {fmt!r} (expected 'jsonl' or 'sqlite')"
        raise ValueError(msg)

    suffix = Path(path).suffix.lower()
    if suffix in (".db", ".sqlite", ".sqlite3"):
        return SQLiteReader(path)
    if suffix in (".jsonl", ".ndjson"):
        return JSONLReader(path)
    msg = f"cannot infer format from extension {suffix!r}; pass --db or --jsonl explicitly"
    raise ValueError(msg)


__all__ = ["JSONLReader", "SQLiteReader", "make_reader"]
