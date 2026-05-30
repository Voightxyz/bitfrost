"""SQLite backend — persistent local event log with SQL query mode.

``SQLiteBackend`` writes one row per :class:`~bitfrost.types.EventPayload`
to a local SQLite database in **WAL mode**, so concurrent readers
(``bitfrost watch``, ``bitfrost query``, ``bitfrost serve``) never block
the writer thread that the OTel ``BatchSpanProcessor`` drives.

Schema design
-------------
A **flat** table (one column per first-class field) plus a single
``metadata JSON`` column for the rest. Flat columns let ``bitfrost
query`` users write natural SQL (``SELECT model, COUNT(*) FROM events
GROUP BY model``) without ``json_extract`` boilerplate, and let
covering indexes do their job for the dashboard's "last N events for
agent X" patterns.

Retention
---------
v0.1 ships with a constructor-side retention sweep (``retention_days``,
default 7). On ``__init__`` we ``DELETE`` rows older than the cutoff —
**no background thread, no surprise behaviour at runtime**. For
on-demand cleanup users will run ``bitfrost vacuum --db file.sqlite
--keep-days N`` (shipped in Task #16 alongside the rest of the CLI).
Set ``retention_days <= 0`` to disable the on-init sweep entirely
when the caller manages retention out-of-band.

Threading
---------
SQLite connections aren't thread-safe by default, but
``check_same_thread=False`` plus a process-internal lock around every
write is the canonical pattern for "single-writer, many-readers" use
and matches what the OTel BatchSpanProcessor's worker thread expects.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    session_id TEXT,
    timestamp INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    model TEXT,
    duration_ms INTEGER,
    outcome TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    tool_executed TEXT,
    input TEXT,
    metadata TEXT NOT NULL
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_agent_timestamp ON events(agent_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);",
]

_INSERT_SQL = (
    "INSERT OR REPLACE INTO events "
    "(id, agent_id, session_id, timestamp, event_type, model, duration_ms, "
    "outcome, input_tokens, output_tokens, tool_executed, input, metadata) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class SQLiteBackend:
    """SQLite-backed event log.

    Parameters
    ----------
    path
        Filesystem path to the database file. **Required** — Bitfrost
        deliberately ships no default location so a misconfigured
        backend never silently creates ``./bitfrost.db`` next to the
        user's project. Parent directories are auto-created.
    retention_days
        Rows older than this many days are deleted at construction
        time. Default 7. Set to ``0`` or a negative value to skip the
        on-init sweep entirely (useful when retention is managed by
        an external cron / ``bitfrost vacuum`` command).
    on_error
        Optional callback invoked with any :class:`sqlite3.Error` that
        would otherwise drop an event silently. Defaults to ``None``
        (swallow).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        retention_days: int = 7,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._path = Path(path)
        self._on_error = on_error
        self._lock = threading.Lock()
        self._shutdown = False

        # Auto-create the parent directory so users can drop a SQLite
        # file under ``./.bitfrost/events.db`` without setting it up by
        # hand. Cheap on every init (idempotent).
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # ``check_same_thread=False`` because OTel's BatchSpanProcessor
        # worker thread isn't the same one that built the connection.
        # The class-level ``_lock`` guards every write so we never hit
        # SQLite's "database is locked" path.
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we control transactions per-write
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        for stmt in _INDEXES:
            self._conn.execute(stmt)

        if retention_days > 0:
            self._sweep_retention(retention_days)

    def send(self, event: dict[str, Any]) -> None:
        """Insert one event row. Synchronous (no background thread)."""

        if self._shutdown:
            return

        try:
            row = _event_to_row(event)
        except (TypeError, ValueError, AttributeError) as err:
            # Broad enough to catch a malformed payload (None, missing
            # required dict shape, etc.) without swallowing programmer
            # errors elsewhere. The backend's job is "never crash the
            # host app"; we route the misuse to ``on_error`` and move on.
            self._route_error(err)
            return

        with self._lock:
            try:
                self._conn.execute(_INSERT_SQL, row)
            except sqlite3.Error as err:
                self._route_error(err)

    def shutdown(self) -> None:
        """Close the connection. Idempotent."""

        self._shutdown = True
        with self._lock, suppress(sqlite3.Error):
            self._conn.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op — every ``send`` already committed (autocommit + WAL)."""

        del timeout_millis
        return True

    def __del__(self) -> None:
        """Defensive close if the caller never invoked :meth:`shutdown`.

        Without this, a backend that goes out of scope leaks its sqlite
        connection. Python 3.13 raises a ``ResourceWarning`` when an
        unclosed connection is finalised; closing here pre-empts that so
        a forgetful caller doesn't pay for it. Wrapped in broad suppress
        because ``__del__`` can run during interpreter teardown when
        module globals (sqlite3) may already be gone.
        """

        conn = getattr(self, "_conn", None)
        if conn is not None:
            with suppress(Exception):
                conn.close()

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _sweep_retention(self, retention_days: int) -> None:
        """Delete rows older than ``retention_days`` from the current wall clock.

        Uses millisecond-epoch comparison since that's the timestamp
        granularity we store. Errors are swallowed (init must never
        raise from a retention failure) but routed to ``on_error``.
        """

        import time

        cutoff_ms = int((time.time() - retention_days * 86400) * 1000)
        try:
            self._conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_ms,))
        except sqlite3.Error as err:
            self._route_error(err)

    def _route_error(self, err: BaseException) -> None:
        if self._on_error is None:
            return
        with suppress(BaseException):
            self._on_error(err)


def _event_to_row(event: dict[str, Any]) -> tuple[Any, ...]:
    """Flatten an :class:`EventPayload` into the column tuple ``send`` writes.

    Lifts a handful of frequently-queried fields out of ``metadata``
    into typed columns; the rest of ``metadata`` survives as a
    JSON-encoded string for full fidelity.
    """

    metadata = event.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    event_id = event.get("id") or event.get("eventId") or _gen_id()
    agent_id = event.get("agentId") or metadata.get("agentId") or "unknown"
    session_id = metadata.get("sessionId")
    timestamp_ms = _resolve_timestamp_ms(event, metadata)
    event_type = event.get("type") or "action"
    model = event.get("model")
    duration_ms = event.get("durationMs")
    outcome = event.get("outcome")
    tool_executed = event.get("toolExecuted")

    tokens_raw = metadata.get("tokens")
    tokens: dict[str, Any] = tokens_raw if isinstance(tokens_raw, dict) else {}
    input_tokens = _coerce_int(tokens.get("input"))
    output_tokens = _coerce_int(tokens.get("output"))

    # Preserve the (already privacy-filtered) prompt input so SQLite
    # captures match JSONL fidelity — the TUI / replay detail panels
    # show the prompt, not just the response.
    input_obj = event.get("input")
    input_json = (
        json.dumps(input_obj, separators=(",", ":"), default=str) if input_obj is not None else None
    )

    return (
        str(event_id),
        str(agent_id),
        str(session_id) if session_id else None,
        int(timestamp_ms),
        str(event_type),
        str(model) if model else None,
        int(duration_ms) if isinstance(duration_ms, (int, float)) else None,
        str(outcome) if outcome else None,
        input_tokens,
        output_tokens,
        str(tool_executed) if tool_executed else None,
        input_json,
        json.dumps(metadata, separators=(",", ":"), default=str),
    )


def _resolve_timestamp_ms(event: dict[str, Any], metadata: dict[str, Any]) -> int:
    """Return a millisecond-epoch timestamp from whatever the event carries.

    Priority: ``event['timestamp']`` (ms or ISO) → ``metadata['timestamp']``
    → current wall clock. Strings are best-effort parsed; on any parse
    failure we fall back to wall clock so a write never fails on a
    cosmetic field.
    """

    import time

    raw = event.get("timestamp") or metadata.get("timestamp")
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        with suppress(ValueError):
            from datetime import datetime

            cleaned = raw.replace("Z", "+00:00")
            return int(datetime.fromisoformat(cleaned).timestamp() * 1000)
    return int(time.time() * 1000)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gen_id() -> str:
    """Fallback event id when the producer didn't stamp one.

    Uses :func:`uuid.uuid4` — Voight's ingest path always provides a
    cuid in the response, but ``SQLiteBackend`` is also used standalone
    (no Voight backend in the pipeline) where no upstream id exists.
    """

    import uuid

    return uuid.uuid4().hex


__all__ = ["SQLiteBackend"]
