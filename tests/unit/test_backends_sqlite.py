"""Tests for :mod:`bitfrost.backends.sqlite`.

Strategy
--------
The backend writes synchronously into a real SQLite file (a tmp_path)
because:
- WAL-mode activation can only be asserted against a real file
- The integration with json_extract under the hood matters for the
  flat-columns-plus-metadata schema design
- The retention sweep is non-trivial; mocking time around it is more
  fragile than asserting against real rows

Every test uses ``tmp_path`` so files are auto-cleaned.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from bitfrost.backends.sqlite import SQLiteBackend


def _make_event(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "evt_test_1",
        "agentId": "test-agent",
        "type": "action",
        "model": "gpt-4o-mini",
        "durationMs": 150,
        "outcome": "success",
        "metadata": {
            "source": "bitfrost",
            "provider": "openai",
            "sessionId": "sess-abc",
            "tokens": {"input": 12, "output": 5, "total": 17},
            "spanName": "openai.chat",
            "privacyLevel": "standard",
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Init / schema
# ---------------------------------------------------------------------------


def test_init_creates_events_table_and_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    SQLiteBackend(db_path)

    conn = sqlite3.connect(str(db_path))
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex%'"
    ).fetchall()
    conn.close()

    assert ("events",) in tables
    index_names = {row[0] for row in indexes}
    assert "idx_events_agent_timestamp" in index_names
    assert "idx_events_session" in index_names
    assert "idx_events_timestamp" in index_names


def test_init_activates_wal_mode(tmp_path: Path) -> None:
    """WAL mode must be active so concurrent readers don't block the writer.

    Asserting against a fresh connection (not the backend's own) confirms
    the mode is persisted to the file, not just the in-memory connection
    state.
    """

    db_path = tmp_path / "wal.db"
    SQLiteBackend(db_path)

    conn = sqlite3.connect(str(db_path))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal"


def test_init_auto_creates_parent_directory(tmp_path: Path) -> None:
    """Bitfrost users can drop a SQLite file under a missing subdir."""

    nested = tmp_path / "deep" / "nested" / "events.db"
    assert not nested.parent.exists()
    SQLiteBackend(nested)
    assert nested.exists()


def test_path_required_explicit_typeerror() -> None:
    """No default path — calling without one raises clearly.

    Bitfrost intentionally ships no default file location to avoid
    silently writing ``./bitfrost.db`` into the user's project.
    """

    with pytest.raises(TypeError):
        SQLiteBackend()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# send / round-trip
# ---------------------------------------------------------------------------


def test_send_inserts_row_with_flat_columns_populated(tmp_path: Path) -> None:
    db_path = tmp_path / "send.db"
    backend = SQLiteBackend(db_path)
    backend.send(_make_event())
    backend.shutdown()

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT id, agent_id, session_id, event_type, model, duration_ms, "
        "outcome, input_tokens, output_tokens FROM events"
    ).fetchone()
    conn.close()

    assert row == (
        "evt_test_1",
        "test-agent",
        "sess-abc",
        "action",
        "gpt-4o-mini",
        150,
        "success",
        12,
        5,
    )


def test_send_preserves_full_metadata_as_json_column(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.db"
    backend = SQLiteBackend(db_path)
    backend.send(_make_event())
    backend.shutdown()

    conn = sqlite3.connect(str(db_path))
    raw = conn.execute("SELECT metadata FROM events").fetchone()[0]
    conn.close()
    parsed = json.loads(raw)
    assert parsed["source"] == "bitfrost"
    assert parsed["provider"] == "openai"
    assert parsed["tokens"]["total"] == 17
    assert parsed["spanName"] == "openai.chat"


def test_send_json_extract_works_on_metadata(tmp_path: Path) -> None:
    """``bitfrost query`` users rely on json_extract over the metadata column."""

    db_path = tmp_path / "extract.db"
    backend = SQLiteBackend(db_path)
    backend.send(_make_event())
    backend.shutdown()

    conn = sqlite3.connect(str(db_path))
    provider = conn.execute("SELECT json_extract(metadata, '$.provider') FROM events").fetchone()[0]
    conn.close()
    assert provider == "openai"


def test_send_after_shutdown_is_silent_drop(tmp_path: Path) -> None:
    db_path = tmp_path / "shut.db"
    backend = SQLiteBackend(db_path)
    backend.shutdown()
    backend.send(_make_event())

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    assert count == 0


def test_shutdown_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "idempotent.db"
    backend = SQLiteBackend(db_path)
    backend.shutdown()
    backend.shutdown()  # must not raise


def test_send_threadsafe_under_concurrent_writers(tmp_path: Path) -> None:
    """Multiple writer threads must all succeed under the internal lock."""

    db_path = tmp_path / "concurrent.db"
    backend = SQLiteBackend(db_path)

    def worker(i: int) -> None:
        for j in range(20):
            backend.send(_make_event(id=f"evt_{i}_{j}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    backend.shutdown()
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    assert count == 4 * 20


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_retention_sweeps_rows_older_than_cutoff(tmp_path: Path) -> None:
    """A row older than ``retention_days`` is deleted on init."""

    db_path = tmp_path / "retention.db"
    # First instance, no retention applied — we'll seed an old row.
    backend = SQLiteBackend(db_path, retention_days=0)
    old_ms = int((time.time() - 30 * 86400) * 1000)  # 30 days ago
    new_ms = int(time.time() * 1000)
    backend.send(_make_event(id="old", timestamp=old_ms))
    backend.send(_make_event(id="new", timestamp=new_ms))
    backend.shutdown()

    # Second instance with 7-day retention runs the sweep.
    SQLiteBackend(db_path, retention_days=7).shutdown()

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT id FROM events ORDER BY id").fetchall()
    conn.close()
    assert rows == [("new",)]


def test_retention_skipped_when_days_is_zero_or_negative(tmp_path: Path) -> None:
    """``retention_days <= 0`` disables the on-init sweep entirely."""

    db_path = tmp_path / "no_retention.db"
    backend = SQLiteBackend(db_path, retention_days=0)
    old_ms = int((time.time() - 365 * 86400) * 1000)
    backend.send(_make_event(id="ancient", timestamp=old_ms))
    backend.shutdown()

    # Re-open with retention_days=-1 — must NOT delete the old row.
    SQLiteBackend(db_path, retention_days=-1).shutdown()

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT id FROM events").fetchall()
    conn.close()
    assert rows == [("ancient",)]


def test_retention_runs_on_init_not_on_send(tmp_path: Path) -> None:
    """Sweep is one-shot at construction, not a per-write check.

    A row inserted AFTER construction must survive even if it's older
    than the cutoff (its insertion timestamp is what we'd use, but the
    sweep already ran).
    """

    db_path = tmp_path / "on_init.db"
    backend = SQLiteBackend(db_path, retention_days=7)
    # Insert a row with an artificially old timestamp AFTER init.
    very_old = int((time.time() - 30 * 86400) * 1000)
    backend.send(_make_event(id="post-init-old", timestamp=very_old))
    backend.shutdown()

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT id FROM events").fetchall()
    conn.close()
    # The sweep ran on init when the table was empty, then the
    # backdated row was inserted afterwards — it survives.
    assert rows == [("post-init-old",)]


# ---------------------------------------------------------------------------
# Error routing
# ---------------------------------------------------------------------------


def test_serialisation_failure_routed_to_on_error(tmp_path: Path) -> None:
    """A non-serialisable metadata value routes to on_error.

    The ``default=str`` json fallback handles most edge cases, so the
    primary remaining failure path is a payload that lacks required
    fields. Here we pass a clearly broken shape (None) — ``_event_to_row``
    raises a TypeError when iterating ``None.get(...)`` and the backend
    must catch and route, not crash.
    """

    db_path = tmp_path / "broken.db"
    errors: list[BaseException] = []
    backend = SQLiteBackend(db_path, on_error=errors.append)
    backend.send(None)  # type: ignore[arg-type]
    backend.shutdown()

    assert len(errors) == 1
    assert isinstance(errors[0], (TypeError, ValueError, AttributeError))
