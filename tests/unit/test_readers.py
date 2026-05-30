"""Tests for :mod:`bitfrost._readers`.

The readers feed every CLI command, so these pin the contracts the CLI
relies on: full read-back, incremental tail with a stable marker,
read-only SQL, malformed-line tolerance, and extension auto-detection.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from bitfrost._readers import JSONLReader, SQLiteReader, make_reader
from bitfrost.backends.jsonl import JSONLBackend
from bitfrost.backends.sqlite import SQLiteBackend


def _event(i: int) -> dict[str, Any]:
    return {
        "agentId": "reader-test",
        "type": "action",
        "model": "gpt-4o-mini",
        "durationMs": 100 + i,
        "outcome": "success",
        "metadata": {
            "provider": "openai",
            "sessionId": f"s{i}",
            "tokens": {"input": 10 + i, "output": i},
        },
    }


# ---------------------------------------------------------------------------
# JSONLReader
# ---------------------------------------------------------------------------


def test_jsonl_read_all_returns_every_event_in_order(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    backend = JSONLBackend(path)
    for i in range(3):
        backend.send(_event(i))
    backend.shutdown()

    events = JSONLReader(path).read_all()
    assert len(events) == 3
    assert [e["metadata"]["sessionId"] for e in events] == ["s0", "s1", "s2"]


def test_jsonl_read_all_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "messy.jsonl"
    path.write_text(
        json.dumps(_event(0)) + "\n"
        "\n"  # blank
        "{not valid json\n"  # malformed (writer crashed mid-append)
         + json.dumps(_event(1)) + "\n",
        encoding="utf-8",
    )
    events = JSONLReader(path).read_all()
    assert len(events) == 2


def test_jsonl_read_all_empty_when_file_missing(tmp_path: Path) -> None:
    assert JSONLReader(tmp_path / "nope.jsonl").read_all() == []


def test_jsonl_tail_returns_only_new_events_since_marker(tmp_path: Path) -> None:
    path = tmp_path / "tail.jsonl"
    backend = JSONLBackend(path)
    backend.send(_event(0))
    reader = JSONLReader(path)
    first_batch, marker = reader.tail(0)
    assert len(first_batch) == 1

    backend.send(_event(1))
    backend.send(_event(2))
    backend.shutdown()
    second_batch, _ = reader.tail(marker)
    assert len(second_batch) == 2
    assert second_batch[0]["metadata"]["sessionId"] == "s1"


def test_jsonl_tail_holds_back_partial_trailing_line(tmp_path: Path) -> None:
    """A half-written final line is not parsed until it's complete."""

    path = tmp_path / "partial.jsonl"
    # One complete line + a partial second line (no trailing newline).
    path.write_text(json.dumps(_event(0)) + "\n" + '{"agentId":"x","ty', encoding="utf-8")
    events, marker = JSONLReader(path).tail(0)
    assert len(events) == 1  # only the complete line

    # Complete the partial line; next tail picks it up.
    with path.open("a", encoding="utf-8") as fh:
        fh.write('pe":"action"}\n')
    events2, _ = JSONLReader(path).tail(marker)
    assert len(events2) == 1


# ---------------------------------------------------------------------------
# SQLiteReader
# ---------------------------------------------------------------------------


def test_sqlite_read_all_reconstructs_event_shape(tmp_path: Path) -> None:
    path = tmp_path / "e.db"
    backend = SQLiteBackend(path, retention_days=0)
    backend.send(_event(0))
    backend.send(_event(1))
    backend.shutdown()

    events = SQLiteReader(path).read_all()
    assert len(events) == 2
    first = events[0]
    assert first["agentId"] == "reader-test"
    assert first["model"] == "gpt-4o-mini"
    assert first["metadata"]["provider"] == "openai"
    assert first["metadata"]["tokens"]["input"] == 10


def test_sqlite_tail_returns_new_rows_since_rowid_marker(tmp_path: Path) -> None:
    path = tmp_path / "tail.db"
    backend = SQLiteBackend(path, retention_days=0)
    backend.send(_event(0))
    reader = SQLiteReader(path)
    first_batch, marker = reader.tail(0)
    assert len(first_batch) == 1
    assert marker == 1  # first rowid

    backend.send(_event(1))
    backend.send(_event(2))
    backend.shutdown()
    second_batch, new_marker = reader.tail(marker)
    assert len(second_batch) == 2
    assert new_marker == 3


def test_sqlite_query_returns_columns_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "q.db"
    backend = SQLiteBackend(path, retention_days=0)
    for i in range(3):
        backend.send(_event(i))
    backend.shutdown()

    cols, rows = SQLiteReader(path).query("SELECT model, COUNT(*) AS n FROM events GROUP BY model")
    assert cols == ["model", "n"]
    assert rows == [("gpt-4o-mini", 3)]


def test_sqlite_query_is_read_only_rejects_mutation(tmp_path: Path) -> None:
    """A mutation must be refused by the read-only connection."""

    path = tmp_path / "ro.db"
    backend = SQLiteBackend(path, retention_days=0)
    backend.send(_event(0))
    backend.shutdown()

    with pytest.raises(sqlite3.OperationalError):
        SQLiteReader(path).query("DELETE FROM events")
    # Confirm the row survived the rejected mutation.
    _cols, rows = SQLiteReader(path).query("SELECT COUNT(*) FROM events")
    assert rows == [(1,)]


# ---------------------------------------------------------------------------
# make_reader — auto-detection
# ---------------------------------------------------------------------------


def test_make_reader_detects_sqlite_by_extension(tmp_path: Path) -> None:
    assert isinstance(make_reader(tmp_path / "x.db"), SQLiteReader)
    assert isinstance(make_reader(tmp_path / "x.sqlite"), SQLiteReader)


def test_make_reader_detects_jsonl_by_extension(tmp_path: Path) -> None:
    assert isinstance(make_reader(tmp_path / "x.jsonl"), JSONLReader)


def test_make_reader_explicit_fmt_overrides_extension(tmp_path: Path) -> None:
    assert isinstance(make_reader(tmp_path / "x.weird", fmt="jsonl"), JSONLReader)
    assert isinstance(make_reader(tmp_path / "x.weird", fmt="db"), SQLiteReader)


def test_make_reader_raises_on_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot infer format"):
        make_reader(tmp_path / "x.weird")
