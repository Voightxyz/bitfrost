"""Tests for :mod:`bitfrost.cli` via typer's CliRunner.

These exercise the four commands end-to-end against real capture files
(JSONL + SQLite) written by the backends, asserting on exit codes,
rendered output, and the read-only / confirmation guards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bitfrost.backends.jsonl import JSONLBackend
from bitfrost.backends.sqlite import SQLiteBackend
from bitfrost.cli import app

runner = CliRunner()


def _event(i: int) -> dict[str, Any]:
    return {
        "agentId": "cli-test",
        "type": "action",
        "model": "gpt-4o-mini",
        "durationMs": 100 + i,
        "outcome": "success",
        "metadata": {
            "provider": "openai",
            "sessionId": f"s{i}",
            "tokens": {"input": 10 + i, "output": i + 1},
        },
    }


@pytest.fixture
def jsonl_capture(tmp_path: Path) -> Path:
    path = tmp_path / "capture.jsonl"
    backend = JSONLBackend(path)
    for i in range(3):
        backend.send(_event(i))
    backend.shutdown()
    return path


@pytest.fixture
def sqlite_capture(tmp_path: Path) -> Path:
    path = tmp_path / "capture.db"
    backend = SQLiteBackend(path, retention_days=0)
    for i in range(3):
        backend.send(_event(i))
    backend.shutdown()
    return path


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def test_help_lists_all_four_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("watch", "replay", "query", "vacuum"):
        assert cmd in result.output


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_jsonl_renders_all_events(jsonl_capture: Path) -> None:
    result = runner.invoke(app, ["replay", str(jsonl_capture), "--no-color"])
    assert result.exit_code == 0
    assert result.output.count("cli-test") == 3
    assert "replayed 3 events" in result.output


def test_replay_sqlite_renders_all_events(sqlite_capture: Path) -> None:
    result = runner.invoke(app, ["replay", str(sqlite_capture), "--no-color"])
    assert result.exit_code == 0
    assert result.output.count("cli-test") == 3


def test_replay_empty_file_reports_no_events(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["replay", str(empty)])
    assert result.exit_code == 0
    assert "no events" in result.output


def test_replay_unknown_extension_errors(tmp_path: Path) -> None:
    weird = tmp_path / "capture.weird"
    weird.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["replay", str(weird)])
    assert result.exit_code == 2
    assert "cannot infer format" in result.output


def test_replay_forced_jsonl_on_weird_extension(tmp_path: Path) -> None:
    weird = tmp_path / "capture.weird"
    backend = JSONLBackend(weird)
    backend.send(_event(0))
    backend.shutdown()
    result = runner.invoke(app, ["replay", str(weird), "--jsonl", "--no-color"])
    assert result.exit_code == 0
    assert "cli-test" in result.output


def test_replay_rejects_both_format_flags(jsonl_capture: Path) -> None:
    result = runner.invoke(app, ["replay", str(jsonl_capture), "--db", "--jsonl"])
    assert result.exit_code == 2
    assert "only one of" in result.output


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def test_query_returns_grouped_results(sqlite_capture: Path) -> None:
    result = runner.invoke(
        app,
        ["query", str(sqlite_capture), "SELECT model, COUNT(*) AS n FROM events GROUP BY model"],
    )
    assert result.exit_code == 0
    assert "gpt-4o-mini" in result.output
    assert "(1 rows)" in result.output


def test_query_read_only_rejects_mutation(sqlite_capture: Path) -> None:
    result = runner.invoke(app, ["query", str(sqlite_capture), "DELETE FROM events"])
    assert result.exit_code == 1
    assert "read-only" in result.output.lower()
    # The data must survive the rejected mutation.
    check = runner.invoke(app, ["query", str(sqlite_capture), "SELECT COUNT(*) FROM events"])
    assert "3" in check.output


def test_query_missing_file_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["query", str(tmp_path / "nope.db"), "SELECT 1"])
    assert result.exit_code == 2
    assert "no such file" in result.output


# ---------------------------------------------------------------------------
# vacuum
# ---------------------------------------------------------------------------


def test_vacuum_keep_days_deletes_old_rows(tmp_path: Path) -> None:
    import time

    path = tmp_path / "vac.db"
    backend = SQLiteBackend(path, retention_days=0)
    old = _event(0)
    old["timestamp"] = int((time.time() - 60 * 86400) * 1000)  # 60 days old
    backend.send(old)
    backend.send(_event(1))  # recent (no timestamp → now)
    backend.shutdown()

    result = runner.invoke(app, ["vacuum", str(path), "--keep-days", "30"])
    assert result.exit_code == 0
    assert "deleted 1 events" in result.output

    check = runner.invoke(app, ["query", str(path), "SELECT COUNT(*) FROM events"])
    assert "1" in check.output


def test_vacuum_all_with_yes_wipes_table(sqlite_capture: Path) -> None:
    result = runner.invoke(app, ["vacuum", str(sqlite_capture), "--all", "--yes"])
    assert result.exit_code == 0
    assert "deleted 3 events" in result.output


def test_vacuum_all_aborts_when_confirmation_declined(sqlite_capture: Path) -> None:
    # Feed "n" to the confirmation prompt.
    result = runner.invoke(app, ["vacuum", str(sqlite_capture), "--all"], input="n\n")
    assert result.exit_code == 0
    assert "aborted" in result.output
    check = runner.invoke(app, ["query", str(sqlite_capture), "SELECT COUNT(*) FROM events"])
    assert "3" in check.output


def test_vacuum_requires_exactly_one_mode(sqlite_capture: Path) -> None:
    # Neither --keep-days nor --all.
    result = runner.invoke(app, ["vacuum", str(sqlite_capture)])
    assert result.exit_code == 2
    assert "exactly one" in result.output


def test_vacuum_missing_file_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["vacuum", str(tmp_path / "nope.db"), "--all", "--yes"])
    assert result.exit_code == 2
    assert "no such file" in result.output
