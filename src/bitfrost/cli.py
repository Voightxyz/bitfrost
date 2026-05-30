"""Bitfrost CLI — inspect, replay, and query captured LLM telemetry.

Four commands close the local loop: capture events to a JSONL file or
SQLite DB (via :class:`~bitfrost.backends.jsonl.JSONLBackend` /
:class:`~bitfrost.backends.sqlite.SQLiteBackend`), then:

- ``bitfrost watch <file>``  — live tail, one styled line per new event
- ``bitfrost replay <file>`` — re-render a captured run start to finish
- ``bitfrost query <db> <sql>`` — read-only SQL over a SQLite capture
- ``bitfrost vacuum <db>``   — prune old rows from a SQLite capture

Both ``watch`` and ``replay`` accept either format and auto-detect by
extension (``.jsonl`` → JSON Lines, ``.db`` / ``.sqlite`` → SQLite); the
``--db`` / ``--jsonl`` flags force a format for unconventional names.
``query`` and ``vacuum`` are SQLite-only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from bitfrost._readers import JSONLReader, SQLiteReader, make_reader
from bitfrost.console_renderer import render_event

app = typer.Typer(
    name="bitfrost",
    help="Inspect, replay, and query captured LLM telemetry.",
    no_args_is_help=True,
    add_completion=False,
)


def _resolve_reader(
    source: Path,
    db: bool,
    jsonl: bool,
) -> JSONLReader | SQLiteReader:
    """Pick a reader honouring explicit --db/--jsonl, else auto-detect."""

    if db and jsonl:
        typer.secho(
            "error: pass only one of --db / --jsonl, not both.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    fmt: str | None = "db" if db else "jsonl" if jsonl else None
    try:
        return make_reader(source, fmt=fmt)
    except ValueError as err:
        typer.secho(f"error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from err


@app.command()
def watch(
    source: Path = typer.Argument(..., help="Capture file to tail (.jsonl or .db)."),
    db: bool = typer.Option(False, "--db", help="Force SQLite format."),
    jsonl: bool = typer.Option(False, "--jsonl", help="Force JSON Lines format."),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Seconds between polls.", min=0.1),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colour."),
) -> None:
    """Live-tail a capture file, printing one line per new event.

    Polls the file every ``--interval`` seconds and renders any events
    that appeared since the last poll — the terminal equivalent of
    ``tail -f`` for LLM telemetry. Press Ctrl-C to stop.
    """

    reader = _resolve_reader(source, db, jsonl)
    colorize = not no_color
    typer.secho(f"watching {source} (every {interval}s) — Ctrl-C to stop", fg=typer.colors.CYAN)
    marker = 0
    # Prime the marker to "now" so watch shows only NEW events, not the
    # entire backlog — matching `tail -f` semantics.
    _events, marker = reader.tail(marker)
    try:
        while True:
            events, marker = reader.tail(marker)
            for event in events:
                sys.stdout.write(render_event(event, colorize=colorize) + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.secho("\nstopped.", fg=typer.colors.CYAN)


@app.command()
def replay(
    source: Path = typer.Argument(..., help="Capture file to replay (.jsonl or .db)."),
    db: bool = typer.Option(False, "--db", help="Force SQLite format."),
    jsonl: bool = typer.Option(False, "--jsonl", help="Force JSON Lines format."),
    follow_timing: bool = typer.Option(
        False,
        "--follow-timing",
        help="Pace the replay using the original inter-event gaps.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colour."),
) -> None:
    """Re-render every event in a capture file, start to finish.

    By default events stream out instantly (fast inspection). With
    ``--follow-timing`` the replay honours the wall-clock gaps between
    the original events — useful for recording a realistic demo.
    """

    reader = _resolve_reader(source, db, jsonl)
    events = reader.read_all()
    colorize = not no_color
    if not events:
        typer.secho(f"no events in {source}", fg=typer.colors.YELLOW)
        return

    prev_ts: int | None = None
    for event in events:
        if follow_timing:
            ts = _event_timestamp_ms(event)
            if prev_ts is not None and ts is not None:
                gap = (ts - prev_ts) / 1000.0
                # Clamp so a multi-minute idle gap doesn't stall a demo,
                # and a clock skew (negative gap) never sleeps.
                time.sleep(max(0.0, min(gap, 5.0)))
            if ts is not None:
                prev_ts = ts
        sys.stdout.write(render_event(event, colorize=colorize) + "\n")
        sys.stdout.flush()
    typer.secho(f"\nreplayed {len(events)} events", fg=typer.colors.CYAN)


@app.command()
def tui(
    source: Path = typer.Argument(..., help="Capture file to explore (.jsonl or .db)."),
    db: bool = typer.Option(False, "--db", help="Force SQLite format."),
    jsonl: bool = typer.Option(False, "--jsonl", help="Force JSON Lines format."),
) -> None:
    """Open the full-screen interactive dashboard for a capture.

    A keyboard-navigable event table with a live-updating detail panel —
    the in-terminal counterpart to ``bitfrost serve``. Requires the
    ``[tui]`` extra (``pip install 'bitfrost[tui]'``).
    """

    if db and jsonl:
        typer.secho(
            "error: pass only one of --db / --jsonl, not both.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    if not source.exists():
        typer.secho(f"error: no such file: {source}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    fmt: str | None = "db" if db else "jsonl" if jsonl else None
    from bitfrost.tui import run_tui

    try:
        run_tui(source, fmt=fmt)
    except ImportError as err:
        typer.secho(f"error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from err
    except ValueError as err:
        typer.secho(f"error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from err


@app.command()
def query(
    db_path: Path = typer.Argument(..., help="SQLite capture DB."),
    sql: str = typer.Argument(..., help="Read-only SQL to run."),
) -> None:
    """Run a read-only SQL query over a SQLite capture and print a table.

    The connection is opened read-only, so any mutating statement
    (INSERT / UPDATE / DELETE / DROP) is rejected — use ``bitfrost
    vacuum`` for the one supported mutation (pruning old rows).
    """

    import sqlite3

    if not db_path.exists():
        typer.secho(f"error: no such file: {db_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    reader = SQLiteReader(db_path)
    try:
        columns, rows = reader.query(sql)
    except sqlite3.OperationalError as err:
        msg = str(err)
        if "readonly" in msg.lower() or "read-only" in msg.lower():
            typer.secho(
                "error: query is read-only — mutations aren't allowed. "
                "Use `bitfrost vacuum` to prune rows.",
                fg=typer.colors.RED,
                err=True,
            )
        else:
            typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from err

    _print_table(columns, rows)


@app.command()
def vacuum(
    db_path: Path = typer.Argument(..., help="SQLite capture DB to prune."),
    keep_days: int | None = typer.Option(
        None, "--keep-days", help="Delete events older than N days."
    ),
    all_rows: bool = typer.Option(
        False, "--all", help="Delete ALL events (requires confirmation)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Prune old events from a SQLite capture, then reclaim disk with VACUUM.

    Pass ``--keep-days N`` to delete events older than N days, or
    ``--all`` to wipe the table (prompts for confirmation unless
    ``--yes``). Exactly one of the two must be given.
    """

    import sqlite3

    if not db_path.exists():
        typer.secho(f"error: no such file: {db_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    if (keep_days is None) == (not all_rows):
        typer.secho(
            "error: pass exactly one of --keep-days N or --all.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)

    if all_rows and not yes:
        confirmed = typer.confirm(f"Delete ALL events from {db_path}?")
        if not confirmed:
            typer.secho("aborted.", fg=typer.colors.YELLOW)
            raise typer.Exit(0)

    conn = sqlite3.connect(str(db_path))
    try:
        if all_rows:
            cursor = conn.execute("DELETE FROM events")
        else:
            cutoff_ms = int((time.time() - keep_days * 86400) * 1000)  # type: ignore[operator]
            cursor = conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_ms,))
        deleted = cursor.rowcount
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    typer.secho(f"deleted {deleted} events from {db_path}", fg=typer.colors.CYAN)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_timestamp_ms(event: dict[str, object]) -> int | None:
    """Best-effort millisecond timestamp from an event for --follow-timing."""

    raw = event.get("timestamp")
    if isinstance(raw, (int, float)):
        return int(raw)
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        meta_ts = metadata.get("timestamp")
        if isinstance(meta_ts, (int, float)):
            return int(meta_ts)
    return None


def _print_table(columns: list[str], rows: list[tuple[object, ...]]) -> None:
    """Render query results as a rich Table, or plain text if rich is absent."""

    if not columns:
        typer.secho("(no columns)", fg=typer.colors.YELLOW)
        return
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        for col in columns:
            table.add_column(str(col))
        for row in rows:
            table.add_row(*[str(v) for v in row])
        Console().print(table)
    except ImportError:
        # Plain fallback when rich isn't installed.
        sys.stdout.write(" | ".join(columns) + "\n")
        for row in rows:
            sys.stdout.write(" | ".join(str(v) for v in row) + "\n")
    typer.secho(f"({len(rows)} rows)", fg=typer.colors.CYAN)


if __name__ == "__main__":  # pragma: no cover
    app()
