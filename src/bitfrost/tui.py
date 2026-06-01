"""Interactive terminal dashboard for captured LLM telemetry.

``bitfrost tui <file>`` opens a full-screen, keyboard-navigable view of a
capture (JSONL or SQLite): a live-updating event table on top, a detail
panel below showing the selected event's prompt / response / tokens /
cost. It's the in-terminal counterpart to ``bitfrost serve`` — no browser
required.

Built on `textual <https://textual.textualize.io/>`_, shipped as the
optional ``[tui]`` extra so the base install stays lean. Import is lazy:
:func:`run_tui` raises a clear message if ``textual`` isn't installed.

Keybindings
-----------
- ``↑`` / ``↓`` — move the selection
- ``m`` — toggle masking of prompt / response content (masked by default)
- ``q`` — quit

The detail panel masks prompt and response text by default (bullet
glyphs) so opening the dashboard on a shared screen never leaks content;
``m`` reveals it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bitfrost._readers import JSONLReader, SQLiteReader, make_reader
from bitfrost.console_renderer import build_columns

# textual is an optional dependency — imported lazily inside run_tui so
# `import bitfrost.tui` doesn't hard-require it. The App subclass is built
# inside a factory for the same reason.
_POLL_INTERVAL_SECONDS = 1.0
_MASK_GLYPH = "•"


def _mask(text: str, *, masked: bool) -> str:
    """Return ``text`` or a bullet-glyph stand-in of the same rough length."""

    if not masked:
        return text
    # Mirror token shape without revealing content; cap so a huge prompt
    # doesn't fill the panel.
    return _MASK_GLYPH * min(len(text), 60) if text else ""


def _event_detail_lines(event: dict[str, Any], *, masked: bool) -> list[str]:
    """Build the detail-panel text for one event."""

    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    metadata = metadata or {}
    tokens = metadata.get("tokens") if isinstance(metadata.get("tokens"), dict) else {}
    tokens = tokens or {}

    lines: list[str] = []
    lines.append(f"agent:     {event.get('agentId', '—')}")
    lines.append(f"model:     {event.get('model', '—')}")
    lines.append(f"provider:  {metadata.get('provider', '—')}")
    lines.append(f"outcome:   {event.get('outcome', '—')}")
    lines.append(f"duration:  {event.get('durationMs', '—')} ms")
    lines.append(
        f"tokens:    in={tokens.get('input', 0)} out={tokens.get('output', 0)}"
        + (
            f" cache_read={tokens['cache_read']}"
            if isinstance(tokens.get("cache_read"), int)
            else ""
        )
    )
    if metadata.get("sessionId"):
        lines.append(f"session:   {metadata['sessionId']}")
    if metadata.get("spanName"):
        lines.append(f"span:      {metadata['spanName']}")

    # Content fields — masked by default.
    input_obj = event.get("input")
    prompt_text = ""
    if isinstance(input_obj, dict):
        messages = input_obj.get("messages")
        if isinstance(messages, list) and messages:
            parts = [str(m.get("content", "")) for m in messages if isinstance(m, dict)]
            prompt_text = " / ".join(p for p in parts if p)
    response_text = str(metadata.get("responseText") or "")

    lines.append("")
    lines.append(f"prompt:    {_mask(prompt_text, masked=masked) or '—'}")
    lines.append(f"response:  {_mask(response_text, masked=masked) or '—'}")
    if masked and (prompt_text or response_text):
        lines.append("")
        lines.append("(press 'm' to reveal content)")
    return lines


def _build_app_class() -> type:
    """Build the textual App subclass. Imported lazily so textual stays optional."""

    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.widgets import DataTable, Footer, Header, Static

    class BitfrostTUI(App[None]):
        """Full-screen interactive view of a Bitfrost capture."""

        TITLE = "bitfrost"
        SUB_TITLE = "by voight.xyz"
        # Brand palette (mirrors bitfrost._brand.PALETTE) — replaces textual's
        # default blue accent with Bitfrost's teal so the TUI, the web
        # dashboard and the CLI read as the same product.
        CSS = """
        Screen { background: #0c0e0d; }
        Header { background: #14171a; color: #5eead4; }
        DataTable { height: 60%; }
        DataTable > .datatable--cursor { background: #5eead4 30%; }
        DataTable > .datatable--header { color: #8b9197; }
        #detail {
            height: 40%;
            border: round #5eead4;
            padding: 0 1;
        }
        """
        # textual manages BINDINGS as a class-level list; RUF012
        # (mutable class attr) is a false positive for the framework idiom.
        BINDINGS = [  # noqa: RUF012
            Binding("q", "quit", "Quit"),
            Binding("m", "toggle_mask", "Toggle content mask"),
        ]

        def __init__(
            self,
            reader: JSONLReader | SQLiteReader,
            *,
            poll_interval: float = _POLL_INTERVAL_SECONDS,
        ) -> None:
            super().__init__()
            self._reader = reader
            self._poll_interval = poll_interval
            self._marker = 0
            self._events: list[dict[str, Any]] = []
            self._masked = True
            # Mirror of the detail panel's current text, exposed for
            # headless tests (textual's Static internals shift between
            # releases; this attribute is a stable contract).
            self._detail_text = "select an event"

        def compose(self) -> ComposeResult:
            yield Header()
            table: DataTable[Any] = DataTable(id="events", cursor_type="row")
            table.add_columns("time", "agent", "model", "tokens", "dur", "", "cost")
            yield table
            yield Vertical(Static("select an event", id="detail-body"), id="detail")
            yield Footer()

        def on_mount(self) -> None:
            # Backfill existing events, then poll for new ones.
            for event in self._reader.read_all():
                self._append_event(event)
            # Prime the tail marker past the backfill so we don't double-add.
            _events, self._marker = self._reader.tail(self._marker)
            self.set_interval(self._poll_interval, self._poll)
            self._refresh_detail()

        def _poll(self) -> None:
            events, self._marker = self._reader.tail(self._marker)
            for event in events:
                self._append_event(event)

        def _append_event(self, event: dict[str, Any]) -> None:
            self._events.append(event)
            cols = build_columns(event)
            table = self.query_one("#events", DataTable)
            table.add_row(
                cols["time"],
                cols["agent"],
                cols["model"],
                cols["tokens"],
                cols["duration"],
                cols["marker"],
                cols["cost"],
            )

        def on_data_table_row_highlighted(self, _event: Any) -> None:
            self._refresh_detail()

        def action_toggle_mask(self) -> None:
            self._masked = not self._masked
            self._refresh_detail()

        def _refresh_detail(self) -> None:
            body = self.query_one("#detail-body", Static)
            table = self.query_one("#events", DataTable)
            idx = table.cursor_row
            if idx is None or idx < 0 or idx >= len(self._events):
                self._detail_text = "select an event"
                body.update(self._detail_text)
                return
            lines = _event_detail_lines(self._events[idx], masked=self._masked)
            self._detail_text = "\n".join(lines)
            body.update(self._detail_text)

    return BitfrostTUI


def run_tui(
    source: Path | str,
    *,
    fmt: str | None = None,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
) -> None:
    """Launch the interactive TUI against a capture file.

    Raises a clear ImportError-derived message if the ``[tui]`` extra
    (textual) isn't installed.
    """

    try:
        import textual  # noqa: F401
    except ImportError as err:  # pragma: no cover - exercised via message test
        msg = (
            "bitfrost tui requires the 'tui' extra. Install it with:\n"
            "    pip install 'bitfrost[tui]'"
        )
        raise ImportError(msg) from err

    reader = make_reader(source, fmt=fmt)
    app_cls = _build_app_class()
    app = app_cls(reader, poll_interval=poll_interval)
    app.run()


__all__ = ["run_tui"]
