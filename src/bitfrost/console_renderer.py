"""Video-grade single-line console renderer for Bitfrost events.

Outputs one line per :class:`~bitfrost.types.EventPayload` in a
``tail -f``-style flow suitable for live demos and screen recordings.
The line packs the high-signal fields (time, agent, model, tokens,
duration, status, cost) into a fixed column layout so multiple events
read as a coherent stream rather than a wall of text.

::

    13:42:18  bitfrost-smoke    gpt-4o-mini      12 → 1 tok    187ms  ✓ success   $0.0001
    13:42:21  bitfrost-smoke    claude-haiku-4-5 13 → 5 tok    243ms  ✓ success   $0.0001

Color behaviour
---------------
Colour is opt-in via the ``rich`` library when available. When ``rich``
is not installed (no ``[rich]`` extras), the renderer still produces
the same plain-text columns — Bitfrost never silently breaks because a
nice-to-have dep is missing. The same plain path runs automatically
when ``NO_COLOR`` is set in the environment or when stdout isn't a TTY.
"""

from __future__ import annotations

import datetime as _dt
import os
from collections.abc import Mapping
from typing import Any

from bitfrost.pricing import compute_cost

try:
    from rich.console import Console as _RichConsole
    from rich.text import Text as _RichText

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the import-guard test
    _RichConsole = None  # type: ignore[assignment,misc]
    _RichText = None  # type: ignore[assignment,misc]
    _RICH_AVAILABLE = False


# Column widths chosen so a typical event fits in 100 cols (the line
# length most demos film at). Longer names truncate with an ellipsis.
_COL_AGENT = 18
_COL_MODEL = 22


def is_rich_available() -> bool:
    """Public introspection — used by ``ConsoleBackend`` to decide fallback."""

    return _RICH_AVAILABLE


def render_event(
    event: Mapping[str, Any],
    *,
    colorize: bool = True,
    now: _dt.datetime | None = None,
) -> str:
    """Render one event payload to a single tail-f-style line.

    Parameters
    ----------
    event
        An :class:`~bitfrost.types.EventPayload` dict (or anything with
        the same shape — we read defensively).
    colorize
        When True and ``rich`` is installed, the returned string carries
        ANSI escape codes for status colour. When False (or ``NO_COLOR``
        env is set, or rich is missing), the same plain string is
        returned without escape codes.
    now
        Override for the timestamp column. Defaults to the event's own
        timestamp falling back to wall clock. Mostly useful in tests.
    """

    columns = _build_columns(event, now=now)
    plain = (
        f"{columns['time']}  "
        f"{columns['agent']:<{_COL_AGENT}}  "
        f"{columns['model']:<{_COL_MODEL}}  "
        f"{columns['tokens']:>13}  "
        f"{columns['duration']:>7}  "
        f"{columns['marker']} {columns['outcome']:<8}  "
        f"{columns['cost']:>9}"
    )

    if not colorize or not _RICH_AVAILABLE or _no_color_active():
        return plain

    return _colorize(columns, plain)


def _build_columns(
    event: Mapping[str, Any],
    *,
    now: _dt.datetime | None,
) -> dict[str, str]:
    """Extract and format every column from the event payload.

    The helper exists so the layout assembly and the colour pass can
    share the same field extraction logic without duplicating the
    defensive ``.get`` ladder.
    """

    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    metadata = metadata or {}

    tokens = metadata.get("tokens") if isinstance(metadata.get("tokens"), dict) else {}
    tokens = tokens or {}

    time_str = _format_time(event, metadata, now)
    agent = _truncate(str(event.get("agentId") or "—"), _COL_AGENT)
    model = _truncate(str(event.get("model") or "—"), _COL_MODEL)
    input_tokens = _coerce_int(tokens.get("input"))
    output_tokens = _coerce_int(tokens.get("output"))
    tokens_str = _format_tokens(input_tokens, output_tokens)
    duration_str = _format_duration(event.get("durationMs"))
    outcome = str(event.get("outcome") or "pending")
    marker = _outcome_marker(outcome)
    cost_str = _format_cost(
        model=str(event.get("model") or ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=_coerce_int(tokens.get("cache_read")),
        cache_creation=_coerce_int(tokens.get("cache_creation")),
    )

    return {
        "time": time_str,
        "agent": agent,
        "model": model,
        "tokens": tokens_str,
        "duration": duration_str,
        "outcome": outcome,
        "marker": marker,
        "cost": cost_str,
    }


def _colorize(columns: dict[str, str], plain: str) -> str:
    """Re-render the assembled line with rich styling around outcome + cost.

    We build a :class:`rich.text.Text` so the colour stays scoped to the
    semantic spans (status marker, cost) and the rest of the line stays
    in the default terminal palette.
    """

    assert _RichConsole is not None and _RichText is not None  # narrowed by guard

    # cast: ``_NullIO`` satisfies the structural ``IO[str]`` Console needs,
    # but mypy can't see that without a Protocol declaration. The cast is
    # narrow — we only use rich.Console for its export_text(), the file
    # sink never receives meaningful bytes.
    from typing import IO, cast

    console = _RichConsole(
        record=True,
        file=cast("IO[str]", _NullIO()),
        force_terminal=True,
    )
    text = _RichText(plain)
    # Style the outcome marker + word.
    marker = columns["marker"]
    outcome = columns["outcome"]
    style = _outcome_style(outcome)
    if style:
        marker_idx = plain.find(marker)
        if marker_idx >= 0:
            outcome_idx = plain.find(outcome, marker_idx)
            end = outcome_idx + len(outcome) if outcome_idx >= 0 else marker_idx + len(marker)
            text.stylize(style, marker_idx, end)
    # Dim the cost column when it's the "—" placeholder.
    if columns["cost"] == "—":
        cost_idx = plain.rfind("—")
        if cost_idx >= 0:
            text.stylize("dim", cost_idx, cost_idx + 1)
    console.print(text, end="")
    return console.export_text(styles=True).rstrip("\n")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_time(
    event: Mapping[str, Any],
    metadata: Mapping[str, Any],
    now: _dt.datetime | None,
) -> str:
    """Wall-clock HH:MM:SS timestamp for the event."""

    if now is not None:
        return now.strftime("%H:%M:%S")
    raw = event.get("timestamp") or metadata.get("timestamp")
    if isinstance(raw, (int, float)):
        return _dt.datetime.fromtimestamp(raw / 1000).strftime("%H:%M:%S")
    if isinstance(raw, str):
        try:
            cleaned = raw.replace("Z", "+00:00")
            return _dt.datetime.fromisoformat(cleaned).strftime("%H:%M:%S")
        except ValueError:
            pass
    return _dt.datetime.now().strftime("%H:%M:%S")


def _format_tokens(input_tokens: int, output_tokens: int) -> str:
    """Format the token column. Zeros render as the placeholder dash."""

    if input_tokens == 0 and output_tokens == 0:
        return "— tok"
    return f"{input_tokens} → {output_tokens} tok"


def _format_duration(duration_ms: Any) -> str:
    if not isinstance(duration_ms, (int, float)) or duration_ms <= 0:
        return "—"
    duration_int = int(duration_ms)
    if duration_int >= 1000:
        return f"{duration_int / 1000:.1f}s"
    return f"{duration_int}ms"


def _format_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
) -> str:
    cost = compute_cost(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=cache_read,
        cache_creation=cache_creation,
    )
    if cost is None:
        return "—"
    if cost == 0:
        return "$0"
    if cost < 1:
        # 4 decimals enough for tiny LLM calls; quantize to avoid float-drift display.
        return f"${cost:.4f}".rstrip("0").rstrip(".")
    return f"${cost:.2f}"


def _outcome_marker(outcome: str) -> str:
    return {
        "success": "✓",
        "failed": "✗",
        "pending": "…",
    }.get(outcome, "·")


def _outcome_style(outcome: str) -> str:
    return {
        "success": "green",
        "failed": "red bold",
        "pending": "yellow",
    }.get(outcome, "")


def _truncate(s: str, width: int) -> str:
    if len(s) <= width:
        return s
    return s[: max(0, width - 1)] + "…"


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _no_color_active() -> bool:
    """Respect the ``NO_COLOR`` env var convention (https://no-color.org/)."""

    return bool(os.environ.get("NO_COLOR"))


class _NullIO:
    """File-like sink we pass to rich.Console so it doesn't write to stdout itself.

    We just want the *recording* behaviour so we can export the styled
    string and return it to the caller.
    """

    def write(self, _data: str) -> int:
        return 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return True  # force colour codes in the export


__all__ = ["is_rich_available", "render_event"]
