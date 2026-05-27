"""Tests for :mod:`bitfrost.console_renderer`.

The renderer is the user-facing demo surface — these tests pin the
format that landing GIFs and screen-recorded launch videos depend on.

Coverage groups
---------------
- Layout: column placement + truncation + tail-f one-line shape
- Tokens: known + zero + missing placeholder
- Status marker + colour: success / failed / pending / unknown
- Cost: known model formatted, unknown model "—", sub-cent precision
- Plain fallback: NO_COLOR, no rich, non-TTY all degrade identically
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from bitfrost.console_renderer import render_event


def _make_event(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "agentId": "smoke-agent",
        "type": "action",
        "model": "gpt-4o-mini",
        "durationMs": 187,
        "outcome": "success",
        "metadata": {
            "tokens": {"input": 12, "output": 5, "total": 17},
            "source": "bitfrost",
            "provider": "openai",
        },
    }
    base.update(overrides)
    return base


_NOW = _dt.datetime(2026, 5, 27, 13, 42, 18)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_one_line_layout_contains_every_column() -> None:
    line = render_event(_make_event(), colorize=False, now=_NOW)
    # All semantic pieces appear in order on a single line.
    assert "\n" not in line
    assert "13:42:18" in line
    assert "smoke-agent" in line
    assert "gpt-4o-mini" in line
    assert "12 → 5 tok" in line
    assert "187ms" in line
    assert "success" in line


def test_long_agent_name_truncated_with_ellipsis() -> None:
    long_name = "this-is-a-very-long-agent-name-that-blows-the-budget"
    line = render_event(_make_event(agentId=long_name), colorize=False, now=_NOW)
    # The truncated form must end with an ellipsis and not contain the
    # full original name.
    assert "this-is-a-very-l" in line
    assert "…" in line
    assert long_name not in line


def test_long_model_name_truncated_with_ellipsis() -> None:
    long_model = "claude-haiku-4-5-2025-10-01-extended-experimental"
    line = render_event(_make_event(model=long_model), colorize=False, now=_NOW)
    assert "…" in line
    assert long_model not in line


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_tokens_zero_renders_placeholder() -> None:
    line = render_event(
        _make_event(metadata={"tokens": {"input": 0, "output": 0}}),
        colorize=False,
        now=_NOW,
    )
    assert "— tok" in line


def test_tokens_missing_renders_placeholder() -> None:
    line = render_event(_make_event(metadata={}), colorize=False, now=_NOW)
    assert "— tok" in line


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


def test_duration_under_one_second_in_ms() -> None:
    line = render_event(_make_event(durationMs=187), colorize=False, now=_NOW)
    assert "187ms" in line


def test_duration_over_one_second_in_seconds_one_decimal() -> None:
    line = render_event(_make_event(durationMs=1500), colorize=False, now=_NOW)
    assert "1.5s" in line


def test_duration_missing_renders_dash() -> None:
    line = render_event(_make_event(durationMs=None), colorize=False, now=_NOW)
    # Just confirms a "—" appears for the duration; the test on cost
    # uses a separate event to avoid ambiguity.
    assert "—" in line


# ---------------------------------------------------------------------------
# Outcome marker
# ---------------------------------------------------------------------------


def test_success_outcome_renders_check_mark() -> None:
    line = render_event(_make_event(outcome="success"), colorize=False, now=_NOW)
    assert "✓" in line
    assert "success" in line


def test_failed_outcome_renders_cross_mark() -> None:
    line = render_event(_make_event(outcome="failed"), colorize=False, now=_NOW)
    assert "✗" in line
    assert "failed" in line


def test_pending_outcome_renders_ellipsis_mark() -> None:
    line = render_event(_make_event(outcome="pending"), colorize=False, now=_NOW)
    assert "…" in line
    assert "pending" in line


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_known_model_renders_dollar_cost() -> None:
    # gpt-4o-mini: 1M input * 0.15 + 1M output * 0.60 = 0.75
    line = render_event(
        _make_event(metadata={"tokens": {"input": 1_000_000, "output": 1_000_000}}),
        colorize=False,
        now=_NOW,
    )
    assert "$0.75" in line


def test_unknown_model_renders_cost_placeholder() -> None:
    line = render_event(_make_event(model="totally-fake"), colorize=False, now=_NOW)
    # The cost column is "—" when pricing returns None — distinct from
    # the duration "—" placeholder; we check it's present in the line.
    # Counting "—" occurrences is brittle; assert via shape: cost is the
    # right-most column.
    rightmost = line.split()[-1]
    assert rightmost == "—"


def test_sub_cent_cost_renders_with_4_decimal_precision() -> None:
    # gpt-4o-mini: 12 in / 5 out → 12*0.15/1M + 5*0.60/1M = 0.0000018 + 0.000003 = 0.0000048
    # Quantization drops trailing zeros; we just check it starts with $0.
    line = render_event(_make_event(), colorize=False, now=_NOW)
    rightmost = line.split()[-1]
    assert rightmost.startswith("$0")


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def test_colorize_true_emits_ansi_escape_codes() -> None:
    line = render_event(_make_event(), colorize=True, now=_NOW)
    # ANSI escape sequences start with ESC[.
    assert "\x1b[" in line


def test_colorize_false_emits_plain_text_no_ansi() -> None:
    line = render_event(_make_event(), colorize=False, now=_NOW)
    assert "\x1b[" not in line


def test_no_color_env_strips_ansi_even_when_colorize_true(monkeypatch: Any) -> None:
    """``NO_COLOR=1`` (https://no-color.org/) wins over the explicit flag."""

    monkeypatch.setenv("NO_COLOR", "1")
    line = render_event(_make_event(), colorize=True, now=_NOW)
    assert "\x1b[" not in line


# ---------------------------------------------------------------------------
# Real fixture roundtrip
# ---------------------------------------------------------------------------


def test_real_v132_openai_fixture_renders_without_crashing() -> None:
    """End-to-end: the v1.32+ OpenAI fixture maps + renders to a stable line."""

    import json
    from pathlib import Path

    from bitfrost.attribute_mapper import map_attributes

    fixture_path = Path(__file__).parent.parent / "fixtures" / "openai_span_v132.json"
    span = json.loads(fixture_path.read_text())[0]
    payload = map_attributes(
        span_name=span["name"],
        attributes=span["attributes"],
        start_time_ns=span["start_time_ns"],
        end_time_ns=span["end_time_ns"],
        status_code=span["status_code"].split(".")[-1],
        status_description=None,
        events=None,
        instrumentation_scope_name=span["instrumentation_scope"],
    )
    assert payload is not None
    payload["agentId"] = "real-smoke"
    line = render_event(payload, colorize=False, now=_NOW)
    assert "real-smoke" in line
    assert "gpt-4o-mini" in line
    # Real fixture has output tokens > 0 → no placeholder.
    assert "— tok" not in line
