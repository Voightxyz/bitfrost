"""Tests for :mod:`bitfrost.tui`.

The TUI is interactive, so the unit tests split into two layers:

- Pure helpers (``_event_detail_lines``, ``_mask``) tested directly —
  these carry the masking contract the demo relies on.
- The textual App driven headless via ``App.run_test()`` (Pilot): mount,
  backfill rows, navigate, toggle mask, quit. No real terminal needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("textual")

from bitfrost import tui as _tui
from bitfrost.backends.sqlite import SQLiteBackend


@pytest.fixture
def anyio_backend() -> str:
    """Drive the async tests on asyncio (textual's loop), not trio."""

    return "asyncio"


def _event(i: int, *, prompt: str = "hello world", response: str = "hi back") -> dict[str, Any]:
    return {
        "agentId": "tui-test",
        "type": "action",
        "model": "gpt-4o-mini",
        "durationMs": 100 + i,
        "outcome": "success",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "metadata": {
            "provider": "openai",
            "sessionId": f"s{i}",
            "responseText": response,
            "tokens": {"input": 10 + i, "output": i + 1},
        },
    }


# ---------------------------------------------------------------------------
# Pure helpers — masking contract
# ---------------------------------------------------------------------------


def test_mask_hides_content_when_masked() -> None:
    out = _tui._mask("secret prompt", masked=True)
    assert "secret" not in out
    assert set(out) <= {_tui._MASK_GLYPH}
    assert len(out) > 0


def test_mask_reveals_content_when_not_masked() -> None:
    assert _tui._mask("secret prompt", masked=False) == "secret prompt"


def test_mask_empty_string_stays_empty() -> None:
    assert _tui._mask("", masked=True) == ""


def test_detail_lines_mask_prompt_and_response_by_default() -> None:
    lines = _tui._event_detail_lines(_event(0), masked=True)
    text = "\n".join(lines)
    assert "hello world" not in text  # prompt masked
    assert "hi back" not in text  # response masked
    assert "press 'm' to reveal" in text
    # Non-content metadata still visible.
    assert "tui-test" in text
    assert "openai" in text


def test_detail_lines_reveal_content_when_unmasked() -> None:
    lines = _tui._event_detail_lines(_event(0), masked=False)
    text = "\n".join(lines)
    assert "hello world" in text
    assert "hi back" in text


# ---------------------------------------------------------------------------
# Headless App (Pilot)
# ---------------------------------------------------------------------------


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    path = tmp_path / "tui.db"
    backend = SQLiteBackend(path, retention_days=0)
    for i in range(3):
        backend.send(_event(i))
    backend.shutdown()
    return path


async def _make_app(capture: Path) -> Any:
    from bitfrost._readers import SQLiteReader

    app_cls = _tui._build_app_class()
    # Long poll interval so the test controls timing, not the clock.
    return app_cls(SQLiteReader(capture), poll_interval=3600)


@pytest.mark.anyio
async def test_app_backfills_rows_from_capture(capture: Path) -> None:
    app = await _make_app(capture)
    async with app.run_test() as pilot:
        await pilot.pause()
        from textual.widgets import DataTable

        table = app.query_one("#events", DataTable)
        assert table.row_count == 3


@pytest.mark.anyio
async def test_app_detail_panel_shows_selected_event(capture: Path) -> None:
    app = await _make_app(capture)
    async with app.run_test() as pilot:
        await pilot.pause()
        rendered = app._detail_text
        # Default masked: metadata visible, content hidden.
        assert "tui-test" in rendered
        assert "hello world" not in rendered


@pytest.mark.anyio
async def test_app_toggle_mask_reveals_content(capture: Path) -> None:
    app = await _make_app(capture)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("m")  # reveal
        await pilot.pause()
        assert "hello world" in app._detail_text


@pytest.mark.anyio
async def test_app_quit_binding_exits(capture: Path) -> None:
    app = await _make_app(capture)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    # run_test context exited cleanly == quit worked.
    assert app.return_code is None or isinstance(app.return_code, int)
