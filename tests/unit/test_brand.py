"""Tests for :mod:`bitfrost._brand` — the shared brand identity + splash.

The splash is visual, so these cover the contract rather than pixels:
the gradient math, the plain-text fallback, and that the rich render
emits the expected sections (full vs compact) without raising.
"""

from __future__ import annotations

import pytest

from bitfrost._brand import BANNER, PALETTE, _hex_at, banner, render_splash


def test_banner_default_and_custom_tagline() -> None:
    assert "by voight.xyz" in BANNER
    assert "my tagline" in banner("my tagline")


def test_palette_has_core_tokens() -> None:
    for key in ("bg", "fg", "accent", "accent2"):
        assert PALETTE[key].startswith("#") and len(PALETTE[key]) == 7


def test_hex_at_endpoints_and_midpoint() -> None:
    stops = [(0, 0, 0), (255, 255, 255)]
    assert _hex_at(0.0, stops) == "#000000"
    assert _hex_at(1.0, stops) == "#ffffff"
    mid = _hex_at(0.5, stops)
    # Midpoint is a grey roughly halfway; not the endpoints.
    assert mid not in ("#000000", "#ffffff")
    # Clamps out-of-range inputs.
    assert _hex_at(-1.0, stops) == "#000000"
    assert _hex_at(2.0, stops) == "#ffffff"


def _render(*, full: bool, width: int = 100) -> str:
    rich = pytest.importorskip("rich")
    console = rich.console.Console(record=True, width=width)
    render_splash(full=full, console=console)
    return console.export_text()


def test_splash_full_has_all_sections() -> None:
    out = _render(full=True)
    assert "bitfrost" in out  # panel title
    assert "by voight.xyz" in out
    for section in ("Backends", "Commands", "Instruments"):
        assert section in out
    assert "bitfrost --help" in out


def test_splash_compact_omits_info_panel() -> None:
    out = _render(full=False)
    # Compact = lockup only; no Backends/Commands listing.
    assert "Backends" not in out
    assert "Commands" not in out


def test_splash_narrow_terminal_does_not_crash() -> None:
    # On a narrow terminal the arch stacks above the wordmark; just assert
    # it renders without raising and still produces the info sections.
    out = _render(full=True, width=70)
    assert "Backends" in out
