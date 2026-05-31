"""Shared brand identity for Bitfrost surfaces (CLI, TUI, web dashboard).

One source of truth for the ASCII wordmark and the colour palette so the
three surfaces read as the same product. The web frontend mirrors
:data:`PALETTE` in its CSS (CSS can't import Python); keep them in sync.

The name nods to the Bifrost — the rainbow bridge of Norse myth that
links worlds — which is what an exporter does: bridge your LLM runtime
to wherever you watch it.
"""

from __future__ import annotations

# Palette — dark, platinum-forward, with a teal→violet "bridge" accent.
# Mirrored in src/bitfrost/serve/static/styles.css (:root variables).
PALETTE = {
    "bg": "#0c0e0d",  # near-black canvas
    "surface": "#14171a",  # card
    "surface2": "#1a1e22",  # raised card / hover
    "border": "#23282d",
    "fg": "#e6e7ea",  # platinum text
    "fg_muted": "#8b9197",
    "accent": "#5eead4",  # teal — primary accent
    "accent2": "#a78bfa",  # violet — bridge gradient end
    "success": "#4ade80",
    "failed": "#f87171",
    "pending": "#fbbf24",
}

# ASCII wordmark. Box-drawing glyphs render crisply in any monospace
# terminal; the trailing line is a nod to the rainbow bridge.
BANNER = r"""
 ┌┐ ┬┌┬┐┌─┐┬─┐┌─┐┌─┐┌┬┐
 ├┴┐│ │ ├┤ ├┬┘│ │└─┐ │
 └─┘┴ ┴ └  ┴└─└─┘└─┘ ┴   by voight.xyz
 ═══════════════════════ bridge your LLM telemetry
""".strip("\n")


def banner(tagline: str | None = None) -> str:
    """Return the ASCII wordmark, optionally replacing the default tagline."""

    if tagline is None:
        return BANNER
    lines = BANNER.splitlines()
    lines[-1] = "═" * 23 + " " + tagline
    return "\n".join(lines)


__all__ = ["BANNER", "PALETTE", "banner"]
