"""Shared brand identity for Bitfrost surfaces (CLI, TUI, web dashboard).

One source of truth for the wordmark, the colour palette, and the
welcome splash so the three surfaces read as the same product. The web
frontend mirrors :data:`PALETTE` in its CSS (CSS can't import Python);
keep them in sync.

The name nods to the Bifrost — the rainbow bridge of Norse myth that
links worlds — which is what an exporter does: bridge your LLM runtime
to wherever you watch it. The splash leads with that bridge: a small
iridescent arch (the only colourful element) beside a metallic wordmark.

The arch + wordmark art below were pre-rendered once (PIL + pyfiglet) and
embedded as plain strings — neither library is a runtime dependency. The
gradients are applied at draw time with pure arithmetic via ``rich``.
"""

from __future__ import annotations

from collections.abc import Sequence

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

# Metallic platinum stops (logo's silver band) — for the wordmark, top→bottom.
_METAL: list[tuple[int, int, int]] = [
    (0xF6, 0xF6, 0xFB),
    (0xC7, 0xC8, 0xD2),
    (0xB9, 0xBA, 0xC6),
    (0x9A, 0x9B, 0xA7),
    (0x8A, 0x8B, 0x97),
]
# Iridescent stops (logo's rainbow inner line) — for the arch, left→right.
_IRID: list[tuple[int, int, int]] = [
    (0x8B, 0x7B, 0xF0),
    (0x46, 0xC5, 0x85),
    (0xEA, 0xBF, 0x4E),
    (0xEF, 0x5A, 0x6A),
]

# The logo arch, rasterised to braille (pre-rendered). The ONLY colourful
# element in the splash — coloured left→right with the iridescent stops.
ARCH_BRAILLE: list[str] = [
    "  ⢀⣠⣤⣶⠶⠾⠿⠿⠷⠶⣶⣦⣄⡀  ",
    " ⣴⡿⠋⠁         ⠙⢿⣦ ",
]

# Wordmark (pyfiglet 'ansi_shadow', pre-rendered) — coloured top→bottom
# with the metallic stops.
WORDMARK: list[str] = [
    "██████╗ ██╗████████╗███████╗██████╗  ██████╗ ███████╗████████╗",
    "██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝",
    "██████╔╝██║   ██║   █████╗  ██████╔╝██║   ██║███████╗   ██║   ",
    "██╔══██╗██║   ██║   ██╔══╝  ██╔══██╗██║   ██║╚════██║   ██║   ",
    "██████╔╝██║   ██║   ██║     ██║  ██║╚██████╔╝███████║   ██║   ",
    "╚═════╝ ╚═╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   ",
]

# Plain-text fallback wordmark for when ``rich`` isn't installed.
BANNER = r"""
 ┌┐ ┬┌┬┐┌─┐┬─┐┌─┐┌─┐┌┬┐
 ├┴┐│ │ ├┤ ├┬┘│ │└─┐ │
 └─┘┴ ┴ └  ┴└─└─┘└─┘ ┴   by voight.xyz
 ═══════════════════════ bridge your LLM telemetry
""".strip("\n")


def banner(tagline: str | None = None) -> str:
    """Return the plain-text wordmark, optionally replacing the tagline."""

    if tagline is None:
        return BANNER
    lines = BANNER.splitlines()
    lines[-1] = "═" * 23 + " " + tagline
    return "\n".join(lines)


def _hex_at(fraction: float, stops: Sequence[tuple[int, int, int]]) -> str:
    """Interpolate a multi-stop gradient and return a ``#rrggbb`` string."""

    f = max(0.0, min(1.0, fraction))
    seg = f * (len(stops) - 1)
    i = int(seg)
    if i >= len(stops) - 1:
        r, g, b = stops[-1]
        return f"#{r:02x}{g:02x}{b:02x}"
    t = seg - i
    a, c = stops[i], stops[i + 1]
    r, g, b = (round(a[k] + (c[k] - a[k]) * t) for k in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def render_splash(*, full: bool = True, console: object | None = None) -> None:
    """Print the Bitfrost welcome splash to the terminal.

    Shown only at startup — bare ``bitfrost`` (full), and ``watch`` /
    ``serve`` (compact: lockup only, ``full=False``). Uses ``rich`` for
    the gradient lockup + info panel; if ``rich`` isn't installed it falls
    back to the plain :func:`banner`.

    The arch is the only multi-coloured element; the wordmark is metallic
    platinum and everything else uses the Voight palette.
    """

    try:
        from rich.box import ROUNDED
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:  # pragma: no cover - exercised only without [rich]
        print(banner())
        return

    con = console if isinstance(console, Console) else Console()

    # Arch — iridescent, the only colourful element.
    arch = Text()
    for line in ARCH_BRAILLE:
        width = max(1, len(line) - 1)
        for i, ch in enumerate(line):
            if ch == " ":
                arch.append(" ")
            else:
                arch.append(ch, style=_hex_at(i / width, _IRID))
        arch.append("\n")

    # Wordmark — metallic platinum, top→bottom.
    wm = Text()
    rows = max(1, len(WORDMARK) - 1)
    for i, line in enumerate(WORDMARK):
        wm.append(line + "\n", style=_hex_at(i / rows, _METAL))

    # Side-by-side lockup when the terminal is wide enough; otherwise stack
    # the arch above the wordmark so neither wraps on an 80-column terminal.
    needed = len(WORDMARK[0]) + len(ARCH_BRAILLE[0]) + 2
    if con.size.width >= needed:
        lockup = Table.grid(padding=(0, 2))
        lockup.add_column()
        lockup.add_column()
        lockup.add_row(Text("\n") + arch, wm)
        con.print(lockup)
    else:
        con.print(arch)
        con.print(wm)

    if not full:
        return

    try:
        from bitfrost import __version__
    except Exception:  # pragma: no cover - defensive
        __version__ = "0.1.0"

    con.print()
    info = Text()
    info.append("Drop-in OpenTelemetry observability for Python LLM apps.\n", style="#b4b4bd")
    info.append(f"v{__version__} · MIT · Python 3.10-3.13\n\n", style="#5c5c66")

    def _section(title: str, items: list[str]) -> None:
        info.append(f"{title}\n", style=f"bold {PALETTE['accent']}")
        info.append("  " + " · ".join(items) + "\n\n", style=PALETTE["fg"])

    _section("Backends", ["console", "sqlite", "jsonl", "otlp", "voight", "tee"])
    _section("Commands", ["watch", "replay", "query", "vacuum", "tui", "serve"])
    _section("Instruments", ["openai", "anthropic", "litellm", "smolagents"])
    info.append("run ", style="#5c5c66")
    info.append("bitfrost --help", style=PALETTE["accent2"])

    con.print(
        Panel(
            info,
            title=f"[bold {PALETTE['fg']}]bitfrost[/]  [{PALETTE['fg_muted']}]by voight.xyz[/]",
            title_align="left",
            border_style=PALETTE["accent"],
            box=ROUNDED,
            padding=(1, 3),
            width=84,
        )
    )


__all__ = ["ARCH_BRAILLE", "BANNER", "PALETTE", "WORDMARK", "banner", "render_splash"]
