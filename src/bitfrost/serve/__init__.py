"""``bitfrost serve`` — local web dashboard backend.

A Starlette app + Server-Sent Events stream over a SQLite capture,
launched with ``bitfrost serve <file.db>``. Shipped as the optional
``[serve]`` extra so the base install stays lean; :func:`run_server`
raises a clear message if the extra isn't installed.

The browser dashboard is the visual counterpart to the in-terminal TUI
(``bitfrost tui``) — same capture, same masked-by-default content.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080


def run_server(
    source: str | Path,
    *,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    poll_interval: float = 1.0,
) -> None:
    """Launch the dashboard server against a SQLite capture.

    Binds to ``127.0.0.1`` by default — the dashboard is local-first and
    is never exposed to the network unless the caller passes an explicit
    ``host``. Raises a clear ImportError-derived message if the
    ``[serve]`` extra (starlette + uvicorn + sse-starlette) isn't present.
    """

    try:
        import sse_starlette  # noqa: F401
        import starlette  # noqa: F401
        import uvicorn
    except ImportError as err:  # pragma: no cover - exercised via message test
        msg = (
            "bitfrost serve requires the 'serve' extra. Install it with:\n"
            "    pip install 'bitfrost[serve]'"
        )
        raise ImportError(msg) from err

    from bitfrost.serve.app import build_app

    app = build_app(source, poll_interval=poll_interval)
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["run_server"]
