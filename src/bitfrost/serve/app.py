"""Starlette application factory for the ``bitfrost serve`` dashboard.

:func:`build_app` wires the JSON + SSE API (see :mod:`bitfrost.serve.api`)
over a SQLite capture and, when present, mounts the static frontend
bundle at ``/``. The static directory is populated in Task #18 — until
then the API is fully usable on its own (``/api/*``), and the mount is
simply skipped if ``static/`` doesn't exist yet.
"""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from bitfrost.serve.api import (
    SpanStore,
    span_detail_endpoint,
    spans_endpoint,
    sse_endpoint,
    stats_endpoint,
)


def build_app(db_path: str | Path, *, poll_interval: float = 1.0) -> Starlette:
    """Build the Starlette app serving ``db_path``.

    Parameters
    ----------
    db_path
        Path to the SQLite capture to serve (read-only).
    poll_interval
        Seconds between polls for the ``/api/sse/live`` stream. Tests
        lower this so they don't wait a real second for a new event.
    """

    routes: list[Route | Mount] = [
        Route("/api/spans", spans_endpoint),
        Route("/api/spans/{span_id}", span_detail_endpoint),
        Route("/api/stats", stats_endpoint),
        Route("/api/sse/live", sse_endpoint),
    ]

    # Mount the frontend bundle when it exists (shipped in Task #18). The
    # API stays fully functional without it, so #17 ships standalone.
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        routes.append(Mount("/", app=StaticFiles(directory=str(static_dir), html=True)))

    app = Starlette(routes=routes)
    app.state.store = SpanStore(db_path)
    app.state.poll_interval = poll_interval
    return app


__all__ = ["build_app"]
