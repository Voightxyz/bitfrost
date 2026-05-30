"""End-to-end check for ``bitfrost serve`` (B.8.8 fetch-back).

Drives the full path a real dashboard exercises: capture real-shaped
events to a SQLite DB via :class:`SQLiteBackend`, build the Starlette app
over it, and fetch each JSON endpoint back through Starlette's
``TestClient`` (an actual ASGI round-trip, not an in-process call).

The SSE stream is **not** exercised here: an infinite ``text/event-stream``
read over ``TestClient``'s single-thread portal hangs, and would add
flakiness across CI's 10 cells for no extra coverage. Its data path is
proven deterministically in ``test_serve_api.py`` (the async generator
emits a real frame for a new row; the endpoint returns
``text/event-stream``), and the HTTP wire format is confirmed by the
manual ``bitfrost serve`` + ``curl -N /api/sse/live`` smoke.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")
pytest.importorskip("sse_starlette")

from starlette.testclient import TestClient

from bitfrost.backends.sqlite import SQLiteBackend
from bitfrost.serve.app import build_app


def _event(i: int, *, model: str, provider: str) -> dict[str, Any]:
    return {
        "agentId": "e2e-serve",
        "type": "action",
        "model": model,
        "durationMs": 120 + i,
        "outcome": "success",
        "input": {"messages": [{"role": "user", "content": f"prompt {i}"}]},
        "metadata": {
            "provider": provider,
            "sessionId": "e2e-session",
            "responseText": f"response {i}",
            "tokens": {"input": 20, "output": 8},
        },
    }


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    path = tmp_path / "e2e.db"
    backend = SQLiteBackend(path, retention_days=0)
    backend.send(_event(0, model="gpt-4o-mini", provider="openai"))
    backend.send(_event(1, model="claude-haiku-4-5", provider="anthropic"))
    backend.shutdown()
    return path


def test_e2e_all_endpoints_round_trip(capture: Path) -> None:
    with TestClient(build_app(capture, poll_interval=0.01)) as client:
        # /api/spans — list
        spans_body = client.get("/api/spans").json()
        assert spans_body["total"] == 2
        first_id = spans_body["spans"][0]["id"]

        # /api/spans/{id} — detail with content fetched back
        detail = client.get(f"/api/spans/{first_id}").json()
        assert detail["id"] == first_id
        assert detail["input"]["messages"][0]["content"].startswith("prompt")
        assert detail["responseText"].startswith("response")

        # /api/spans/{id} — 404
        assert client.get("/api/spans/nope").status_code == 404

        # /api/stats — rollups cuadran con la captura
        stats = client.get("/api/stats").json()
        assert stats["totalCalls"] == 2
        assert stats["totalCost"] > 0
        providers = {p["provider"] for p in stats["providers"]}
        assert providers == {"openai", "anthropic"}
