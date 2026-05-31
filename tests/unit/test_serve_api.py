"""Tests for :mod:`bitfrost.serve.api` — the dashboard JSON + SSE API.

Most endpoints are exercised through Starlette's ``TestClient`` against a
real SQLite capture seeded via :class:`SQLiteBackend` (real write path, no
hand-rolled rows). The SSE stream is tested at the async-generator level —
deterministic, no hanging HTTP connection.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")
pytest.importorskip("sse_starlette")

from starlette.testclient import TestClient

from bitfrost.backends.sqlite import SQLiteBackend
from bitfrost.serve.api import SpanStore, live_span_stream, sse_endpoint
from bitfrost.serve.app import build_app


def _event(
    i: int,
    *,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    outcome: str = "success",
    agent: str = "serve-test",
    prompt: str = "hello world",
    response: str = "hi back",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> dict[str, Any]:
    return {
        "agentId": agent,
        "type": "action",
        "model": model,
        "durationMs": 100 + i,
        "outcome": outcome,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "metadata": {
            "provider": provider,
            "sessionId": f"s{i}",
            "responseText": response,
            "tokens": {"input": input_tokens, "output": output_tokens},
        },
    }


def _seed(path: Path, events: list[dict[str, Any]]) -> None:
    backend = SQLiteBackend(path, retention_days=0)
    for event in events:
        backend.send(event)
    backend.shutdown()


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    path = tmp_path / "serve.db"
    _seed(
        path,
        [
            _event(0, model="gpt-4o-mini", provider="openai", outcome="success"),
            _event(1, model="gpt-4o-mini", provider="openai", outcome="success"),
            _event(2, model="claude-haiku-4-5", provider="anthropic", outcome="failed"),
            _event(3, model="mystery-model-9", provider="openai", outcome="success"),
        ],
    )
    return path


@pytest.fixture
def client(populated_db: Path) -> TestClient:
    return TestClient(build_app(populated_db, poll_interval=0.01))


# ---------------------------------------------------------------------------
# /api/spans
# ---------------------------------------------------------------------------


def test_spans_lists_all_seeded(client: TestClient) -> None:
    resp = client.get("/api/spans")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 4
    assert len(body["spans"]) == 4
    span = body["spans"][0]
    # Shape contract the frontend relies on.
    for key in ("id", "agentId", "model", "provider", "outcome", "tokens", "cost"):
        assert key in span


def test_spans_computes_cost_for_known_model(client: TestClient) -> None:
    resp = client.get("/api/spans?model=gpt-4o-mini")
    spans = resp.json()["spans"]
    assert spans
    assert all(s["cost"] is not None and s["cost"] > 0 for s in spans)


def test_spans_cost_null_for_unknown_model(client: TestClient) -> None:
    resp = client.get("/api/spans?model=mystery-model-9")
    spans = resp.json()["spans"]
    assert len(spans) == 1
    assert spans[0]["cost"] is None


def test_spans_pagination(client: TestClient) -> None:
    page1 = client.get("/api/spans?limit=2&offset=0").json()
    page2 = client.get("/api/spans?limit=2&offset=2").json()
    assert page1["total"] == 4
    assert len(page1["spans"]) == 2
    assert len(page2["spans"]) == 2
    ids1 = {s["id"] for s in page1["spans"]}
    ids2 = {s["id"] for s in page2["spans"]}
    assert ids1.isdisjoint(ids2)


def test_spans_filter_by_provider(client: TestClient) -> None:
    resp = client.get("/api/spans?provider=anthropic")
    spans = resp.json()["spans"]
    assert len(spans) == 1
    assert spans[0]["provider"] == "anthropic"


def test_spans_filter_by_outcome(client: TestClient) -> None:
    resp = client.get("/api/spans?outcome=failed")
    spans = resp.json()["spans"]
    assert len(spans) == 1
    assert spans[0]["outcome"] == "failed"


def test_spans_invalid_limit_returns_400(client: TestClient) -> None:
    resp = client.get("/api/spans?limit=not-a-number")
    assert resp.status_code == 400
    assert "limit" in resp.json()["error"]


# ---------------------------------------------------------------------------
# /api/spans/{id}
# ---------------------------------------------------------------------------


def test_span_detail_includes_content(client: TestClient) -> None:
    list_resp = client.get("/api/spans?provider=anthropic").json()
    span_id = list_resp["spans"][0]["id"]
    resp = client.get(f"/api/spans/{span_id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == span_id
    # Detail-only fields.
    assert detail["input"]["messages"][0]["content"] == "hello world"
    assert detail["responseText"] == "hi back"
    assert "metadata" in detail


def test_span_detail_404_for_unknown_id(client: TestClient) -> None:
    resp = client.get("/api/spans/does-not-exist")
    assert resp.status_code == 404
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------


def test_stats_totals(client: TestClient) -> None:
    stats = client.get("/api/stats").json()
    assert stats["totalCalls"] == 4
    assert stats["outcomes"] == {"success": 3, "failed": 1}
    assert stats["totalTokens"]["input"] == 40  # 4 events x 10
    assert stats["totalTokens"]["output"] == 20  # 4 events x 5
    # Only the 3 priced models contribute; mystery-model-9 is null.
    assert stats["totalCost"] > 0


def test_stats_per_model_and_provider(client: TestClient) -> None:
    stats = client.get("/api/stats").json()
    models = {m["model"]: m for m in stats["models"]}
    assert models["gpt-4o-mini"]["calls"] == 2
    assert models["mystery-model-9"]["cost"] is None
    providers = {p["provider"]: p["calls"] for p in stats["providers"]}
    assert providers["openai"] == 3
    assert providers["anthropic"] == 1


def test_stats_series_and_latency(client: TestClient) -> None:
    stats = client.get("/api/stats").json()
    # Cost-over-time series: one bucket per day, calls summed across models.
    assert isinstance(stats["series"], list)
    assert sum(b["calls"] for b in stats["series"]) == 4
    assert all({"day", "calls", "cost"} <= set(b) for b in stats["series"])
    # Latency percentiles present and ordered.
    assert "p50" in stats["latency"] and "p95" in stats["latency"]
    assert stats["latency"]["p95"] >= stats["latency"]["p50"] > 0


def test_stats_empty_db_series_and_latency(tmp_path: Path) -> None:
    path = tmp_path / "empty2.db"
    _seed(path, [])
    stats = TestClient(build_app(path)).get("/api/stats").json()
    assert stats["series"] == []
    assert stats["latency"] == {"p50": 0, "p95": 0}


def test_stats_empty_db_returns_zeros(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    _seed(path, [])
    with TestClient(build_app(path)) as c:
        stats = c.get("/api/stats").json()
    assert stats["totalCalls"] == 0
    assert stats["totalCost"] == 0
    assert stats["models"] == []
    spans = TestClient(build_app(path)).get("/api/spans").json()
    assert spans["spans"] == []
    assert spans["total"] == 0


# ---------------------------------------------------------------------------
# Read-only invariant
# ---------------------------------------------------------------------------


def test_server_never_mutates_capture(populated_db: Path, client: TestClient) -> None:
    store = SpanStore(populated_db)
    before = store.list_spans(limit=1000, offset=0)[1]
    # Exercise every read endpoint.
    client.get("/api/spans")
    client.get("/api/stats")
    client.get(f"/api/spans/{client.get('/api/spans').json()['spans'][0]['id']}")
    after = store.list_spans(limit=1000, offset=0)[1]
    assert before == after == 4


# ---------------------------------------------------------------------------
# SSE — async generator level (deterministic, no hanging connection)
# ---------------------------------------------------------------------------


def test_sse_endpoint_is_event_stream(populated_db: Path) -> None:
    app = build_app(populated_db, poll_interval=0.01)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/sse/live",
        "headers": [],
        "query_string": b"",
        "app": app,
    }
    from starlette.requests import Request

    resp = asyncio.run(sse_endpoint(Request(scope)))
    assert resp.media_type == "text/event-stream"


def test_sse_stream_emits_frame_for_new_span(tmp_path: Path) -> None:
    path = tmp_path / "live.db"
    _seed(path, [_event(0, prompt="pre-existing")])
    store = SpanStore(path)
    marker = store.max_rowid()  # cursor past the pre-existing row

    # A span captured AFTER the client connected.
    _seed(path, [_event(1, agent="late-agent", prompt="arrived later")])

    async def first_frame() -> dict[str, str]:
        agen = live_span_stream(store, marker, poll_interval=0.01)
        try:
            return await asyncio.wait_for(agen.__anext__(), timeout=3.0)
        finally:
            await agen.aclose()

    frame = asyncio.run(first_frame())
    payload = json.loads(frame["data"])
    assert payload["agentId"] == "late-agent"
