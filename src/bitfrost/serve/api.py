"""HTTP API + SSE for the ``bitfrost serve`` local dashboard.

Read-only access to a SQLite capture written by
:class:`~bitfrost.backends.sqlite.SQLiteBackend`, exposed as JSON so the
embedded frontend (Task #18) can render charts, filters and a per-span
detail panel without any build step.

Every connection is opened with SQLite's ``mode=ro`` URI, so a dashboard
left open in a browser tab can never mutate — let alone corrupt — the
user's capture. The store re-opens a short-lived connection per request
(WAL mode means reads never block the exporter's writer thread).

Endpoints (all under ``/api``)
------------------------------
- ``GET /api/spans``        — paginated, filterable list of spans
- ``GET /api/spans/{id}``   — one span with prompt / response / metadata
- ``GET /api/stats``        — rollups: calls, tokens, cost, per-model, per-provider
- ``GET /api/sse/live``     — Server-Sent Events stream of newly captured spans

Cost is computed locally via :func:`bitfrost.pricing.compute_cost`; an
unknown model yields ``null`` (the frontend renders ``—``) rather than a
fabricated price.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from starlette.responses import JSONResponse

from bitfrost.pricing import compute_cost

_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000

# Column projection shared by every read. ``rowid`` is aliased so the SSE
# tail can derive its cursor; ``id`` is the stable PRIMARY KEY the detail
# endpoint looks up by.
_SELECT = (
    "rowid AS _rowid, id, agent_id, session_id, timestamp, event_type, "
    "model, duration_ms, outcome, input_tokens, output_tokens, "
    "tool_executed, input, metadata"
)


class SpanStore:
    """Read-only SQLite access tailored to the dashboard's query shapes."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh read-only connection (``mode=ro``) with row access."""

        conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def list_spans(
        self,
        *,
        limit: int,
        offset: int,
        agent: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        outcome: str | None = None,
        since_ms: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return ``(spans, total)`` — a page of spans plus the unpaged count."""

        where, params = _build_where(agent, model, provider, outcome, since_ms)
        conn = self._connect()
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT {_SELECT} FROM events{where} "
                "ORDER BY timestamp DESC, rowid DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        finally:
            conn.close()
        return [_serialize(row) for row in rows], int(total)

    def get_span(self, span_id: str) -> dict[str, Any] | None:
        """Return one span by its ``id`` with full detail, or ``None``."""

        conn = self._connect()
        try:
            row = conn.execute(f"SELECT {_SELECT} FROM events WHERE id = ?", (span_id,)).fetchone()
        finally:
            conn.close()
        return _serialize(row, detail=True) if row is not None else None

    def stats(self, *, since_ms: int | None = None) -> dict[str, Any]:
        """Aggregate calls, tokens, cost, and per-model / per-provider rollups."""

        where, params = _build_where(None, None, None, None, since_ms)
        conn = self._connect()
        try:
            total_calls = int(
                conn.execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0]
            )
            outcomes = {
                (row["outcome"] or "unknown"): int(row["n"])
                for row in conn.execute(
                    f"SELECT outcome, COUNT(*) AS n FROM events{where} GROUP BY outcome",
                    params,
                ).fetchall()
            }
            model_rows = conn.execute(
                f"SELECT model, COUNT(*) AS calls, "
                "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
                "COALESCE(SUM(output_tokens), 0) AS output_tokens, "
                "COALESCE(SUM(CAST(json_extract(metadata, '$.tokens.cache_read') "
                "AS INTEGER)), 0) AS cache_read, "
                "COALESCE(SUM(CAST(json_extract(metadata, '$.tokens.cache_creation') "
                "AS INTEGER)), 0) AS cache_creation "
                f"FROM events{where} GROUP BY model ORDER BY calls DESC",
                params,
            ).fetchall()
            provider_rows = conn.execute(
                "SELECT json_extract(metadata, '$.provider') AS provider, "
                f"COUNT(*) AS calls FROM events{where} GROUP BY provider ORDER BY calls DESC",
                params,
            ).fetchall()
        finally:
            conn.close()

        models: list[dict[str, Any]] = []
        total_cost = 0.0
        totals = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        for row in model_rows:
            model = row["model"] or ""
            tok = {
                "input": int(row["input_tokens"]),
                "output": int(row["output_tokens"]),
                "cache_read": int(row["cache_read"]),
                "cache_creation": int(row["cache_creation"]),
            }
            for key, value in tok.items():
                totals[key] += value
            cost = compute_cost(
                model,
                input_tokens=tok["input"],
                output_tokens=tok["output"],
                cache_read=tok["cache_read"],
                cache_creation=tok["cache_creation"],
            )
            cost_f = float(cost) if cost is not None else None
            if cost_f is not None:
                total_cost += cost_f
            models.append(
                {
                    "model": row["model"],
                    "calls": int(row["calls"]),
                    "tokens": tok,
                    "cost": cost_f,
                }
            )

        providers = [
            {"provider": row["provider"] or "unknown", "calls": int(row["calls"])}
            for row in provider_rows
        ]

        return {
            "totalCalls": total_calls,
            "totalCost": total_cost,
            "totalTokens": totals,
            "outcomes": outcomes,
            "models": models,
            "providers": providers,
        }

    def tail(self, marker: int) -> tuple[list[dict[str, Any]], int]:
        """Return spans with ``rowid > marker`` and the advanced cursor.

        rowid is SQLite's monotonic insertion counter, giving the SSE
        stream a stable "everything since last poll" cursor even when two
        events share a millisecond timestamp.
        """

        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {_SELECT} FROM events WHERE rowid > ? ORDER BY rowid",
                (marker,),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return [], marker
        new_marker = max(int(row["_rowid"]) for row in rows)
        return [_serialize(row) for row in rows], new_marker

    def max_rowid(self) -> int:
        """Current max rowid — the SSE stream starts here so it shows only new spans."""

        conn = self._connect()
        try:
            row = conn.execute("SELECT MAX(rowid) FROM events").fetchone()
        finally:
            conn.close()
        return int(row[0]) if row is not None and row[0] is not None else 0


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize(row: sqlite3.Row, *, detail: bool = False) -> dict[str, Any]:
    """Turn a SQLite row into the JSON span shape the frontend consumes.

    The flat ``input_tokens`` / ``output_tokens`` columns are authoritative
    for the headline counts; cache token flavours live only in the
    ``metadata`` JSON blob, so they're read back from there.
    """

    parsed = _parse_json(row["metadata"])
    metadata: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
    tokens_raw = metadata.get("tokens")
    tokens_meta: dict[str, Any] = tokens_raw if isinstance(tokens_raw, dict) else {}

    tokens = {
        "input": _coerce_int(row["input_tokens"]),
        "output": _coerce_int(row["output_tokens"]),
        "cache_read": _coerce_int(tokens_meta.get("cache_read")),
        "cache_creation": _coerce_int(tokens_meta.get("cache_creation")),
    }
    model = row["model"] or ""
    cost = compute_cost(
        model,
        input_tokens=tokens["input"],
        output_tokens=tokens["output"],
        cache_read=tokens["cache_read"],
        cache_creation=tokens["cache_creation"],
    )

    span: dict[str, Any] = {
        "id": row["id"],
        "agentId": row["agent_id"],
        "sessionId": row["session_id"],
        "timestamp": row["timestamp"],
        "type": row["event_type"],
        "model": row["model"],
        "provider": metadata.get("provider"),
        "durationMs": row["duration_ms"],
        "outcome": row["outcome"],
        "toolExecuted": row["tool_executed"],
        "tokens": tokens,
        "cost": float(cost) if cost is not None else None,
    }
    if detail:
        span["input"] = _parse_json(row["input"])
        span["responseText"] = metadata.get("responseText")
        span["metadata"] = metadata
    return span


def _parse_json(raw: Any) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _build_where(
    agent: str | None,
    model: str | None,
    provider: str | None,
    outcome: str | None,
    since_ms: int | None,
) -> tuple[str, list[Any]]:
    """Assemble a parameterised WHERE clause from the active filters."""

    clauses: list[str] = []
    params: list[Any] = []
    if agent:
        clauses.append("agent_id = ?")
        params.append(agent)
    if model:
        clauses.append("model = ?")
        params.append(model)
    if outcome:
        clauses.append("outcome = ?")
        params.append(outcome)
    if provider:
        clauses.append("json_extract(metadata, '$.provider') = ?")
        params.append(provider)
    if since_ms is not None:
        clauses.append("timestamp >= ?")
        params.append(since_ms)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


def _get_store(request: Request) -> SpanStore:
    store: SpanStore = request.app.state.store
    return store


def _parse_int(raw: str | None, default: int, *, name: str) -> int:
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as err:
        msg = f"invalid '{name}': expected an integer, got {raw!r}"
        raise ValueError(msg) from err


def _parse_since(raw: str | None) -> int | None:
    """Interpret ``?since=<minutes>`` as a millisecond-epoch lower bound."""

    if raw is None or raw == "":
        return None
    try:
        minutes = int(raw)
    except ValueError as err:
        msg = f"invalid 'since': expected minutes as an integer, got {raw!r}"
        raise ValueError(msg) from err
    if minutes <= 0:
        return None
    return int((time.time() - minutes * 60) * 1000)


async def spans_endpoint(request: Request) -> JSONResponse:
    """``GET /api/spans`` — paginated, filterable list."""

    store = _get_store(request)
    qp = request.query_params
    try:
        limit = _parse_int(qp.get("limit"), _DEFAULT_LIMIT, name="limit")
        offset = _parse_int(qp.get("offset"), 0, name="offset")
        since_ms = _parse_since(qp.get("since"))
    except ValueError as err:
        return JSONResponse({"error": str(err)}, status_code=400)

    limit = max(1, min(limit, _MAX_LIMIT))
    offset = max(0, offset)
    spans, total = store.list_spans(
        limit=limit,
        offset=offset,
        agent=qp.get("agent"),
        model=qp.get("model"),
        provider=qp.get("provider"),
        outcome=qp.get("outcome"),
        since_ms=since_ms,
    )
    return JSONResponse({"spans": spans, "total": total, "limit": limit, "offset": offset})


async def span_detail_endpoint(request: Request) -> JSONResponse:
    """``GET /api/spans/{span_id}`` — one span, or 404."""

    store = _get_store(request)
    span = store.get_span(request.path_params["span_id"])
    if span is None:
        return JSONResponse({"error": "span not found"}, status_code=404)
    return JSONResponse(span)


async def stats_endpoint(request: Request) -> JSONResponse:
    """``GET /api/stats`` — rollups for the dashboard's summary tiles + charts."""

    store = _get_store(request)
    try:
        since_ms = _parse_since(request.query_params.get("since"))
    except ValueError as err:
        return JSONResponse({"error": str(err)}, status_code=400)
    return JSONResponse(store.stats(since_ms=since_ms))


async def sse_endpoint(request: Request) -> EventSourceResponse:
    """``GET /api/sse/live`` — stream newly captured spans as they land."""

    store = _get_store(request)
    poll_interval = float(getattr(request.app.state, "poll_interval", 1.0))
    marker = store.max_rowid()
    return EventSourceResponse(
        live_span_stream(store, marker, poll_interval=poll_interval), ping=15
    )


async def live_span_stream(
    store: SpanStore,
    marker: int,
    *,
    poll_interval: float,
) -> AsyncGenerator[dict[str, str], None]:
    """Yield one SSE ``data`` frame per newly captured span.

    Starts at ``marker`` (the max rowid at connect time) so a client only
    sees spans captured *after* it connected. ``sse-starlette`` injects
    keepalive pings on its own (``ping=15``), so this loop only emits real
    data and sleeps between polls.
    """

    while True:
        spans, marker = store.tail(marker)
        for span in spans:
            yield {"data": json.dumps(span)}
        await asyncio.sleep(poll_interval)


__all__ = [
    "SpanStore",
    "live_span_stream",
    "span_detail_endpoint",
    "spans_endpoint",
    "sse_endpoint",
    "stats_endpoint",
]
