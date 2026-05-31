"""Static-file wiring + offline guarantee for the ``bitfrost serve`` frontend.

The frontend itself is verified visually (Stage 9), but the *wiring* is
testable: the static bundle must be mounted at ``/`` (without shadowing
``/api/*``), and the dashboard must run fully offline — no asset may be
loaded from a CDN. This guards the "works on a plane" promise the docs
make, since a single ``https://cdn...`` slipped into ``index.html`` would
break it silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("starlette")
pytest.importorskip("sse_starlette")

from starlette.testclient import TestClient

from bitfrost.backends.sqlite import SQLiteBackend
from bitfrost.serve.app import build_app

_STATIC = Path(__file__).resolve().parents[2] / "src" / "bitfrost" / "serve" / "static"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    db = tmp_path / "static.db"
    backend = SQLiteBackend(db, retention_days=0)
    backend.send(
        {
            "agentId": "static-test",
            "type": "action",
            "model": "gpt-4o-mini",
            "outcome": "success",
            "metadata": {"provider": "openai", "tokens": {"input": 1, "output": 1}},
        }
    )
    backend.shutdown()
    return TestClient(build_app(db))


def test_index_served_at_root(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "bitfrost" in resp.text


def test_static_assets_served(client: TestClient) -> None:
    for asset, ctype in [
        ("/styles.css", "css"),
        ("/app.js", "javascript"),
        ("/chart.min.js", "javascript"),
    ]:
        resp = client.get(asset)
        assert resp.status_code == 200, asset
        assert ctype in resp.headers["content-type"], asset


def test_api_not_shadowed_by_static_mount(client: TestClient) -> None:
    # The static mount lives at "/", but /api/* must still resolve to JSON.
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    assert "totalCalls" in resp.json()


def test_dashboard_is_offline_no_cdn() -> None:
    # No *asset* (script / stylesheet / image / font / CSS import) may load
    # from the network — that's the "runs on a plane" guarantee. A brand
    # anchor (<a href="https://voight.xyz">) is fine: it's a link the user
    # may click, not a resource the page loads to render.
    import re

    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src="https?://', html)
    assert not re.search(r'<link[^>]+href="https?://', html)
    assert not re.search(r'<img[^>]+src="https?://', html)
    assert "url(http" not in html  # no CSS @import / url() to a CDN


def test_chart_js_is_vendored_real_js() -> None:
    chart = (_STATIC / "chart.min.js").read_text(encoding="utf-8")
    assert "Chart" in chart
    assert len(chart) > 50_000  # a real Chart.js bundle, not a stub/redirect
