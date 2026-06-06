<div align="center">

<img src="https://raw.githubusercontent.com/Voightxyz/bitfrost/main/assets/logo.png" alt="bitfrost" height="56" />

# bitfrost

**Drop-in OpenTelemetry observability for Python LLM apps.**
Standalone or with any OTLP backend — runs on your machine, sends nothing unless you ask it to.

[![PyPI](https://img.shields.io/pypi/v/bitfrost.svg)](https://pypi.org/project/bitfrost/)
[![Python](https://img.shields.io/pypi/pyversions/bitfrost.svg)](https://pypi.org/project/bitfrost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/Voightxyz/bitfrost/actions/workflows/ci.yml/badge.svg)](https://github.com/Voightxyz/bitfrost/actions/workflows/ci.yml)

</div>

---

Bitfrost turns the OpenTelemetry spans your LLM libraries already emit into a clean, queryable event stream — in your terminal, in a local web dashboard, in SQLite, or shipped to any backend you choose. One line to start, zero accounts required.

```python
import bitfrost

bitfrost.quickstart(agent="my-app")   # auto-detects openai / anthropic / litellm / smolagents
# ...your normal LLM calls now stream to the terminal, color-coded, with tokens + cost.
```

<div align="center">
  <img src="https://raw.githubusercontent.com/Voightxyz/bitfrost/main/assets/dashboard.png" alt="bitfrost serve — local web dashboard" width="100%" />
  <br />
  <em><code>bitfrost serve capture.db</code> — a local, offline dashboard. No account, no upload.</em>
</div>

## Why bitfrost

- **Standalone first.** Capture to your terminal, a JSONL file, or SQLite with zero configuration and zero network calls. Voight is one optional backend, never a requirement.
- **Drop-in.** Built on the OpenTelemetry GenAI semantic conventions, so it works with the instrumentation libraries you already use — across both the v1.27 and v1.32+ attribute generations.
- **Batteries included.** Five backends, four auto-instrument helpers, a rich CLI, an interactive TUI, and an offline web dashboard.
- **Private by default.** Three privacy levels with PII scrubbing (12 patterns + Luhn) applied before any event leaves the process.
- **Never crashes your app.** Every failure path degrades gracefully — a dead endpoint or a missing optional dependency never takes down your code.

## Install

```bash
pip install bitfrost                 # core
pip install 'bitfrost[cli,serve]'    # + CLI, TUI and local web dashboard
pip install 'bitfrost[all]'          # everything
```

Python 3.10–3.13.

## Capture anywhere

Pick a backend and pass it to any instrument helper (or `quickstart`). All concrete backends live under `bitfrost.backends.*`:

| Backend | Import | Use it for |
|---|---|---|
| Console | `bitfrost.backends.console.ConsoleBackend` | live, color-coded terminal output |
| SQLite | `bitfrost.backends.sqlite.SQLiteBackend` | persistent local log; powers `bitfrost serve` |
| JSONL | `bitfrost.backends.jsonl.JSONLBackend` | one JSON object per line; replay-able |
| OTLP/HTTP | `bitfrost.backends.otlp.OTLPBackend` | POST events as JSON to any collector or webhook |
| Voight | `bitfrost.backends.voight.VoightBackend` | hosted dashboards (optional, opt-in) |
| Tee | `bitfrost.backends.tee.TeeBackend` | fan out to several backends at once |

```python
import bitfrost
from bitfrost.backends.sqlite import SQLiteBackend

bitfrost.instrument_openai(backend=SQLiteBackend("events.db"), agent="my-app")
# then:  bitfrost serve events.db
```

Fan out to several at once:

```python
import bitfrost
from bitfrost.backends.tee import TeeBackend
from bitfrost.backends.console import ConsoleBackend
from bitfrost.backends.sqlite import SQLiteBackend

bitfrost.instrument_auto(
    backend=TeeBackend(ConsoleBackend(), SQLiteBackend("events.db")),
    agent="my-app",
)
```

## Auto-instrument helpers

```python
import bitfrost

bitfrost.instrument_openai()       # opentelemetry-instrumentation-openai
bitfrost.instrument_anthropic()    # opentelemetry-instrumentation-anthropic
bitfrost.instrument_litellm()      # litellm CustomLogger adapter
bitfrost.instrument_smolagents()   # openinference-instrumentation-smolagents
bitfrost.instrument_auto()         # instrument every supported lib that's installed
```

Already wiring OpenTelemetry yourself? Skip the helpers and attach the exporter to your own `TracerProvider`:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from bitfrost.exporter import BitfrostExporter
from bitfrost.backends.console import ConsoleBackend

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(BitfrostExporter(ConsoleBackend(), agent="my-app"))
)
```

## CLI

```bash
bitfrost watch  capture.db        # live tail, one styled line per event
bitfrost replay capture.jsonl     # re-render a captured run start to finish
bitfrost query  capture.db "SELECT model, COUNT(*) FROM events GROUP BY model"
bitfrost vacuum capture.db --keep-days 7
bitfrost tui    capture.db        # full-screen interactive dashboard
bitfrost serve  capture.db        # local web dashboard at http://127.0.0.1:8080
```

<div align="center">
  <img src="https://raw.githubusercontent.com/Voightxyz/bitfrost/main/assets/tui.png" alt="bitfrost tui — interactive terminal dashboard" width="100%" />
  <br />
  <em><code>bitfrost tui</code> — the same capture, navigable in your terminal. Prompts masked by default.</em>
</div>

## Privacy

Three levels, applied in-process before any event is sent:

- `minimal` — metadata only, no prompt/response content
- `standard` *(default)* — content kept, PII scrubbed (12 patterns + credit-card Luhn check)
- `full` — everything verbatim (use only for local debugging)

```python
bitfrost.instrument_openai(privacy="minimal")
```

## Standalone, or with Voight

Bitfrost is useful entirely on its own — the core has no dependency on any hosted service. If you want managed dashboards, team sharing, and per-user cost attribution, the optional `VoightBackend` ships your events to [voight.xyz](https://voight.xyz). Everything else stays exactly the same.

## Documentation

- [Cookbook](https://github.com/Voightxyz/bitfrost/blob/main/docs/cookbook.md) — task-oriented recipes
- [Write your own backend](https://github.com/Voightxyz/bitfrost/blob/main/docs/custom_backend.md) — the `ExportBackend` protocol
- Full docs: [docs.voight.xyz/python/bitfrost](https://docs.voight.xyz/python/bitfrost)

## License

MIT. Built by [Voight](https://voight.xyz).
