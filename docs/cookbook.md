# Cookbook

Task-oriented recipes. Every snippet uses the real public API — copy,
paste, run.

> Most recipes assume you've instrumented a library that emits OpenTelemetry
> GenAI spans (e.g. `pip install 'bitfrost[http]' openai
> opentelemetry-instrumentation-openai`). The capture/replay recipes work
> with any captured file.

---

## 1. See every LLM call in your terminal

The fastest start. `quickstart` installs a console backend and instruments
whatever supported libraries are importable.

```python
import bitfrost

bitfrost.quickstart(agent="my-app")

# ...your normal OpenAI/Anthropic/LiteLLM/smolagents calls...
```

Each call prints one color-coded line: time, agent, model, tokens,
duration, status, and an estimated cost.

---

## 2. Keep a persistent log and open the dashboard

```python
import bitfrost
from bitfrost.backends.sqlite import SQLiteBackend

bitfrost.instrument_openai(backend=SQLiteBackend("events.db"), agent="my-app")
```

Then, in another terminal:

```bash
bitfrost serve events.db          # web dashboard at http://127.0.0.1:8080
# or
bitfrost tui events.db            # full-screen terminal dashboard
```

`SQLiteBackend` keeps 7 days by default; change it with
`SQLiteBackend("events.db", retention_days=30)`, or prune on demand with
`bitfrost vacuum events.db --keep-days 7`.

---

## 3. Log to a file and replay it later

```python
import bitfrost
from bitfrost.backends.jsonl import JSONLBackend

bitfrost.instrument_anthropic(backend=JSONLBackend("run.jsonl"), agent="my-app")
```

```bash
bitfrost replay run.jsonl                  # re-render instantly
bitfrost replay run.jsonl --follow-timing  # honor the original gaps (great for demos)
```

---

## 4. Send to several places at once

`TeeBackend` fans every event out to multiple backends and isolates
errors per backend, so a flaky one never breaks the others.

```python
import bitfrost
from bitfrost.backends.tee import TeeBackend
from bitfrost.backends.console import ConsoleBackend
from bitfrost.backends.sqlite import SQLiteBackend

backend = TeeBackend(
    ConsoleBackend(),               # watch it live
    SQLiteBackend("events.db"),     # and keep a queryable log
)
bitfrost.instrument_auto(backend=backend, agent="my-app")
```

---

## 5. Ship to a hosted Voight project

```python
import bitfrost
from bitfrost.backends.voight import VoightBackend

backend = VoightBackend(api_key="vk_...")   # or set VOIGHT_KEY in the env
bitfrost.instrument_auto(backend=backend, agent="my-app")
```

`VoightBackend` is the only backend that talks to a hosted service, and
it's entirely opt-in. Without it, Bitfrost makes no network calls.

---

## 6. Control what content is captured

```python
import bitfrost

# metadata only — no prompt/response text leaves the process
bitfrost.instrument_openai(privacy="minimal")

# default — content kept, PII scrubbed (12 patterns + Luhn)
bitfrost.instrument_openai(privacy="standard")

# everything verbatim — local debugging only
bitfrost.instrument_openai(privacy="full")
```

---

## 7. Run alongside another OpenTelemetry exporter

Bitfrost augments your `TracerProvider` instead of taking it over, so it
co-exists with any other OTel exporter. If you also want to ship spans to
a backend that speaks OTLP protobuf, register its `OTLPSpanExporter` on
the same provider — both receive the full span batch.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from bitfrost.exporter import BitfrostExporter
from bitfrost.backends.console import ConsoleBackend

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(BitfrostExporter(ConsoleBackend())))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")))
trace.set_tracer_provider(provider)
```

---

## 8. Wire the exporter by hand

If you manage OpenTelemetry yourself, skip the helpers and attach
`BitfrostExporter` directly. It accepts any backend plus optional `agent`,
`session_id`, and `privacy`.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from bitfrost.exporter import BitfrostExporter
from bitfrost.backends.sqlite import SQLiteBackend

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(
        BitfrostExporter(SQLiteBackend("events.db"), agent="my-app", privacy="standard")
    )
)
trace.set_tracer_provider(provider)

OpenAIInstrumentor().instrument()
```

---

## 9. LangChain (and other OTel-native frameworks)

LangChain emits OpenTelemetry spans through its own instrumentation, so
Bitfrost captures it without a dedicated helper — instrument LangChain the
usual OTel way, with a `BitfrostExporter` on the provider (see recipe 8).
A first-class `instrument_langchain()` helper is planned for a later
release; until then the manual-wiring recipe covers it fully.

---

## 10. Query a capture with SQL

`SQLiteBackend` writes a flat table, so ad-hoc analysis is just SQL:

```bash
bitfrost query events.db "SELECT model, COUNT(*) AS calls, SUM(input_tokens) AS in_tok \
                          FROM events GROUP BY model ORDER BY calls DESC"
```

The connection is opened read-only, so a typo can never modify your
capture.
