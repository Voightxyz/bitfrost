# Bitfrost

Drop-in OpenTelemetry observability for Python LLM apps.

Bitfrost is a Python `SpanExporter` that turns OpenTelemetry-instrumented LLM calls (OpenAI, Anthropic, smolagents, LiteLLM, LlamaIndex, anything emitting `gen_ai.*` spans) into a live observability pipeline. Use it standalone with the bundled rich console + local web dashboard, or send to any backend — Voight, Langfuse, Phoenix, Datadog, or your own OTLP target.

> **Status**: in active development. v0.1.0 launch target Thu 2026-06-04.

## Quick start

```bash
pip install bitfrost
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from bitfrost.backends.console import ConsoleBackend

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleBackend()))
trace.set_tracer_provider(provider)

OpenAIInstrumentor().instrument()

import openai
client = openai.OpenAI()
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
```

That's it. Every LLM call lands in your terminal as a pretty-printed span.

## Backends

Bitfrost ships with five backends out of the box:

- `ConsoleBackend` — rich coloured terminal output (default standalone)
- `VoightBackend` — POST to `api.voight.xyz` (multi-tenant by `VOIGHT_KEY`)
- `OTLPBackend` — generic OTLP HTTP for Langfuse, Phoenix, Datadog, etc.
- `JSONLBackend` — file logging (replay-able)
- `SQLiteBackend` — persistent local query mode (powers `bitfrost serve`)

## CLI

```bash
bitfrost watch                # live tail of spans
bitfrost serve --port 8080    # local web dashboard
bitfrost replay spans.jsonl   # re-emit logged spans
bitfrost query spans.sqlite "SELECT * WHERE model='gpt-4o-mini'"
```

## License

MIT — see [LICENSE](./LICENSE).

Made by [Voight](https://voight.xyz).
