# Write your own backend

A **backend** is any object that receives mapped events and forwards them
somewhere. Bitfrost ships six (console, SQLite, JSONL, OTLP/HTTP, Voight,
Tee), but the contract is tiny and duck-typed — if your object has the
three methods below, it's a backend. No base class to inherit.

## The contract

```python
from typing import Any


class MyBackend:
    def send(self, event: dict[str, Any]) -> None:
        """Deliver one event. Fire-and-forget."""

    def shutdown(self) -> None:
        """Release resources. Must be safe to call more than once."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Block until pending events are dispatched, or the timeout elapses."""
        return True
```

This matches the `bitfrost.backends.base.ExportBackend` `Protocol`, so a
type checker will accept your class anywhere a backend is expected — even
without importing it.

### Three rules

1. **Never raise out of `send`.** `send` runs on OpenTelemetry's
   `BatchSpanProcessor` worker thread. An exception there can disrupt
   export for *every* span. Catch your own errors and route them to an
   `on_error` callback (the convention every bundled backend follows) or
   swallow them.
2. **Don't block `send`.** If delivery is slow (network), dispatch it to a
   thread or a queue and return immediately. The bundled `VoightBackend`
   and `OTLPBackend` do exactly this.
3. **Make `shutdown` idempotent.** It may be called more than once
   (explicit shutdown + interpreter teardown).

## The event shape

Each `event` is a plain dict — the post-mapping, privacy-filtered payload.
The fields you'll typically read:

```python
{
    "agentId": "my-app",
    "type": "action",
    "model": "gpt-4o-mini",
    "durationMs": 412,
    "outcome": "success",            # or "failed" / "pending"
    "input": {"messages": [{"role": "user", "content": "..."}]},
    "metadata": {
        "provider": "openai",
        "sessionId": "…",
        "responseText": "…",
        "tokens": {"input": 1200, "output": 180, "cache_read": 0},
    },
}
```

Read everything defensively (`event.get(...)`); fields are absent when the
source span didn't provide them, and content fields are stripped at the
`minimal` privacy level.

## A complete example

A backend that POSTs each event to a webhook, off the worker thread:

```python
import json
import threading
import urllib.request
from typing import Any


class WebhookBackend:
    def __init__(self, url: str, on_error=None) -> None:
        self._url = url
        self._on_error = on_error
        self._threads: set[threading.Thread] = set()

    def send(self, event: dict[str, Any]) -> None:
        t = threading.Thread(target=self._post, args=(event,), daemon=True)
        self._threads.add(t)
        t.start()

    def _post(self, event: dict[str, Any]) -> None:
        try:
            data = json.dumps(event).encode("utf-8")
            req = urllib.request.Request(
                self._url, data=data, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5).close()
        except Exception as err:  # never raise out of send
            if self._on_error is not None:
                self._on_error(err)
        finally:
            self._threads.discard(threading.current_thread())

    def shutdown(self) -> None:
        for t in list(self._threads):
            t.join(timeout=5)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        deadline = timeout_millis / 1000.0
        for t in list(self._threads):
            t.join(timeout=deadline)
        return True
```

Use it like any bundled backend:

```python
import bitfrost

bitfrost.instrument_openai(backend=WebhookBackend("https://example.com/hook"))
```

## Tips

- Wrap several backends with `bitfrost.backends.tee.TeeBackend` to deliver
  the same event to more than one place; it isolates errors per backend.
- If your destination speaks OTLP protobuf, you don't need a backend at
  all — register OpenTelemetry's own `OTLPSpanExporter` on the same
  `TracerProvider` alongside Bitfrost's exporter, and both receive the
  full span batch.
