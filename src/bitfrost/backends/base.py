"""Backend protocol — every Bitfrost destination satisfies this contract.

A backend receives :class:`~bitfrost.types.EventPayload` dicts (the post-mapper
Voight wire shape) and forwards them to its destination. Backends MUST:

- Be non-blocking. ``send`` is called from the OTel ``BatchSpanProcessor``
  worker thread; expensive I/O dispatches to a separate thread.
- Never raise out of ``send``. Errors route to the backend's own
  ``on_error`` hook (when configured) or get swallowed silently.
- Survive ``shutdown`` being called multiple times.

The protocol is :class:`~typing.Protocol`-based so custom backends don't have
to inherit from any class — match the signatures and you're done. The
companion ``docs/custom_backend.md`` walks through writing one.
"""

from __future__ import annotations

from typing import Any, Protocol


class ExportBackend(Protocol):
    """Destination for mapped Bitfrost events.

    Required surface:

    - ``send(event)`` — fire-and-forget delivery of one
      :class:`~bitfrost.types.EventPayload`. Returns synchronously, never
      raises, may dispatch I/O on a background thread.
    - ``shutdown()`` — release resources. Idempotent.
    - ``force_flush(timeout_millis)`` — block until pending events have
      been dispatched, or ``timeout_millis`` elapses. Returns ``True`` on
      successful flush, ``False`` on timeout or error.
    """

    def send(self, event: dict[str, Any]) -> None: ...
    def shutdown(self) -> None: ...
    def force_flush(self, timeout_millis: int = 30000) -> bool: ...


__all__ = ["ExportBackend"]
