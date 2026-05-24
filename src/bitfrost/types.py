"""Type definitions for Bitfrost.

Three core types:

- :class:`PrivacyLevel` — enum of capture levels (minimal / standard / full).
- :class:`EventPayload` — the JSON shape ``VoightBackend`` POSTs to
  ``api.voight.xyz/v1/events``. Mirrors the TypeScript ``EventPayload`` from
  ``@voightxyz/openai/src/types.ts`` 1:1.
- :class:`BitfrostOptions` — user-facing configuration for the backends.

``EventPayload`` is a :class:`~typing.TypedDict` so any backend can build it
incrementally with regular dict operations and still get static type checking
on field names and value types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict

# ---------------------------------------------------------------------------
# Privacy levels
# ---------------------------------------------------------------------------


class PrivacyLevel(str, Enum):
    """Capture aggressiveness for prompts, response text, and tool arguments.

    - ``MINIMAL`` — drop all user/model content; keep metadata only
      (tokens, model id, timing, tool names).
    - ``STANDARD`` — full content with local PII scrubbing applied before
      transmission (12 patterns + Luhn-validated credit cards).
    - ``FULL`` — capture verbatim; no scrubbing. Use only when you trust the
      destination backend and have explicit user consent.
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


# ---------------------------------------------------------------------------
# EventPayload — wire shape for Voight ingest
# ---------------------------------------------------------------------------


class TokenBreakdown(TypedDict, total=False):
    """Token usage breakdown emitted under ``metadata.tokens``.

    ``input``, ``output``, and ``total`` are always present on captured LLM
    events. ``cache_read`` and ``cache_creation`` are emitted only when
    strictly positive. ``reasoning`` is emitted for o1/o3-class models when
    the instrumentation reports it.
    """

    input: int
    output: int
    total: int
    cache_read: int
    cache_creation: int
    reasoning: int


class ToolCallRecord(TypedDict, total=False):
    """One tool / function call entry under ``metadata.toolCalls``.

    ``arguments`` is always a JSON-encoded string (not a parsed object) — that
    matches the wire shape ``@voightxyz/openai`` and ``@voightxyz/anthropic``
    produce, so dashboards render both providers identically.
    """

    id: str
    name: str
    arguments: str


EventType = Literal["reasoning", "tool", "tx", "decision", "action", "error"]
"""Event categories accepted by ``POST /v1/events``."""

Outcome = Literal["pending", "success", "failed"]
"""Terminal outcome states for an event."""


class EventPayload(TypedDict, total=False):
    """JSON shape accepted by ``POST https://api.voight.xyz/v1/events``.

    Mirrors ``@voightxyz/openai/src/types.ts`` ``EventPayload`` interface 1:1.
    All fields are optional from the wire perspective — the backend resolves
    sensible defaults (e.g. ``timestamp`` defaults to server receive time).

    Critical wire constraint: ``durationMs`` MUST be an integer. The Voight
    ingest Zod schema rejects floats with HTTP 400. Bitfrost mappers round
    nanosecond span durations down to integer milliseconds before populating
    this field.
    """

    agentId: str
    timestamp: int | str
    type: EventType
    input: dict[str, Any]
    reasoning: str
    toolsConsidered: list[str]
    toolExecuted: str
    transaction: str | None
    amount: dict[str, Any] | None
    outcome: Outcome
    durationMs: int
    errorMessage: str
    model: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# BitfrostOptions — user-facing config
# ---------------------------------------------------------------------------


class BitfrostOptions(TypedDict, total=False):
    """User-facing configuration shared across backends.

    Every field is optional. ``agent`` resolves from environment when absent
    (``VOIGHT_AGENT`` → ``HOSTNAME`` → ``"unknown-agent"``). ``sessionId``
    auto-generates a UUID v4 per backend instance unless overridden.
    """

    agent: str
    sessionId: str
    privacy: PrivacyLevel | str
    apiBase: str
    voightApiKey: str
    enabled: bool


__all__ = [
    "BitfrostOptions",
    "EventPayload",
    "EventType",
    "Outcome",
    "PrivacyLevel",
    "TokenBreakdown",
    "ToolCallRecord",
]
