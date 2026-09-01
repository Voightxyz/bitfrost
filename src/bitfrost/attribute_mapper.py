"""Pure mapper from OpenTelemetry span attributes to Voight :class:`EventPayload`.

The mapper accepts a span's primitive fields (name, attributes, timing, status)
and returns a populated :class:`~bitfrost.types.EventPayload`. It returns
``None`` for non-LLM spans (no ``gen_ai.*`` and no ``ai.model.*`` attributes)
so non-LLM telemetry (HTTP, DB, FS) never pollutes the event stream.

The implementation reads OTel GenAI semantic-convention attributes as the
primary path and falls back to Vercel AI SDK ``ai.*`` attributes per-field.
Traceloop OpenLLMetry ``llm.*`` extension attributes (``reasoning_tokens``,
``total_tokens``, ``is_streaming``) are read additively when present.

Wire contract
-------------
``EventPayload["durationMs"]`` MUST be an ``int``. The Voight ingest Zod
schema rejects floats with HTTP 400. Nanosecond span durations are rounded
to the nearest millisecond using :func:`round`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import json
from collections.abc import Mapping, Sequence
from typing import Any

from bitfrost.conventions import (
    AI_MODEL_ID,
    AI_MODEL_PROVIDER,
    AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    AI_USAGE_CACHED_INPUT_TOKENS,
    AI_USAGE_COMPLETION_TOKENS,
    AI_USAGE_PROMPT_TOKENS,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS_DOTTED,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS_DOTTED,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LLM_USAGE_REASONING_TOKENS,
)
from bitfrost.types import EventPayload, Outcome, TokenBreakdown


def map_attributes(
    span_name: str,
    attributes: Mapping[str, Any],
    start_time_ns: int,
    end_time_ns: int,
    status_code: str = "UNSET",
    status_description: str | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    instrumentation_scope_name: str | None = None,
) -> EventPayload | None:
    """Map raw OTel span fields to a Voight :class:`EventPayload`.

    Returns ``None`` if the span carries no LLM attributes (neither
    ``gen_ai.*`` nor ``ai.model.*``). Non-LLM spans (HTTP, DB, FS) are
    silently skipped — this keeps the Voight event stream clean even when
    the host process emits many non-LLM spans through the same TracerProvider.

    Parameters
    ----------
    span_name
        OTel span name (e.g. ``"openai.chat"``, ``"anthropic.chat"``).
    attributes
        OTel span attributes dict. Read-only — never mutated.
    start_time_ns, end_time_ns
        OTel span timing, in nanoseconds since UNIX epoch.
    status_code
        Last segment of the OTel ``StatusCode`` enum — one of ``"UNSET"``,
        ``"OK"``, ``"ERROR"``. ``"UNSET"`` is treated as success because
        ``opentelemetry-instrumentation-openai`` (and other instrumentations)
        do not set OK explicitly on success.
    status_description
        Free-text status description. Surfaced as ``errorMessage`` when
        ``status_code == "ERROR"``.
    events
        OTel span events. Reserved for future tool-call extraction (v0.2+).
    instrumentation_scope_name
        Name of the OTel instrumentation that emitted the span (e.g.
        ``"opentelemetry.instrumentation.openai.v1"``). Surfaced for
        debugging under ``metadata.instrumentationScope``.

    Returns
    -------
    EventPayload | None
        Populated event payload, or ``None`` for non-LLM spans.
    """

    if not _is_llm_span(attributes):
        return None

    provider = _extract_provider(attributes)
    model = _extract_model(attributes)
    response_model = attributes.get(GEN_AI_RESPONSE_MODEL)

    duration_ms = max(0, round((end_time_ns - start_time_ns) / 1_000_000))

    outcome, error_message = _outcome_from_status(status_code, status_description)

    tokens = _extract_tokens(attributes)
    prompts = _extract_indexed_prompts(attributes)
    completions = _extract_indexed_completions(attributes)
    response_text = completions[0]["content"] if completions else None
    finish_reason = completions[0].get("finish_reason") if completions else None

    metadata: dict[str, Any] = {
        "source": "bitfrost",
        "provider": provider,
        "spanName": span_name,
        "tokens": tokens,
    }
    if response_model:
        metadata["responseModel"] = response_model
    if instrumentation_scope_name:
        metadata["instrumentationScope"] = instrumentation_scope_name
    if response_text is not None:
        metadata["responseText"] = response_text
    if finish_reason is not None:
        metadata["finishReason"] = finish_reason

    event: EventPayload = {
        "type": "action",
        "model": model or "unknown",
        "outcome": outcome,
        "durationMs": duration_ms,
        "metadata": metadata,
    }
    if start_time_ns:
        # The span's own start time, not the ingest arrival time — batched or
        # delayed sends would otherwise land with a lying timeline.
        event["timestamp"] = (
            datetime.fromtimestamp(start_time_ns / 1e9, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if prompts:
        event["input"] = {"messages": prompts}
    if error_message is not None:
        event["errorMessage"] = error_message

    return event


# ---------------------------------------------------------------------------
# Internal helpers (pure, no side effects)
# ---------------------------------------------------------------------------


def _is_llm_span(attributes: Mapping[str, Any]) -> bool:
    """An LLM span carries at least one ``gen_ai.*`` or ``ai.model.*`` attr."""

    return any(key.startswith("gen_ai.") or key.startswith("ai.model.") for key in attributes)


def _extract_provider(attributes: Mapping[str, Any]) -> str:
    """Return the lowercased base provider name.

    Priority chain (OTel GenAI semconv evolution + Vercel fallback):

    1. ``gen_ai.provider.name`` — OTel GenAI semconv **v1.32+** canonical
       (opentelemetry-instrumentation-openai >= 0.60, anthropic >= 0.60,
       smolagents, LiteLLM modern).
    2. ``gen_ai.system`` — OTel GenAI semconv **v1.27-1.31** (older
       instrumentation libraries; kept as fallback so older spans still
       map cleanly).
    3. ``ai.model.provider`` — **Vercel AI SDK** convention. May include
       a sub-surface (``"openai.responses"``) — we keep only the
       segment before the first dot so dashboards bucket all OpenAI
       surfaces under one provider.

    Returns the empty string when no source matches; callers cascade.
    """

    raw = (
        attributes.get(GEN_AI_PROVIDER_NAME)
        or attributes.get(GEN_AI_SYSTEM)
        or attributes.get(AI_MODEL_PROVIDER)
        or ""
    )
    raw_str = str(raw).lower()
    if "." in raw_str:
        raw_str = raw_str.split(".", 1)[0]
    return raw_str


def _extract_model(attributes: Mapping[str, Any]) -> str | None:
    raw = attributes.get(GEN_AI_REQUEST_MODEL) or attributes.get(AI_MODEL_ID)
    return str(raw) if raw else None


def _outcome_from_status(
    status_code: str, status_description: str | None
) -> tuple[Outcome, str | None]:
    """Map OTel status to Voight outcome + optional error message.

    ``UNSET`` and ``OK`` both map to ``success`` (instrumentation libraries
    differ in whether they set OK explicitly). ``ERROR`` maps to ``failed``
    with the status description surfaced as ``errorMessage``.
    """

    code = (status_code or "").upper()
    if code == "ERROR":
        return "failed", status_description or "unknown error"
    return "success", None


def _extract_tokens(attributes: Mapping[str, Any]) -> TokenBreakdown:
    """Build the ``metadata.tokens`` dict from gen_ai.* / ai.* / llm.* attrs.

    ``input``, ``output``, ``total`` are always present (zeros are valid).
    ``cache_read``, ``cache_creation``, ``reasoning`` are included only when
    strictly positive so the on-wire payload stays tight for non-cache calls.
    """

    input_tokens = _coerce_int(
        attributes.get(GEN_AI_USAGE_INPUT_TOKENS) or attributes.get(AI_USAGE_PROMPT_TOKENS)
    )
    output_tokens = _coerce_int(
        attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS) or attributes.get(AI_USAGE_COMPLETION_TOKENS)
    )
    # Cache attrs: v1.32+ uses a DOTTED variant
    # (``gen_ai.usage.cache_read.input_tokens``) while v1.27 uses an
    # underscore variant (``gen_ai.usage.cache_read_input_tokens``).
    # Modern instrumentation libraries emit the dotted form; we read
    # whichever is present (they never coexist on the same span) and
    # fall back to the Vercel AI SDK convention last.
    cache_read = _coerce_int(
        attributes.get(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS_DOTTED)
        or attributes.get(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)
        or attributes.get(AI_USAGE_CACHED_INPUT_TOKENS)
    )
    cache_creation = _coerce_int(
        attributes.get(GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS_DOTTED)
        or attributes.get(GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS)
        or attributes.get(AI_USAGE_CACHE_CREATION_INPUT_TOKENS)
    )
    reasoning = _coerce_int(attributes.get(LLM_USAGE_REASONING_TOKENS))

    tokens: TokenBreakdown = {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }
    if cache_read > 0:
        tokens["cache_read"] = cache_read
    if cache_creation > 0:
        tokens["cache_creation"] = cache_creation
    if reasoning > 0:
        tokens["reasoning"] = reasoning
    return tokens


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_indexed_prompts(
    attributes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract prompt messages, supporting both semconv generations.

    v1.32+ ships one attribute ``gen_ai.input.messages`` containing a
    JSON-stringified array. v1.27 ships indexed attributes
    ``gen_ai.prompt.N.role`` / ``content``. We try v1.32+ first because
    it is the canonical modern shape; fall back to the indexed walk if
    not present (or if the JSON is malformed).
    """

    parsed = _parse_messages_blob(attributes.get(GEN_AI_INPUT_MESSAGES))
    if parsed is not None:
        return parsed
    return _extract_indexed("gen_ai.prompt.", attributes)


def _extract_indexed_completions(
    attributes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract completion messages, supporting both semconv generations.

    v1.32+ ships ``gen_ai.output.messages`` (JSON blob) plus a sibling
    top-level array ``gen_ai.response.finish_reasons`` aligned by index.
    We fold the matching finish reason into each parsed message so
    downstream callers see the same shape as the v1.27 indexed walk
    (``role`` / ``content`` / ``finish_reason``).
    """

    parsed = _parse_messages_blob(attributes.get(GEN_AI_OUTPUT_MESSAGES))
    if parsed is not None:
        finish_reasons = attributes.get(GEN_AI_RESPONSE_FINISH_REASONS)
        if isinstance(finish_reasons, (list, tuple)):
            for i, msg in enumerate(parsed):
                if i < len(finish_reasons) and "finish_reason" not in msg:
                    msg["finish_reason"] = finish_reasons[i]
        return parsed
    return _extract_indexed("gen_ai.completion.", attributes)


def _parse_messages_blob(value: Any) -> list[dict[str, Any]] | None:
    """Parse a v1.32+ ``gen_ai.{input,output}.messages`` JSON string.

    Spec shape::

        [{"role": "user",
          "parts": [{"type": "text", "content": "Hi"}]}]

    The mapper flattens ``parts[*].content`` into a single ``content``
    string per message (joined by newlines for multi-part messages) so
    the rest of the mapper keeps a flat shape consistent with the v1.27
    indexed walk. Non-text parts (image, audio) are skipped — the
    payload metadata stays scrubbing-friendly. Returns ``None`` on
    parse error or wrong shape so the caller falls back to the
    indexed walk.
    """

    if not isinstance(value, str) or not value:
        return None
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    out: list[dict[str, Any]] = []
    for msg in data:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = _extract_message_content(msg)
        entry: dict[str, Any] = {"role": role, "content": content}
        # Preserve any explicit finish_reason on the message itself
        # (Anthropic 0.60+ stamps it inline on output messages).
        if "finish_reason" in msg:
            entry["finish_reason"] = msg["finish_reason"]
        out.append(entry)
    return out


def _extract_message_content(msg: dict[str, Any]) -> str:
    """Pull text content from a v1.32+ message dict.

    Handles both shapes the spec allows:
    - ``{"content": "..."}`` (legacy passthrough)
    - ``{"parts": [{"type": "text", "content": "..."}, ...]}``
    """

    parts = msg.get("parts")
    if isinstance(parts, list):
        texts = [
            str(p.get("content", ""))
            for p in parts
            if isinstance(p, dict) and p.get("type") in (None, "text")
        ]
        return "\n".join(t for t in texts if t)
    return str(msg.get("content", ""))


def _extract_indexed(prefix: str, attributes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generic indexed-attribute walker.

    Reads ``<prefix><i>.<field>`` keys and groups them by index into dicts.
    Returns an ordered list sorted by index. Missing intermediate indices
    are tolerated (gap-friendly).
    """

    by_index: dict[int, dict[str, Any]] = {}
    for key, value in attributes.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        head, _, field = remainder.partition(".")
        if not head.isdigit() or not field:
            continue
        idx = int(head)
        by_index.setdefault(idx, {})[field] = value
    return [by_index[i] for i in sorted(by_index)]


__all__ = ["map_attributes"]
