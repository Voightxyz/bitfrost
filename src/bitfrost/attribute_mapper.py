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

from collections.abc import Mapping, Sequence
from typing import Any

from bitfrost.conventions import (
    AI_MODEL_ID,
    AI_MODEL_PROVIDER,
    AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    AI_USAGE_CACHED_INPUT_TOKENS,
    AI_USAGE_COMPLETION_TOKENS,
    AI_USAGE_PROMPT_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
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

    Primary source: ``gen_ai.system`` (e.g. ``"openai"`` or ``"Anthropic"``).
    Fallback: ``ai.model.provider`` (e.g. ``"openai.responses"``).

    Vercel AI SDK provider strings may include a sub-surface
    (``"openai.responses"``) — we keep only the segment before the first dot
    so dashboards bucket all OpenAI surfaces under one provider.
    """

    raw = attributes.get(GEN_AI_SYSTEM) or attributes.get(AI_MODEL_PROVIDER) or ""
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
    cache_read = _coerce_int(
        attributes.get(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)
        or attributes.get(AI_USAGE_CACHED_INPUT_TOKENS)
    )
    cache_creation = _coerce_int(
        attributes.get(GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS)
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
    """Walk ``gen_ai.prompt.<i>.role`` / ``content`` indexed attributes."""

    return _extract_indexed("gen_ai.prompt.", attributes)


def _extract_indexed_completions(
    attributes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Walk ``gen_ai.completion.<i>.role`` / ``content`` / ``finish_reason``."""

    return _extract_indexed("gen_ai.completion.", attributes)


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
