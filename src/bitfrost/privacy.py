"""PII scrubbing for Bitfrost.

Three public surfaces:

- :func:`scrub_pii` — pattern catalogue (12 patterns + Luhn-validated cards)
  applied to a single string. Pure, idempotent.
- :func:`scrub_any_value` — recursive walker over JSON-like structures.
- :func:`apply_privacy` — :class:`~bitfrost.types.EventPayload`-level filter
  driven by :class:`~bitfrost.types.PrivacyLevel`. ``minimal`` drops user
  content, ``standard`` scrubs it, ``full`` passes it through verbatim.

Pattern order
-------------
Multi-line / most-specific patterns run FIRST so that once a span is consumed
(e.g. a PEM block) later patterns cannot re-match its insides. The Anthropic
key pattern precedes the generic OpenAI rule so ``sk-ant-…`` keys never get
partially consumed as ``sk-…``. JWTs run before generic key patterns to
avoid partial matches across their dot-separated segments.
"""

from __future__ import annotations

import re
from typing import Any

from bitfrost.types import PrivacyLevel

# ---------------------------------------------------------------------------
# Pattern catalogue
# ---------------------------------------------------------------------------

RE_PEM_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)

RE_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")

# Anthropic must precede the generic OpenAI rule.
RE_ANTHROPIC = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{40,}\b")

RE_OPENAI = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")

RE_STRIPE_LIVE = re.compile(r"\b(?:sk|pk)_live_[A-Za-z0-9]{20,}\b")

RE_GITHUB_FINE = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")
RE_GITHUB_CLASSIC = re.compile(r"\bghp_[A-Za-z0-9]{36}\b")

RE_AWS_ACCESS_KEY = re.compile(r"\bAKIA[A-Z0-9]{16}\b")

RE_SLACK = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")

# Voight's own keys. Defence-in-depth: even if a user accidentally pastes
# their ``vk_…`` into a prompt, it never leaves the process in the clear
# under standard privacy.
RE_VOIGHT = re.compile(r"\bvk_[A-Za-z0-9_-]{32,}\b")

# Strict email: requires a TLD with >= 2 letters. ``support@app`` (no TLD)
# and ``email_template`` (no @) do not match.
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Phone in E.164. Looser variants produce too many false positives over
# order numbers and identifiers.
RE_PHONE_E164 = re.compile(r"\+\d{10,15}\b")


# ``(compiled_re, replacement)`` ordered by precedence.
_KEY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (RE_PEM_PRIVATE_KEY, "[REDACTED-PRIVATE-KEY]"),
    (RE_JWT, "[REDACTED-JWT]"),
    (RE_ANTHROPIC, "[REDACTED-API-KEY]"),
    (RE_OPENAI, "[REDACTED-API-KEY]"),
    (RE_STRIPE_LIVE, "[REDACTED-API-KEY]"),
    (RE_GITHUB_FINE, "[REDACTED-API-KEY]"),
    (RE_GITHUB_CLASSIC, "[REDACTED-API-KEY]"),
    (RE_AWS_ACCESS_KEY, "[REDACTED-API-KEY]"),
    (RE_SLACK, "[REDACTED-API-KEY]"),
    (RE_VOIGHT, "[REDACTED-API-KEY]"),
    (RE_EMAIL, "[REDACTED-EMAIL]"),
    (RE_PHONE_E164, "[REDACTED-PHONE]"),
)


# ---------------------------------------------------------------------------
# Credit cards (Luhn-validated)
# ---------------------------------------------------------------------------


def luhn_valid(digits: str) -> bool:
    """Validate a digit string against the Luhn checksum.

    Returns ``False`` on empty / non-digit input. Also rejects the all-zero
    string (technically Luhn-valid but never a real card number).
    """

    if not digits:
        return False
    total = 0
    alternate = False
    has_nonzero = False
    for ch in reversed(digits):
        if not ("0" <= ch <= "9"):
            return False
        n = ord(ch) - ord("0")
        if n != 0:
            has_nonzero = True
        if alternate:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alternate = not alternate
    return has_nonzero and total % 10 == 0


# 13-19 digits, optionally with single spaces or dashes between groups.
RE_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def _scrub_credit_cards(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = re.sub(r"[ -]", "", candidate)
        if not (13 <= len(digits) <= 19):
            return candidate
        if not luhn_valid(digits):
            return candidate
        return "[REDACTED-CARD]"

    return RE_CARD_CANDIDATE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Public scrubbing API
# ---------------------------------------------------------------------------


def scrub_pii(text: str) -> str:
    """Apply the PII catalogue to a single string.

    Pure, idempotent, no I/O. Callers needing to handle unknown-type leaves
    should route through :func:`scrub_any_value`, which short-circuits on
    non-string values before calling this function.
    """

    if not text:
        return text
    out = text
    for pattern, replacement in _KEY_PATTERNS:
        out = pattern.sub(replacement, out)
    return _scrub_credit_cards(out)


def scrub_any_value(value: Any) -> Any:
    """Recursively scrub every string leaf in a JSON-like value.

    Non-string primitives, lists, and plain dicts are walked structurally.
    Anything else (custom classes, sets, etc.) is returned unchanged — this
    package never serialises those.

    Returns a fresh value; the input is never mutated.
    """

    if isinstance(value, str):
        return scrub_pii(value)
    if isinstance(value, list):
        return [scrub_any_value(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_any_value(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Payload-level filter (level-aware)
# ---------------------------------------------------------------------------


# Top-level :class:`~bitfrost.types.EventPayload` fields preserved under MINIMAL.
_MINIMAL_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "type",
        "model",
        "outcome",
        "durationMs",
        "errorMessage",
        "toolExecuted",
        "agentId",
        "timestamp",
        "transaction",
        "amount",
    }
)

# ``metadata`` keys preserved under MINIMAL (numeric, identity, structural).
_MINIMAL_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "source",
        "provider",
        "providerSurface",
        "spanName",
        "api",
        "tokens",
        "sessionId",
        "spanId",
        "parentSpanId",
        "traceId",
        "endpoint",
        "routeTag",
        "streaming",
        "finishReason",
        "privacyLevel",
        "instrumentationScope",
        "responseModel",
        "tags",
    }
)


def _coerce_level(level: PrivacyLevel | str) -> PrivacyLevel:
    if isinstance(level, PrivacyLevel):
        return level
    try:
        return PrivacyLevel(level)
    except ValueError as exc:
        msg = (
            f"invalid privacy level: {level!r}. "
            f"expected one of {[lvl.value for lvl in PrivacyLevel]}"
        )
        raise ValueError(msg) from exc


def _filter_minimal_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only known structural / numeric metadata keys.

    Tool-call arguments are dropped, but the tool name survives as a tag —
    the name is a routing identifier, not user content.
    """

    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in _MINIMAL_METADATA_KEYS:
            continue
        if key == "tags" and isinstance(value, dict):
            # Tags are caller-provided dimensions (user id, plan, org).
            # Keep numeric-or-tag-shaped values; drop strings entirely as
            # they may carry PII the caller did not realise.
            out[key] = {
                tag_key: tag_value
                for tag_key, tag_value in value.items()
                if not isinstance(tag_value, str)
            }
        else:
            out[key] = value
    # Tool-call entries: keep ``id`` and ``name`` only.
    tool_calls = metadata.get("toolCalls")
    if isinstance(tool_calls, list):
        out["toolCalls"] = [
            {key: tc.get(key) for key in ("id", "name") if key in tc}
            for tc in tool_calls
            if isinstance(tc, dict)
        ]
    return out


def apply_privacy(payload: dict[str, Any], level: PrivacyLevel | str) -> dict[str, Any]:
    """Apply a privacy level to an :class:`~bitfrost.types.EventPayload`.

    Returns a fresh dict; the caller's payload is never mutated. The applied
    level is recorded under ``metadata.privacyLevel`` for audit.

    Levels:

    - :attr:`PrivacyLevel.MINIMAL` — drop ``input`` and content-bearing
      ``metadata`` fields (``responseText``, ``toolCalls[*].arguments``).
      Keep numeric, identity, and structural metadata.
    - :attr:`PrivacyLevel.STANDARD` — run :func:`scrub_any_value` over
      ``input``, ``metadata.responseText``, and ``metadata.toolCalls[*].arguments``.
    - :attr:`PrivacyLevel.FULL` — pass through verbatim (still records the
      level on ``metadata.privacyLevel`` so audits stay honest).
    """

    resolved = _coerce_level(level)
    metadata_in = payload.get("metadata") or {}
    if not isinstance(metadata_in, dict):
        metadata_in = {}

    if resolved == PrivacyLevel.FULL:
        out = {key: value for key, value in payload.items() if key != "metadata"}
        out["metadata"] = {**metadata_in, "privacyLevel": resolved.value}
        return out

    if resolved == PrivacyLevel.MINIMAL:
        out_min: dict[str, Any] = {
            key: value for key, value in payload.items() if key in _MINIMAL_TOP_LEVEL
        }
        metadata_filtered = _filter_minimal_metadata(metadata_in)
        metadata_filtered["privacyLevel"] = resolved.value
        out_min["metadata"] = metadata_filtered
        return out_min

    # STANDARD: deep-scrub user content; numeric metadata untouched.
    out_std: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "input":
            out_std[key] = scrub_any_value(value)
        elif key != "metadata":
            out_std[key] = value
    scrubbed_metadata: dict[str, Any] = {}
    for key, value in metadata_in.items():
        if key == "responseText" and isinstance(value, str):
            scrubbed_metadata[key] = scrub_pii(value)
        elif key == "toolCalls" and isinstance(value, list):
            scrubbed_metadata[key] = [
                {
                    **tool_call,
                    "arguments": scrub_pii(tool_call["arguments"])
                    if isinstance(tool_call.get("arguments"), str)
                    else tool_call.get("arguments"),
                }
                if isinstance(tool_call, dict) and "arguments" in tool_call
                else tool_call
                for tool_call in value
            ]
        else:
            scrubbed_metadata[key] = value
    scrubbed_metadata["privacyLevel"] = resolved.value
    out_std["metadata"] = scrubbed_metadata
    return out_std


__all__ = [
    "RE_ANTHROPIC",
    "RE_AWS_ACCESS_KEY",
    "RE_CARD_CANDIDATE",
    "RE_EMAIL",
    "RE_GITHUB_CLASSIC",
    "RE_GITHUB_FINE",
    "RE_JWT",
    "RE_OPENAI",
    "RE_PEM_PRIVATE_KEY",
    "RE_PHONE_E164",
    "RE_SLACK",
    "RE_STRIPE_LIVE",
    "RE_VOIGHT",
    "apply_privacy",
    "luhn_valid",
    "scrub_any_value",
    "scrub_pii",
]
