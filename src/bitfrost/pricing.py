"""Embedded LLM pricing table for local cost display.

This table is a **convenience** so Bitfrost's console renderer and
local dashboard can show approximate USD next to each event without
phoning home to a pricing service. It is intentionally compact —
roughly the top-20 most common production models — and uses longest-
prefix matching so dated variants (``gpt-4o-mini-2024-07-18``) resolve
to their family entry (``gpt-4o-mini``).

Authoritative source
--------------------
For the canonical pricing that drives invoiced billing, connect a Voight
backend — the API maintains a comprehensive pricing table covering every
shipping model variant including ones too volatile to embed here (GLM
weekly drops, xAI Grok tier changes, …). The numbers below are
**estimates** sufficient for "what did my agent cost in the last hour"
console output, **not** for finance reconciliation.

Pricing semantics (per 1M tokens, USD)
--------------------------------------
- ``input`` — fresh prompt tokens at full rate
- ``output`` — generation tokens at full rate
- ``cache_read`` — 0.10x input rate (Anthropic Path-A standard; OpenAI
  cached_tokens also priced at 0.5x — we use 0.10x as the per-provider
  default and document the OpenAI variant in cookbook recipes)
- ``cache_creation`` — 1.25x input rate (Anthropic 5-minute ephemeral
  cache write; the 1-hour cache write at 2x is not separately tracked
  in v0.1)
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

# (input_per_1m_usd, output_per_1m_usd)
_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "o1-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    # Anthropic — order longest-prefix-first so dated variants
    # resolve correctly under the prefix-match lookup below.
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4": (1.00, 5.00),
    "claude-haiku-3-5": (0.80, 4.00),
    # Google
    "gemini-2.0-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
}


_ONE_MILLION = Decimal(1_000_000)
_CACHE_READ_MULTIPLIER = Decimal("0.10")
_CACHE_CREATION_MULTIPLIER = Decimal("1.25")


def lookup_pricing(model: str) -> tuple[float, float] | None:
    """Resolve a model identifier to its ``(input, output)`` rate tuple.

    Uses longest-prefix matching so a dated variant
    (``"claude-haiku-4-5-20251001"``) maps to its family entry
    (``"claude-haiku-4-5"``). Returns ``None`` when no prefix matches —
    callers display ``"—"`` in the cost column.
    """

    if not model:
        return None
    # Longest-first iteration so e.g. "claude-haiku-4-5" wins over
    # "claude-haiku-4" when both would match.
    for key in sorted(_PRICING, key=len, reverse=True):
        if model.startswith(key):
            return _PRICING[key]
    return None


def compute_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> Decimal | None:
    """Return the estimated USD cost of one LLM call, or ``None``.

    Returns ``None`` when the model is unknown to the embedded table —
    callers should render ``"—"`` rather than fabricate a price.

    Decimal arithmetic is used throughout so the displayed cents are
    exact (no float drift on long-running aggregations in the console).
    """

    rates = lookup_pricing(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    input_rate_dec = Decimal(str(input_rate))
    output_rate_dec = Decimal(str(output_rate))

    cost = Decimal(0)
    if input_tokens > 0:
        cost += (Decimal(input_tokens) * input_rate_dec) / _ONE_MILLION
    if output_tokens > 0:
        cost += (Decimal(output_tokens) * output_rate_dec) / _ONE_MILLION
    if cache_read > 0:
        cost += (Decimal(cache_read) * input_rate_dec * _CACHE_READ_MULTIPLIER) / _ONE_MILLION
    if cache_creation > 0:
        cost += (
            Decimal(cache_creation) * input_rate_dec * _CACHE_CREATION_MULTIPLIER
        ) / _ONE_MILLION
    return cost


def known_models() -> Mapping[str, tuple[float, float]]:
    """Expose the embedded pricing table as a read-only mapping.

    Useful for ``ConsoleBackend`` instantiation when callers want to
    augment or override entries (e.g. add a private model). The
    returned mapping is the live module-level dict — wrap with
    ``dict(known_models())`` if you intend to mutate.
    """

    return _PRICING


__all__ = ["compute_cost", "known_models", "lookup_pricing"]
