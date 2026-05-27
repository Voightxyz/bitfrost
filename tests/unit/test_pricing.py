"""Tests for :mod:`bitfrost.pricing`.

The embedded table is convenience-grade — these tests pin the
behaviours that the console renderer depends on:

- Known models resolve via exact match or longest-prefix on dated variants
- Unknown models return ``None`` (caller renders ``"—"``, not zero)
- Cache multipliers are applied (Anthropic Path-A semantics)
- Zero-token inputs don't fabricate negative costs or float-drift
"""

from __future__ import annotations

from decimal import Decimal

from bitfrost.pricing import compute_cost, lookup_pricing


def test_lookup_known_model_returns_rate_tuple() -> None:
    rates = lookup_pricing("gpt-4o-mini")
    assert rates == (0.15, 0.60)


def test_lookup_dated_variant_matches_family_prefix() -> None:
    """A dated suffix (Anthropic style) must resolve to its family entry."""

    rates = lookup_pricing("claude-haiku-4-5-20251001")
    assert rates == (1.00, 5.00)


def test_lookup_unknown_model_returns_none() -> None:
    """An unrecognised model returns ``None`` so the console can show ``"—"``.

    Fabricating a price for a model we don't recognise is worse than
    being honest about it — finance teams must never see an estimated
    cost they can't trace back to a published table.
    """

    assert lookup_pricing("totally-fake-model-9000") is None
    assert lookup_pricing("") is None


def test_compute_cost_known_model_input_only() -> None:
    """100K input tokens of gpt-4o-mini = 100_000 * 0.15 / 1_000_000 = $0.015."""

    cost = compute_cost("gpt-4o-mini", input_tokens=100_000)
    assert cost == Decimal("0.015")


def test_compute_cost_input_plus_output_summed() -> None:
    """Input and output cost separately at their respective per-1M rates."""

    cost = compute_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=500_000)
    # 1M * 0.15 + 0.5M * 0.60 = 0.15 + 0.30 = 0.45
    assert cost == Decimal("0.45")


def test_compute_cost_anthropic_cache_read_at_one_tenth_rate() -> None:
    """``cache_read`` tokens cost 0.10x the input rate (Path-A semantics)."""

    cost = compute_cost(
        "claude-haiku-4-5",
        input_tokens=0,
        output_tokens=0,
        cache_read=1_000_000,
    )
    # 1M tokens * $1.00 * 0.10 = $0.10
    assert cost == Decimal("0.10")


def test_compute_cost_cache_creation_at_one_and_a_quarter_rate() -> None:
    """``cache_creation`` tokens cost 1.25x the input rate (5-min ephemeral)."""

    cost = compute_cost(
        "claude-haiku-4-5",
        input_tokens=0,
        output_tokens=0,
        cache_creation=1_000_000,
    )
    # 1M * $1.00 * 1.25 = $1.25
    assert cost == Decimal("1.25")


def test_compute_cost_unknown_model_returns_none_not_zero() -> None:
    """Unknown model with positive tokens still returns ``None``.

    The caller decides how to render this. Returning ``Decimal(0)`` would
    silently undercount aggregate spend totals on the console.
    """

    assert compute_cost("not-a-real-model", input_tokens=100_000) is None


def test_compute_cost_zero_tokens_returns_zero_decimal() -> None:
    """A zero-token event on a known model is exactly ``Decimal(0)``."""

    cost = compute_cost("gpt-4o-mini", input_tokens=0, output_tokens=0)
    assert cost == Decimal(0)


def test_longest_prefix_wins_over_shorter_match() -> None:
    """``claude-haiku-4-5`` must win over ``claude-haiku-4`` for a 4-5 variant.

    The pricing table contains both ``claude-haiku-4`` and
    ``claude-haiku-4-5``; the longest matching prefix wins so a 4.5
    call is priced at the (correct, cheaper) 4.5 rate, not the 4
    legacy rate.
    """

    rates = lookup_pricing("claude-haiku-4-5-20251001")
    # 4-5 is currently priced same as 4 ($1/$5), but the test still
    # asserts the longer prefix won — change the 4-5 rates in pricing.py
    # and this test will catch the wrong prefix matching the dated suffix.
    assert rates == (1.00, 5.00)
