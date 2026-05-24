"""Tests for :mod:`bitfrost.privacy`.

Coverage groups
---------------
- :func:`luhn_valid` — checksum validation for credit-card scrubbing.
- :func:`scrub_pii` — string-level pattern catalogue (12 patterns + cards).
- :func:`scrub_any_value` — recursive walker over JSON-like structures.
- :func:`apply_privacy` — payload-level filter for minimal / standard / full.

Each pattern in the catalogue gets its own positive test plus at least one
adjacency test to confirm order-sensitivity (e.g. Anthropic ``sk-ant-…``
must not be partially consumed as a generic OpenAI ``sk-…``).
"""

from __future__ import annotations

from typing import Any

import pytest

from bitfrost.privacy import apply_privacy, luhn_valid, scrub_any_value, scrub_pii
from bitfrost.types import PrivacyLevel

# ---------------------------------------------------------------------------
# luhn_valid
# ---------------------------------------------------------------------------


def test_luhn_valid_accepts_known_good_card_numbers() -> None:
    # Visa, MasterCard, Amex test numbers from the official spec.
    assert luhn_valid("4242424242424242") is True
    assert luhn_valid("5555555555554444") is True
    assert luhn_valid("378282246310005") is True


def test_luhn_valid_rejects_off_by_one_digits() -> None:
    """A single-digit edit must break the checksum (this is what Luhn is for)."""
    assert luhn_valid("4242424242424243") is False


def test_luhn_valid_returns_false_on_empty_or_non_digit() -> None:
    assert luhn_valid("") is False
    assert luhn_valid("not-a-card") is False
    assert luhn_valid("4242-4242-4242-4242") is False  # caller must strip separators


def test_luhn_valid_returns_false_on_all_zeros() -> None:
    """All-zero is technically Luhn-valid but never a real card; we reject it."""
    assert luhn_valid("0000000000000000") is False


# ---------------------------------------------------------------------------
# scrub_pii — pattern coverage (12 patterns + cards)
# ---------------------------------------------------------------------------


def test_scrub_pem_private_key_block() -> None:
    text = (
        "Here is a key:\n-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA...etc...\n-----END RSA PRIVATE KEY-----\nKeep this safe."
    )
    out = scrub_pii(text)
    assert "PRIVATE KEY" not in out
    assert "[REDACTED-PRIVATE-KEY]" in out
    assert out.startswith("Here is a key:")
    assert out.endswith("Keep this safe.")


def test_scrub_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyXzEyMyJ9.abcdefg_hij_klmnop"
    out = scrub_pii(f"Bearer {jwt}")
    assert jwt not in out
    assert "[REDACTED-JWT]" in out


def test_scrub_anthropic_key_runs_before_openai_pattern() -> None:
    """``sk-ant-…`` must NOT be partially consumed as a generic ``sk-…``."""
    text = "ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf0123456789-ZyXwVuTsRqPoNmLkJiHgFe-AbCdEf0123456789"
    out = scrub_pii(text)
    assert "sk-ant-" not in out
    assert "[REDACTED-API-KEY]" in out


def test_scrub_openai_project_and_legacy_keys() -> None:
    proj = "sk-proj-AbCdEf0123456789AbCdEf0123456789AbCdEf0123456789"
    legacy = "sk-AbCdEf0123456789AbCdEf01234"
    out = scrub_pii(f"keys: {proj} and {legacy}")
    assert proj not in out
    assert legacy not in out
    assert out.count("[REDACTED-API-KEY]") == 2


def test_scrub_stripe_live_keys() -> None:
    sk = "sk_live_AbCdEf0123456789AbCdEf01"
    pk = "pk_live_ZyXwVu0987654321ZyXwVu98"
    out = scrub_pii(f"stripe sk={sk} pk={pk}")
    assert sk not in out
    assert pk not in out
    assert out.count("[REDACTED-API-KEY]") == 2


def test_scrub_github_fine_grained_pat() -> None:
    token = "github_pat_AbCdEf0123456789AbCdEf0123456789ABcdef"
    out = scrub_pii(f"GITHUB_TOKEN={token}")
    assert token not in out
    assert "[REDACTED-API-KEY]" in out


def test_scrub_github_classic_pat() -> None:
    token = "ghp_AbCdEf0123456789AbCdEf0123456789ABcd"  # 40 chars total
    out = scrub_pii(f"GH={token}")
    assert token not in out
    assert "[REDACTED-API-KEY]" in out


def test_scrub_aws_access_key() -> None:
    key = "AKIAIOSFODNN7EXAMPLE"
    out = scrub_pii(f"AWS_ACCESS_KEY_ID={key}")
    assert key not in out
    assert "[REDACTED-API-KEY]" in out


def test_scrub_slack_token() -> None:
    token = "xoxb-12345-67890-AbCdEfGhIjKlMnOpQrStUvWx"
    out = scrub_pii(f"slack={token}")
    assert token not in out
    assert "[REDACTED-API-KEY]" in out


def test_scrub_voight_key_defense_in_depth() -> None:
    """Even our own keys get scrubbed — they shouldn't appear in user prompts."""
    key = "vk_AbCdEf0123456789AbCdEf0123456789ZyXw"
    out = scrub_pii(f"VOIGHT_KEY={key}")
    assert key not in out
    assert "[REDACTED-API-KEY]" in out


def test_scrub_email_strict_tld() -> None:
    out = scrub_pii("contact me at jane.doe@example.com or via support")
    assert "jane.doe@example.com" not in out
    assert "[REDACTED-EMAIL]" in out


def test_scrub_email_does_not_match_partial_strings() -> None:
    """``support@app`` (no TLD) and ``email_template`` (no @) must not match."""
    text = "send to support@app and load email_template"
    out = scrub_pii(text)
    assert out == text


def test_scrub_phone_e164() -> None:
    out = scrub_pii("call +14155552671 or +442071838750")
    assert "+14155552671" not in out
    assert "+442071838750" not in out
    assert out.count("[REDACTED-PHONE]") == 2


def test_scrub_credit_card_with_luhn() -> None:
    out = scrub_pii("card: 4242 4242 4242 4242 expires 12/26")
    assert "4242" not in out.replace("12/26", "")
    assert "[REDACTED-CARD]" in out


def test_scrub_credit_card_rejects_non_luhn_id_string() -> None:
    """Long digit strings that fail Luhn must NOT be redacted (order numbers, etc.)."""
    text = "order 1234567890123456 confirmed"
    out = scrub_pii(text)
    assert out == text


# ---------------------------------------------------------------------------
# scrub_pii — edge cases
# ---------------------------------------------------------------------------


def test_scrub_pii_returns_empty_string_unchanged() -> None:
    assert scrub_pii("") == ""


def test_scrub_any_value_falls_through_non_string_leaves_safely() -> None:
    """``scrub_any_value`` is the safety net for non-string leaves, not ``scrub_pii``."""
    assert scrub_any_value(42) == 42
    assert scrub_any_value(None) is None
    assert scrub_any_value(True) is True


def test_scrub_pii_is_idempotent() -> None:
    """Re-scrubbing already-scrubbed text must produce the same output."""
    text = "email jane.doe@example.com or phone +14155552671"
    once = scrub_pii(text)
    twice = scrub_pii(once)
    assert once == twice


# ---------------------------------------------------------------------------
# scrub_any_value — recursive walker
# ---------------------------------------------------------------------------


def test_scrub_any_value_scrubs_string_leaves_in_dict() -> None:
    value = {
        "user": {"email": "jane.doe@example.com", "id": 42},
        "tokens": [1, 2, 3],
        "note": "phone +14155552671",
    }
    out = scrub_any_value(value)
    assert isinstance(out, dict)
    assert out["user"]["email"] == "[REDACTED-EMAIL]"
    assert out["user"]["id"] == 42
    assert out["tokens"] == [1, 2, 3]
    assert "+14155552671" not in out["note"]


def test_scrub_any_value_walks_lists_and_nested_dicts() -> None:
    value = [
        {"role": "user", "content": "email me at a@b.com"},
        {"role": "assistant", "content": "got it"},
    ]
    out = scrub_any_value(value)
    assert isinstance(out, list)
    assert out[0]["content"] == "email me at [REDACTED-EMAIL]"
    assert out[1]["content"] == "got it"


def test_scrub_any_value_passes_through_unknown_types() -> None:
    """Non-JSON types (sets, custom classes) are returned unchanged."""

    class Marker:
        pass

    marker = Marker()
    out = scrub_any_value(marker)
    assert out is marker


def test_scrub_any_value_does_not_mutate_input() -> None:
    original = {"note": "email a@b.com"}
    snapshot = dict(original)
    scrub_any_value(original)
    assert original == snapshot


# ---------------------------------------------------------------------------
# apply_privacy — payload-level filter for minimal / standard / full
# ---------------------------------------------------------------------------


def _sample_payload() -> dict[str, Any]:
    return {
        "type": "action",
        "model": "gpt-4o-mini",
        "outcome": "success",
        "durationMs": 1234,
        "input": {"messages": [{"role": "user", "content": "email me at a@b.com"}]},
        "metadata": {
            "source": "bitfrost",
            "provider": "openai",
            "tokens": {"input": 12, "output": 5, "total": 17},
            "responseText": "Sure, contact info: jane.doe@example.com",
            "toolCalls": [
                {"id": "call_1", "name": "get_user", "arguments": '{"email": "x@y.com"}'},
            ],
        },
    }


def test_apply_privacy_full_passes_content_through_verbatim() -> None:
    payload = _sample_payload()
    snapshot = _sample_payload()
    out = apply_privacy(payload, PrivacyLevel.FULL)
    # Content fields are byte-identical to input — FULL never scrubs.
    assert out["input"] == snapshot["input"]
    assert out["metadata"]["responseText"] == snapshot["metadata"]["responseText"]
    assert out["metadata"]["toolCalls"] == snapshot["metadata"]["toolCalls"]
    # FULL still records the level on metadata for audit consistency.
    assert out["metadata"]["privacyLevel"] == "full"
    # Verbatim level must NOT mutate the caller's dict.
    assert payload == snapshot


def test_apply_privacy_standard_scrubs_user_content_but_keeps_metadata() -> None:
    payload = _sample_payload()
    out = apply_privacy(payload, PrivacyLevel.STANDARD)
    # User prompts scrubbed.
    assert out["input"]["messages"][0]["content"] == "email me at [REDACTED-EMAIL]"
    # Response text scrubbed.
    assert "jane.doe@example.com" not in out["metadata"]["responseText"]
    # Tool-call arguments scrubbed.
    assert "x@y.com" not in out["metadata"]["toolCalls"][0]["arguments"]
    # Tool name (not user content) preserved as a tag.
    assert out["metadata"]["toolCalls"][0]["name"] == "get_user"
    # Numeric metadata untouched.
    assert out["metadata"]["tokens"] == {"input": 12, "output": 5, "total": 17}
    assert out["durationMs"] == 1234
    assert out["model"] == "gpt-4o-mini"


def test_apply_privacy_minimal_drops_content_keeps_numeric_and_tags() -> None:
    payload = _sample_payload()
    out = apply_privacy(payload, PrivacyLevel.MINIMAL)
    # Content dropped entirely.
    assert "input" not in out or "messages" not in out.get("input", {})
    assert "responseText" not in out["metadata"]
    # Tool-call arguments dropped, but tool name preserved as a tag.
    if out["metadata"].get("toolCalls"):
        assert "arguments" not in out["metadata"]["toolCalls"][0]
        assert out["metadata"]["toolCalls"][0]["name"] == "get_user"
    # Numeric / tag metadata preserved.
    assert out["metadata"]["tokens"] == {"input": 12, "output": 5, "total": 17}
    assert out["durationMs"] == 1234
    assert out["model"] == "gpt-4o-mini"
    assert out["outcome"] == "success"


def test_apply_privacy_accepts_string_level_values() -> None:
    """Public API tolerates the level passed as the underlying string."""
    out = apply_privacy(_sample_payload(), "standard")  # type: ignore[arg-type]
    assert "[REDACTED-EMAIL]" in out["input"]["messages"][0]["content"]


def test_apply_privacy_rejects_invalid_level() -> None:
    with pytest.raises(ValueError):
        apply_privacy(_sample_payload(), "paranoid")  # type: ignore[arg-type]


def test_apply_privacy_stamps_privacy_level_marker_on_metadata() -> None:
    """The applied level is recorded under ``metadata.privacyLevel`` for audit."""
    out_min = apply_privacy(_sample_payload(), PrivacyLevel.MINIMAL)
    out_std = apply_privacy(_sample_payload(), PrivacyLevel.STANDARD)
    out_full = apply_privacy(_sample_payload(), PrivacyLevel.FULL)
    assert out_min["metadata"]["privacyLevel"] == "minimal"
    assert out_std["metadata"]["privacyLevel"] == "standard"
    assert out_full["metadata"]["privacyLevel"] == "full"
