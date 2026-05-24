"""Tests for :mod:`bitfrost.identity`.

Three pure resolvers:

- :func:`resolve_api_key` — ``options['voightApiKey']`` → ``env['VOIGHT_KEY']`` → ``None``.
- :func:`resolve_agent` — options → ``VOIGHT_AGENT`` → ``HOSTNAME`` → ``'unknown-agent'``.
- :func:`resolve_session_id` — options override → fresh UUID v4.

Empty / whitespace-only strings are treated as missing at every layer so a
mis-set environment variable (``VOIGHT_KEY=``) falls through cleanly.
"""

from __future__ import annotations

import re

from bitfrost.identity import resolve_agent, resolve_api_key, resolve_session_id

UUID_V4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# resolve_api_key
# ---------------------------------------------------------------------------


def test_resolve_api_key_uses_options_when_present() -> None:
    result = resolve_api_key({"voightApiKey": "vk_from_options"}, {"VOIGHT_KEY": "vk_from_env"})
    assert result == "vk_from_options"


def test_resolve_api_key_falls_back_to_env_when_options_missing() -> None:
    result = resolve_api_key({}, {"VOIGHT_KEY": "vk_from_env"})
    assert result == "vk_from_env"


def test_resolve_api_key_returns_none_when_both_missing() -> None:
    result = resolve_api_key({}, {})
    assert result is None


def test_resolve_api_key_treats_blank_strings_as_missing() -> None:
    """Empty and whitespace-only values fall through to the next layer."""
    # Options blank → env wins.
    assert resolve_api_key({"voightApiKey": ""}, {"VOIGHT_KEY": "vk_real"}) == "vk_real"
    assert resolve_api_key({"voightApiKey": "   "}, {"VOIGHT_KEY": "vk_real"}) == "vk_real"
    # Both blank → None.
    assert resolve_api_key({"voightApiKey": ""}, {"VOIGHT_KEY": "  "}) is None


def test_resolve_api_key_trims_whitespace_around_real_values() -> None:
    """Realistic env files often ship trailing newlines/spaces; we tolerate them."""
    assert resolve_api_key({}, {"VOIGHT_KEY": "  vk_padded  "}) == "vk_padded"


# ---------------------------------------------------------------------------
# resolve_agent
# ---------------------------------------------------------------------------


def test_resolve_agent_options_wins_over_env() -> None:
    result = resolve_agent(
        {"agent": "from-options"},
        {"VOIGHT_AGENT": "from-env-voight", "HOSTNAME": "from-env-host"},
    )
    assert result == "from-options"


def test_resolve_agent_falls_back_to_voight_agent_then_hostname() -> None:
    """Full fallback chain: options → VOIGHT_AGENT → HOSTNAME → default."""
    # VOIGHT_AGENT wins over HOSTNAME.
    assert resolve_agent({}, {"VOIGHT_AGENT": "agent-x", "HOSTNAME": "host-y"}) == "agent-x"
    # No VOIGHT_AGENT → HOSTNAME.
    assert resolve_agent({}, {"HOSTNAME": "host-y"}) == "host-y"


def test_resolve_agent_defaults_to_unknown_agent_when_everything_missing() -> None:
    """The default keeps the event ingestable even on misconfigured hosts."""
    assert resolve_agent({}, {}) == "unknown-agent"
    # Blank values are treated as missing.
    assert resolve_agent({"agent": ""}, {"VOIGHT_AGENT": "  ", "HOSTNAME": ""}) == "unknown-agent"


# ---------------------------------------------------------------------------
# resolve_session_id
# ---------------------------------------------------------------------------


def test_resolve_session_id_honors_explicit_override() -> None:
    result = resolve_session_id({"sessionId": "550e8400-e29b-41d4-a716-446655440000"})
    assert result == "550e8400-e29b-41d4-a716-446655440000"


def test_resolve_session_id_treats_blank_override_as_missing() -> None:
    """Blank overrides must NOT short-circuit — we generate a fresh UUID instead."""
    result = resolve_session_id({"sessionId": "   "})
    assert UUID_V4_RE.match(result), f"expected UUID v4, got {result!r}"


def test_resolve_session_id_generates_fresh_uuid_v4_when_no_override() -> None:
    """Two consecutive calls without override produce distinct UUID v4 strings."""
    first = resolve_session_id()
    second = resolve_session_id()
    assert UUID_V4_RE.match(first), f"expected UUID v4, got {first!r}"
    assert UUID_V4_RE.match(second), f"expected UUID v4, got {second!r}"
    assert first != second
