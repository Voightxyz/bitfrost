"""Identity resolution: API key, agent label, session ID.

All three resolvers are pure functions — they take user-provided options and
an environment dict, and return a deterministic result. The ``env`` argument
defaults to :data:`os.environ` when called bare; passing it explicitly lets
tests exercise every fallback path without mutating global state.

Resolution priorities (left to right):

.. code-block:: text

    resolve_api_key:      options['voightApiKey']  →  env['VOIGHT_KEY']  →  None

    resolve_agent:        options['agent']
                          →  env['VOIGHT_AGENT']
                          →  env['HOSTNAME']
                          →  'unknown-agent'

    resolve_session_id:   options['sessionId']  →  fresh UUID v4

Empty and whitespace-only string values are treated as missing at every layer
so a misconfigured environment (``VOIGHT_KEY=``) falls through cleanly instead
of being mistaken for a real value.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from typing import Any

# Default fallback when nothing else resolves. Centralised so dashboards
# always show the same label for unconfigured hosts.
DEFAULT_AGENT = "unknown-agent"


def _nonblank(value: Any) -> str | None:
    """Trim and return ``value``, or ``None`` if it's missing / blank.

    Centralises the *empty counts as missing* rule that every resolver
    applies at every fallback layer.
    """

    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def resolve_api_key(
    options: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the Voight API key for outgoing events.

    Returns ``None`` when nothing usable is configured. Callers decide how
    to react — :class:`bitfrost.backends.voight.VoightBackend` logs a
    one-time warning and falls back to a no-op transport, so a missing key
    never crashes the host application.
    """

    opts: Mapping[str, Any] = options or {}
    environment: Mapping[str, str] = env if env is not None else os.environ
    return _nonblank(opts.get("voightApiKey")) or _nonblank(environment.get("VOIGHT_KEY"))


def resolve_agent(
    options: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the agent label that groups events in the dashboard.

    Always returns a non-empty string. When nothing else resolves we emit
    :data:`DEFAULT_AGENT` (``'unknown-agent'``) so the event is still
    ingestable and the misconfiguration shows up in the dashboard instead
    of being silently dropped.
    """

    opts: Mapping[str, Any] = options or {}
    environment: Mapping[str, str] = env if env is not None else os.environ
    return (
        _nonblank(opts.get("agent"))
        or _nonblank(environment.get("VOIGHT_AGENT"))
        or _nonblank(environment.get("HOSTNAME"))
        or DEFAULT_AGENT
    )


def resolve_session_id(options: Mapping[str, Any] | None = None) -> str:
    """Resolve the session ID that groups events into one trace.

    Honors an explicit override via ``options['sessionId']``; otherwise
    generates a fresh UUID v4 so every event emitted by a single backend
    instance shares one trace timeline in the dashboard.

    Blank overrides (empty string, whitespace) are treated as missing and
    fall through to a generated UUID — preventing accidentally-unset env
    vars from collapsing all events into one empty-string ``sessionId``.
    """

    opts: Mapping[str, Any] = options or {}
    override = _nonblank(opts.get("sessionId"))
    return override if override else str(uuid.uuid4())


__all__ = [
    "DEFAULT_AGENT",
    "resolve_agent",
    "resolve_api_key",
    "resolve_session_id",
]
