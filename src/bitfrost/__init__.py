"""Bitfrost — drop-in OpenTelemetry observability for Python LLM apps.

Made by `Voight <https://voight.xyz>`_. Licensed under MIT.

Public API
----------

.. code-block:: python

    from bitfrost import BitfrostOptions, EventPayload, PrivacyLevel

Backends ship in :mod:`bitfrost.backends`:

.. code-block:: python

    from bitfrost.backends.console import ConsoleBackend
    from bitfrost.backends.voight import VoightBackend
    from bitfrost.backends.otlp import OTLPBackend
    from bitfrost.backends.jsonl import JSONLBackend
    from bitfrost.backends.sqlite import SQLiteBackend
"""

from __future__ import annotations

from bitfrost.types import (
    BitfrostOptions,
    EventPayload,
    EventType,
    Outcome,
    PrivacyLevel,
    TokenBreakdown,
    ToolCallRecord,
)

__version__ = "0.1.0a1"

__all__ = [
    "BitfrostOptions",
    "EventPayload",
    "EventType",
    "Outcome",
    "PrivacyLevel",
    "TokenBreakdown",
    "ToolCallRecord",
    "__version__",
]
