"""Pluggable backends for Bitfrost.

Five backends ship in v0.1.0; users compose them with
:class:`bitfrost.exporter.BitfrostExporter`.

.. code-block:: python

    from bitfrost.backends.console import ConsoleBackend
    from bitfrost.backends.voight import VoightBackend
    from bitfrost.backends.jsonl import JSONLBackend
    from bitfrost.backends.sqlite import SQLiteBackend
    from bitfrost.backends.otlp import OTLPBackend

Anyone can add a custom backend by satisfying the
:class:`~bitfrost.backends.base.ExportBackend` protocol — see
``docs/custom_backend.md``.
"""

from __future__ import annotations

from bitfrost.backends.base import ExportBackend

__all__ = ["ExportBackend"]
