"""Multi-backend composition — fan one event out to N backends.

``TeeBackend`` lets a single :class:`~bitfrost.exporter.BitfrostExporter`
write to several destinations at once. The canonical example is "send
to Voight for the dashboard AND keep a local SQLite log for offline
queries AND print to the terminal for live demo" — all without
running multiple TracerProviders.

::

    from bitfrost.backends import TeeBackend
    from bitfrost.backends.console import ConsoleBackend
    from bitfrost.backends.sqlite import SQLiteBackend
    from bitfrost.backends.voight import VoightBackend

    backend = TeeBackend(
        ConsoleBackend(),
        SQLiteBackend(path="./.bitfrost/events.db"),
        VoightBackend(),  # reads VOIGHT_KEY from env
    )

Error isolation
---------------
Each child backend's ``send`` is called in turn. A raise in one child
NEVER short-circuits the others — :class:`TeeBackend` catches every
exception, routes it to its own ``on_error`` hook, and continues. This
matches the rest of Bitfrost's "never crash the host app" contract:
one flaky backend (network down, disk full, …) can't take the others
with it.

Lifecycle is broadcast: ``shutdown`` and ``force_flush`` fan out to
every child. ``force_flush`` returns ``True`` only when every child
flushed in time.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any, Protocol


class _Backend(Protocol):
    """Duck-typed shape every Bitfrost backend already satisfies."""

    def send(self, event: dict[str, Any]) -> None: ...
    def shutdown(self) -> None: ...
    def force_flush(self, timeout_millis: int = 30000) -> bool: ...


class TeeBackend:
    """Fan one event out to several backends.

    Parameters
    ----------
    *backends
        Two or more backends. Each must satisfy the
        :class:`~bitfrost.backends.base.ExportBackend` protocol —
        ``send``, ``shutdown``, ``force_flush``.
    on_error
        Optional callback for any per-child exception. The TeeBackend
        itself never raises into the caller.
    """

    def __init__(
        self,
        *backends: _Backend,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        if not backends:
            msg = "TeeBackend requires at least one child backend."
            raise ValueError(msg)
        self._backends: tuple[_Backend, ...] = backends
        self._on_error = on_error

    def send(self, event: dict[str, Any]) -> None:
        """Dispatch the event to every child. Errors in one don't stop others."""

        for backend in self._backends:
            try:
                backend.send(event)
            except BaseException as err:
                self._route_error(err)

    def shutdown(self) -> None:
        """Broadcast shutdown to every child. Idempotent + error-isolating."""

        for backend in self._backends:
            with suppress(BaseException):
                backend.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush every child. Returns ``True`` only when ALL succeed in time.

        Each child gets the full timeout budget independently — total wall
        time can therefore approach ``len(backends) * timeout_millis``
        in the pessimal case. Callers concerned about that should set the
        budget low enough that a single hang doesn't stall a whole demo.
        """

        all_ok = True
        for backend in self._backends:
            try:
                if not backend.force_flush(timeout_millis):
                    all_ok = False
            except BaseException as err:
                self._route_error(err)
                all_ok = False
        return all_ok

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _route_error(self, err: BaseException) -> None:
        if self._on_error is None:
            return
        with suppress(BaseException):
            self._on_error(err)


__all__ = ["TeeBackend"]
