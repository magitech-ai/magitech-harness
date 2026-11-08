"""
Cooperative cancellation primitive (AbortSignal and AbortError).
"""

from __future__ import annotations

from collections.abc import Callable


class AbortError(Exception):
    """Raised when an operation is cancelled via an AbortSignal."""


class AbortSignal:
    """Cooperative cancellation token for in-flight tasks and streaming turns."""

    def __init__(self) -> None:
        self._aborted = False
        self._listeners: list[Callable[[], None]] = []

    @property
    def is_aborted(self) -> bool:
        """Returns True if abort() has been invoked."""
        return self._aborted

    def abort(self) -> None:
        """Trigger cancellation, notifying all registered listeners."""
        if self._aborted:
            return
        self._aborted = True
        listeners = list(self._listeners)
        self._listeners.clear()
        for listener in listeners:
            listener()

    def on_abort(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a callback to run upon cancellation.

        Returns a disposer to unregister.
        """
        if self._aborted:
            callback()
            return lambda: None

        self._listeners.append(callback)

        def disposer() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return disposer

    def throw_if_aborted(self) -> None:
        """Raise AbortError immediately if the signal has been cancelled."""
        if self._aborted:
            raise AbortError("Operation was aborted.")
