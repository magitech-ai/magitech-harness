"""Cooperative cancellation primitive (AbortSignal and AbortError)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class AbortError(Exception):
    """Raised when an operation is cancelled via an AbortSignal."""


class AbortSignal:
    """Cooperative cancellation token for in-flight tasks and streaming turns."""

    def __init__(self) -> None:
        self._aborted = False
        self._reason: Any = None
        self._listeners: list[Callable[[], None]] = []

    @property
    def is_aborted(self) -> bool:
        """Returns True if abort() has been invoked."""
        return self._aborted

    @property
    def aborted(self) -> bool:
        """Alias for is_aborted."""
        return self._aborted

    @property
    def reason(self) -> Any:
        """Returns the optional abort reason."""
        return self._reason

    def abort(self, reason: Any = None) -> None:
        """Trigger cancellation, notifying all registered listeners."""
        if self._aborted:
            return
        self._aborted = True
        self._reason = reason
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

    def add_listener(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Alias for on_abort."""
        return self.on_abort(callback)

    def throw_if_aborted(self) -> None:
        """Raise AbortError immediately if the signal has been cancelled."""
        if self._aborted:
            msg = (
                f"Operation was aborted: {self._reason}"
                if self._reason
                else "Operation was aborted."
            )
            raise AbortError(msg)


class AbortController:
    """Controller paired with an owned AbortSignal."""

    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: Any = None) -> None:
        """Trigger cancellation on the owned signal."""
        self.signal.abort(reason)
