"""Cooperative cancellation primitive (AbortSignal and AbortError)."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from typing import Any, Self

MAX_TIMER_DELAY_MS = 2_147_483_647


class TimeoutReason(Exception):
    """Structured abort reason carrying a capability-owned code and deadline."""

    def __init__(self, code: str, timeout_ms: float) -> None:
        self.code = code
        self.timeout_ms = timeout_ms
        super().__init__(f"{code} after {timeout_ms}ms")


def _assert_timer_delay(timeout_ms: float, name: str) -> None:
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, (int, float))
        or not math.isfinite(timeout_ms)
        or timeout_ms <= 0
        or timeout_ms > MAX_TIMER_DELAY_MS
    ):
        raise ValueError(
            f"{name} must be a positive finite number no greater than "
            f"{MAX_TIMER_DELAY_MS}"
        )


def clamp_timeout(
    requested: float | None,
    default: float,
    maximum: float,
    name: str = "timeoutMs",
) -> float:
    """Validate optional timeout hint, fall back to *default*, and cap at *maximum*."""
    if requested is not None and (
        isinstance(requested, bool)
        or not isinstance(requested, (int, float))
        or not math.isfinite(requested)
        or requested <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return min(default if requested is None else requested, maximum)


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


class Deadline:
    """Deadline signal paired with a dispose-once cleanup hook."""

    def __init__(self, signal: AbortSignal, dispose_fn: Callable[[], None]) -> None:
        self.signal = signal
        self._dispose = dispose_fn
        self._disposed = False

    def dispose(self) -> None:
        """Clear any armed deadline timer (idempotent)."""
        if self._disposed:
            return
        self._disposed = True
        self._dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.dispose()


def deadline(
    upstream: AbortSignal | None,
    timeout_ms: float,
    code: str,
) -> Deadline:
    """Fuse an optional upstream AbortSignal with a timer stamping TimeoutReason."""
    if timeout_ms <= 0:
        return Deadline(
            upstream if upstream is not None else AbortSignal(),
            lambda: None,
        )

    _assert_timer_delay(timeout_ms, "deadline timeoutMs")

    fused = AbortController()
    if upstream is not None and upstream.aborted:
        fused.abort(upstream.reason)
        return Deadline(fused.signal, lambda: None)

    reason = TimeoutReason(code, timeout_ms)
    loop = asyncio.get_running_loop()
    timer_handle = loop.call_later(timeout_ms / 1000.0, fused.abort, reason)
    unlisten: Callable[[], None] | None = None
    if upstream is not None:

        def _on_upstream_abort() -> None:
            fused.abort(upstream.reason)

        unlisten = upstream.on_abort(_on_upstream_abort)

    def _cleanup() -> None:
        timer_handle.cancel()
        if unlisten is not None:
            unlisten()

    return Deadline(fused.signal, _cleanup)


def combine_signals(*signals: AbortSignal) -> Deadline:
    """Fuse multiple AbortSignals into one Deadline whose signal aborts with the first."""
    fused = AbortController()
    unsubs: list[Callable[[], None]] = []

    def _cleanup() -> None:
        for unsub in unsubs:
            unsub()
        unsubs.clear()

    for sig in signals:
        if sig.aborted:
            fused.abort(sig.reason)
            return Deadline(fused.signal, lambda: None)

    for sig in signals:

        def _make_handler(source: AbortSignal) -> Callable[[], None]:
            def _on_abort() -> None:
                fused.abort(source.reason)
                _cleanup()

            return _on_abort

        unsubs.append(sig.on_abort(_make_handler(sig)))

    return Deadline(fused.signal, _cleanup)


def timeout_of(
    carrier: Any,
    code: str | None = None,
) -> TimeoutReason | None:
    """Extract a matching TimeoutReason from an AbortSignal, if present."""
    if not isinstance(carrier, AbortSignal):
        return None
    reason = carrier.reason
    if not isinstance(reason, TimeoutReason):
        return None
    if code is not None and reason.code != code:
        return None
    return reason
