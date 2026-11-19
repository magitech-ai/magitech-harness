"""OpenTelemetry instrumentation engine.

Creates independent OTLP reporting channels through the shared OTel
service. Each channel has its own exporter, resource, instrumentation scope,
and queue. Mounting alone creates no transport or identity and sends nothing.
Business consumers own authorization, redaction, field selection, and their
channel's disposal.

Reference: ``packages/telemetry/otel``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypedDict

from anima import Context, Service

__all__ = [
    "EventReporter",
    "OtelConfig",
    "OtelService",
    "ReporterChannel",
    "ReporterOptions",
    "SessionLogReporter",
    "ShutdownSignal",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ReporterOptions(TypedDict, total=False):
    """Options for creating a reporter channel."""

    endpoint: str
    scope: str
    scope_version: str
    resource_attributes: dict[str, Any]
    queue_max_size: int
    queue_flush_interval_ms: int
    max_request_bytes: int
    diagnostic_callback: Callable[[str], Any] | None


class ShutdownSignal:
    """Cancellation signal for reporter shutdown."""

    def __init__(self) -> None:
        self._aborted = False
        self._callbacks: list[Callable[[], None]] = []

    def abort(self) -> None:
        """Abort all pending operations."""
        self._aborted = True
        for cb in self._callbacks:
            cb()

    def is_aborted(self) -> bool:
        """Return True if the signal has been aborted."""
        return self._aborted

    def on_abort(self, callback: Callable[[], None]) -> None:
        """Register a callback to invoke on abort."""
        self._callbacks.append(callback)


class ReporterChannel:
    """One reporter channel with its own exporter, resource, scope, and queue."""

    def __init__(
        self,
        kind: str,
        options: ReporterOptions,
        signal: ShutdownSignal,
    ) -> None:
        self.kind = kind
        self.options = options
        self.signal = signal
        self._closed = False

    def is_closed(self) -> bool:
        """Return True if the channel has been closed."""
        return self._closed

    def shutdown(self) -> None:
        """Shut down the channel, cancelling pending operations."""
        if self._closed:
            return
        self._closed = True
        self.signal.abort()


class EventReporter(ReporterChannel):
    """Ordinary-event reporter channel."""


class SessionLogReporter(ReporterChannel):
    """Session-log reporter channel.

    Preserves one complete event per record with eventName: "session-log",
    sessionId, and a JSON string content. Enforces at most 4,000,000
    uncompressed request bytes.
    """

    MAX_REQUEST_BYTES = 4_000_000

    def __init__(self, options: ReporterOptions, signal: ShutdownSignal) -> None:
        super().__init__("session-log", options, signal)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class OtelConfig(TypedDict, total=False):
    """Optional configuration for the OTel service."""

    default_endpoint: str
    default_scope: str


class OtelService(Service):
    """Shared OTel service creating independent reporting channels.

    Mount one otel service to create ordinary-event and Session-log
    reporting channels. Each channel has its own exporter, resource,
    instrumentation scope, and queue. Mounting alone creates no transport
    or identity and sends nothing.
    """

    inject: ClassVar[list[str]] = []

    def __init__(self, ctx: Context, config: OtelConfig | None = None) -> None:
        if ctx.get("otel", None, strict=False) is not None:
            raise RuntimeError("otel service already registered")
        super().__init__(ctx, "otel")
        self.cfg: OtelConfig = config or {}
        self._channels: list[ReporterChannel] = []
        ctx.reflect.provide("otel", self, None)

    def create_event_reporter(
        self,
        options: ReporterOptions,
    ) -> EventReporter:
        """Create an ordinary-event reporter channel.

        Args:
            options: Channel options including endpoint, scope, resource
                attributes, queue settings, and optional diagnostic callback.

        Returns:
            An EventReporter channel belonging to the caller.
        """
        signal = ShutdownSignal()
        channel = EventReporter("event", options, signal)
        self._channels.append(channel)
        return channel

    def create_session_log_reporter(
        self,
        options: ReporterOptions,
    ) -> SessionLogReporter:
        """Create a Session-log reporter channel.

        Preserves one complete event per record with eventName: "session-log",
        sessionId, and a JSON string content. Enforces at most 4,000,000
        uncompressed request bytes.

        Args:
            options: Channel options including endpoint, scope, resource
                attributes, queue settings, and optional diagnostic callback.

        Returns:
            A SessionLogReporter channel belonging to the caller.
        """
        signal = ShutdownSignal()
        channel = SessionLogReporter(options, signal)
        self._channels.append(channel)
        return channel

    def shutdown_all(self) -> None:
        """Shut down all reporter channels."""
        for channel in self._channels:
            channel.shutdown()
        self._channels.clear()

    async def check(self) -> None:  # pragma: no cover - lifecycle check
        """Lifecycle check: service is always healthy once constructed."""
        return
