"""OpenTelemetry instrumentation service (ctx.otel).

Creates independent OTLP reporting channels through the shared OTel
service. Each channel has its own exporter, resource, instrumentation scope,
and queue. Mounting alone creates no transport or identity and sends nothing.
Business consumers own authorization, redaction, field selection, and their
channel's disposal.

Reference: ``packages/telemetry/otel``.
"""

from __future__ import annotations

from magitech_harness.telemetry.engine import (
    EventReporter,
    OtelConfig,
    OtelService,
    ReporterChannel,
    ReporterOptions,
    SessionLogReporter,
    ShutdownSignal,
)

__all__ = [
    "EventReporter",
    "OtelConfig",
    "OtelService",
    "ReporterChannel",
    "ReporterOptions",
    "SessionLogReporter",
    "ShutdownSignal",
]
