"""Unit tests for telemetry engine."""

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


def test_shutdown_signal_abort() -> None:
    """ShutdownSignal.abort() sets aborted state and invokes callbacks."""
    signal = ShutdownSignal()
    assert signal.is_aborted() is False

    signal.abort()
    assert signal.is_aborted() is True


def test_shutdown_signal_on_abort() -> None:
    """ShutdownSignal.on_abort() callbacks are invoked on abort."""
    signal = ShutdownSignal()
    called = []

    def cb() -> None:
        called.append(True)

    signal.on_abort(cb)
    signal.abort()
    assert called == [True]


def test_reporter_channel_shutdown() -> None:
    """ReporterChannel.shutdown() marks channel as closed and aborts signal."""
    signal = ShutdownSignal()
    channel = ReporterChannel("test", {"endpoint": "http://example.com"}, signal)
    assert channel.is_closed() is False
    assert signal.is_aborted() is False

    channel.shutdown()
    assert channel.is_closed() is True
    assert signal.is_aborted() is True


def test_reporter_channel_double_shutdown() -> None:
    """ReporterChannel.shutdown() is idempotent."""
    signal = ShutdownSignal()
    channel = ReporterChannel("test", {"endpoint": "http://example.com"}, signal)
    channel.shutdown()
    channel.shutdown()  # Should not raise
    assert channel.is_closed() is True


def test_event_reporter() -> None:
    """EventReporter is a ReporterChannel with kind 'event'."""
    signal = ShutdownSignal()
    reporter = EventReporter("event", {"endpoint": "http://example.com"}, signal)
    assert reporter.kind == "event"
    assert isinstance(reporter, ReporterChannel)


def test_session_log_reporter() -> None:
    """SessionLogReporter is a ReporterChannel with kind 'session-log'."""
    signal = ShutdownSignal()
    reporter = SessionLogReporter({"endpoint": "http://example.com"}, signal)
    assert reporter.kind == "session-log"
    assert isinstance(reporter, ReporterChannel)
    assert SessionLogReporter.MAX_REQUEST_BYTES == 4_000_000


def test_otel_service_creation() -> None:
    """OtelService registers itself as ctx.otel."""
    from animas import Context

    ctx = Context()
    service = OtelService(ctx)
    assert ctx.get("otel", None, strict=False) is service


def test_otel_service_duplicate_raises() -> None:
    """Creating OtelService twice raises RuntimeError."""
    from animas import Context

    ctx = Context()
    OtelService(ctx)
    try:
        OtelService(ctx)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_otel_create_event_reporter() -> None:
    """OtelService.create_event_reporter() creates an EventReporter channel."""
    from animas import Context

    ctx = Context()
    service = OtelService(ctx)
    reporter = service.create_event_reporter(
        {
            "endpoint": "http://localhost:4318",
            "scope": "test.scope",
            "scope_version": "1.0.0",
            "resource_attributes": {"service.name": "test"},
        }
    )
    assert isinstance(reporter, EventReporter)
    assert reporter.kind == "event"
    assert reporter.options["endpoint"] == "http://localhost:4318"


def test_otel_create_session_log_reporter() -> None:
    """OtelService.create_session_log_reporter() creates a SessionLogReporter channel."""
    from animas import Context

    ctx = Context()
    service = OtelService(ctx)
    reporter = service.create_session_log_reporter(
        {
            "endpoint": "http://localhost:4318",
            "scope": "test.session.scope",
            "scope_version": "1.0.0",
            "resource_attributes": {"service.name": "test"},
        }
    )
    assert isinstance(reporter, SessionLogReporter)
    assert reporter.kind == "session-log"
    assert reporter.options["endpoint"] == "http://localhost:4318"


def test_otel_shutdown_all() -> None:
    """OtelService.shutdown_all() shuts down all channels."""
    from animas import Context

    ctx = Context()
    service = OtelService(ctx)
    r1 = service.create_event_reporter({"endpoint": "http://example.com"})
    r2 = service.create_session_log_reporter({"endpoint": "http://example.com"})

    assert r1.is_closed() is False
    assert r2.is_closed() is False

    service.shutdown_all()

    assert r1.is_closed() is True
    assert r2.is_closed() is True


def test_otel_config_default() -> None:
    """OtelService with no config uses empty config."""
    from animas import Context

    ctx = Context()
    service = OtelService(ctx)
    assert service.cfg == {}


def test_otel_config_with_values() -> None:
    """OtelService with config uses provided values."""
    from animas import Context

    ctx = Context()
    service = OtelService(
        ctx, {"default_endpoint": "http://example.com", "default_scope": "test"}
    )
    assert service.cfg["default_endpoint"] == "http://example.com"
    assert service.cfg["default_scope"] == "test"


def test_reporter_options_typed_dict() -> None:
    """ReporterOptions accepts optional fields."""
    options: ReporterOptions = {
        "endpoint": "http://example.com",
        "scope": "test",
        "scope_version": "1.0.0",
        "resource_attributes": {"key": "value"},
        "queue_max_size": 1000,
        "queue_flush_interval_ms": 5000,
        "max_request_bytes": 1000000,
        "diagnostic_callback": None,
    }
    assert options["endpoint"] == "http://example.com"


def test_otel_config_typed_dict() -> None:
    """OtelConfig accepts optional fields."""
    config: OtelConfig = {
        "default_endpoint": "http://example.com",
        "default_scope": "test.scope",
    }
    assert config["default_endpoint"] == "http://example.com"
    assert config["default_scope"] == "test.scope"
