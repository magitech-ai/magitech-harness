"""Unit tests for SessionService."""

from pathlib import Path

from anima import Context

from magitech_harness.session.events import EventType
from magitech_harness.session.service import SessionService


def test_session_service_create_get_and_log(tmp_path: Path) -> None:
    """Verifies SessionService lifecycle, event logging, and session lookup."""
    ctx = Context()
    events_broadcast = []
    ctx.on(
        "session/event",
        lambda sess, ev: events_broadcast.append((sess.session_id, ev.type)),
    )

    service = SessionService(ctx, base_storage_dir=tmp_path)
    session = service.create("test_id")
    assert session.session_id == "test_id"
    assert session.storage_path == tmp_path / "test_id.jsonl"

    # Lookup
    found = service.get("test_id")
    assert found is session
    assert service.get("unknown_id") is None

    # Log event
    ev = service.log_event(session, EventType.USER_MESSAGE, {"text": "hello"})
    assert ev.type == EventType.USER_MESSAGE
    assert len(events_broadcast) == 1
    assert events_broadcast[0] == ("test_id", EventType.USER_MESSAGE)


def test_session_service_create_without_base_dir() -> None:
    """Verifies creating in-memory sessions without base_storage_dir."""
    ctx = Context()
    service = SessionService(ctx)
    session = service.create()
    assert session.session_id is not None
    assert session.storage_path is None
