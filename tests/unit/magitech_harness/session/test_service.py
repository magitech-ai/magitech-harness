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


def test_session_service_load_list_and_delete(tmp_path: Path) -> None:
    """Verifies loading persisted sessions, listing session IDs, and deleting sessions."""
    ctx = Context()
    service = SessionService(ctx, base_storage_dir=tmp_path)

    # 1. Create and persist session 1
    s1 = service.create("sess_1")
    service.log_event(s1, EventType.USER_MESSAGE, {"text": "hello 1"})

    # 2. Persist session 2 directly to disk to test cold loading
    s2_file = tmp_path / "sess_2.jsonl"
    s2_file.write_text(
        '{"id":"e2","session_id":"sess_2","type":"user/message","timestamp":100.0,"payload":{"text":"cold"}}\n',
        encoding="utf-8",
    )

    # 3. List sessions: should include sess_1 (active) and sess_2 (on disk)
    sessions = service.list_sessions()
    assert sessions == ["sess_1", "sess_2"]

    # 4. Load s1 (from active cache)
    loaded_s1 = service.load("sess_1")
    assert loaded_s1 is s1

    # 5. Load s2 (cold load from disk)
    loaded_s2 = service.load("sess_2")
    assert loaded_s2 is not None
    assert loaded_s2.session_id == "sess_2"
    assert len(loaded_s2.events) == 1
    assert loaded_s2.events[0].payload == {"text": "cold"}

    # 6. Load nonexistent
    assert service.load("sess_missing") is None

    # Cold load with repair_on_open
    s_corrupt = tmp_path / "sess_corrupt.jsonl"
    s_corrupt.write_text('{"bad json\n', encoding="utf-8")
    assert service.load("sess_corrupt", repair_on_open=True) is not None

    # 7. Delete s1 (both active and on disk)
    assert service.delete("sess_1") is True
    assert service.get("sess_1") is None
    assert not (tmp_path / "sess_1.jsonl").exists()

    # 8. Delete nonexistent
    assert service.delete("sess_missing") is False


def test_session_service_load_and_list_without_base_dir() -> None:
    """Verifies load and list behavior when base_storage_dir is None."""
    ctx = Context()
    service = SessionService(ctx)
    s = service.create("in_memory")
    assert service.list_sessions() == ["in_memory"]
    assert service.load("in_memory") is s
    assert service.load("missing") is None
    assert service.delete("in_memory") is True
    assert service.list_sessions() == []
