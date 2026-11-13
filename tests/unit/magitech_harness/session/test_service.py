"""Unit tests for SessionService."""

import json
from pathlib import Path

import pytest
from anima import Context

from magitech_harness.session.events import EventType
from magitech_harness.session.format import (
    SessionFormatUnsupportedError,
    SessionPersistenceCorruptionError,
)
from magitech_harness.session.service import SessionService


def test_session_service_create_get_and_log(tmp_path: Path) -> None:
    """Verifies SessionService lifecycle, event logging, and session lookup."""
    ctx = Context()
    events_broadcast = []
    ctx.on(
        "session/event",
        lambda sess, ev: events_broadcast.append((sess.session_id, ev.type)),
    )

    service = SessionService(ctx, {"base_storage_dir": tmp_path})
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
    service = SessionService(ctx, {"base_storage_dir": tmp_path})

    # 1. Create and persist session 1
    s1 = service.create("sess_1")
    service.log_event(s1, EventType.USER_MESSAGE, {"text": "hello 1"})

    # 2. Persist session 2 directly to disk to test cold loading
    s2_file = tmp_path / "sess_2.jsonl"
    s2_file.write_text(
        '{"type":"session","version":4,"id":"sess_2","createdAt":1000,"isSeeded":false,"delegationDepth":0}\n'
        '{"type":"user/message","seq":0,"time":100000,"data":{"text":"cold"},"surfaceOp":"append"}\n',
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

    # Cold load with a corrupt file refuses instead of repairing it.
    s_corrupt = tmp_path / "sess_corrupt.jsonl"
    s_corrupt.write_text('{"bad json\n', encoding="utf-8")
    with pytest.raises(SessionPersistenceCorruptionError):
        service.load("sess_corrupt")

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


def test_session_service_create_over_historical_file_refuses_log_event(
    tmp_path: Path,
) -> None:
    """Verifies create over an existing historical log refuses log_event.

    SessionService.create takes an explicit session id with no existence
    check, so creating over a v3 log used to append a current-version row
    to it and brick the next load. Create still succeeds, but the first
    log_event refuses, the file stays byte-identical, and nothing is
    broadcast.
    """
    file_path = tmp_path / "probe-bypass1.jsonl"
    file_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session",
                        "version": 3,
                        "id": "probe-bypass1",
                        "createdAt": 1000,
                        "isSeeded": False,
                        "delegationDepth": 0,
                    }
                ),
                json.dumps(
                    {
                        "type": "turn/start",
                        "seq": 0,
                        "time": 1000,
                        "data": {"turn": 1},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn/end",
                        "seq": 1,
                        "time": 1001,
                        "data": {"turn": 1, "reason": {"kind": "done"}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before = file_path.read_bytes()
    ctx = Context()
    broadcast = []
    ctx.on("session/event", lambda sess, ev: broadcast.append(ev.type))
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("probe-bypass1")
    with pytest.raises(
        SessionFormatUnsupportedError,
        match="cannot append to a migrated session until it is republished",
    ):
        service.log_event(session, EventType.USER_MESSAGE, {"text": "after"})
    assert session.events == []
    assert broadcast == []
    assert file_path.read_bytes() == before


def test_session_service_delete_store_only_session(tmp_path: Path) -> None:
    """Verifies deleting a session that lives in the store but not in
    the service's active cache still detaches it."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.store.create("store-only")
    assert service.get("store-only") is session
    assert service.delete("store-only") is True
    assert service.store.get("store-only") is None


def test_session_service_store_property() -> None:
    """Verifies the service exposes the store it shells over."""
    from magitech_harness.session.store import SessionStore

    ctx = Context()
    service = SessionService(ctx)
    assert isinstance(service.store, SessionStore)
    assert service.store.get("missing") is None


def test_session_service_delete_live_session(tmp_path: Path) -> None:
    """Verifies deleting a live session detaches it and removes its file."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("deletable")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "hello"})
    assert (tmp_path / "deletable.jsonl").exists()
    assert service.delete("deletable") is True
    assert service.get("deletable") is None
    assert not (tmp_path / "deletable.jsonl").exists()
    assert service.delete("deletable") is False


def test_session_service_log_event_broadcasts_independent_session(
    tmp_path: Path,
) -> None:
    """Verifies log_event broadcasts for an independent (cold-loaded)
    session, whose store publication hook is absent."""
    ctx = Context()
    events_broadcast: list = []
    ctx.on(
        "session/event",
        lambda sess, ev: events_broadcast.append((sess.session_id, ev.type)),
    )
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    created = service.create("cold")
    service.log_event(created, EventType.USER_MESSAGE, {"text": "first"})
    assert len(events_broadcast) == 1

    # A fresh service cold-loads the same log: the loaded session has no
    # publication hook, so log_event must broadcast it itself.
    service2 = SessionService(ctx, {"base_storage_dir": tmp_path})
    loaded = service2.load("cold")
    assert loaded is not None
    assert loaded._publish_hook is None
    events_broadcast.clear()
    service2.log_event(loaded, EventType.USER_MESSAGE, {"text": "second"})
    assert len(events_broadcast) == 1
    assert events_broadcast[0] == ("cold", EventType.USER_MESSAGE)


def test_session_service_log_event_store_owned_broadcasts_once(
    tmp_path: Path,
) -> None:
    """Verifies a store-owned session's event is broadcast exactly once:
    through the store hook, not again by the service."""
    ctx = Context()
    events_broadcast: list = []
    ctx.on(
        "session/event",
        lambda sess, ev: events_broadcast.append((sess.session_id, ev.type)),
    )
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("owned")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "hello"})
    assert events_broadcast == [("owned", EventType.USER_MESSAGE)]
