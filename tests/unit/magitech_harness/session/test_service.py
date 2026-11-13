"""Unit tests for SessionService."""

import asyncio
import json
import shutil
import time
from pathlib import Path

import pytest
from anima import Context

from magitech_harness.session.events import EventType
from magitech_harness.session.format import (
    SessionFormatUnsupportedError,
    SessionPersistenceCorruptionError,
    session_format_log_filename,
)
from magitech_harness.session.persistence import (
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
)
from magitech_harness.session.persistence_jsonl import (
    find_session_dir,
    session_dir,
)
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session


def _v4_header_line(session_id: str, **extra: object) -> str:
    header: dict = {
        "type": "session",
        "version": 4,
        "id": session_id,
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    header.update(extra)
    return json.dumps(header)


def _v4_log(session_id: str, rows: list[dict]) -> str:
    lines = [_v4_header_line(session_id)]
    lines += [json.dumps(row) for row in rows]
    return "\n".join(lines) + "\n"


def _v3_log(session_id: str) -> str:
    return (
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session",
                        "version": 3,
                        "id": session_id,
                        "createdAt": 1000,
                        "isSeeded": False,
                        "delegationDepth": 0,
                    }
                ),
                json.dumps(
                    {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}}
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
        + "\n"
    )


def _stage_log(root: Path, session_id: str, content: str, version: int = 4) -> Path:
    """Write raw log bytes into the backend layout for one session id."""
    directory = session_dir(root, None, session_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / session_format_log_filename(version)
    path.write_text(content, encoding="utf-8")
    return path


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
    # Created but unmaterialized: no log file yet, but already listed.
    assert service.location("test_id") is None
    assert service.list_sessions() == ["test_id"]

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
    assert service.get(session.session_id) is session
    assert service.list_sessions() == [session.session_id]
    assert service.close(session.session_id) is True
    assert service.list_sessions() == []


def test_session_service_create_duplicate_raises(tmp_path: Path) -> None:
    """Verifies creating over a live id is refused before touching the backend."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    service.create("dup")
    with pytest.raises(ValueError, match="already exists"):
        service.create("dup")


@pytest.mark.asyncio
async def test_session_service_fork_in_memory() -> None:
    """Verifies fork works with no persistence backend configured."""
    ctx = Context()
    service = SessionService(ctx)
    parent = service.create("parent")
    service.log_event(parent, "turn/start", {"turn": 1})
    service.log_event(parent, "user/message", {"text": "hi"})
    child = service.fork("parent", child_session_id="child")
    assert child.inherited_event_count == 2
    assert service.get("child") is child
    assert service.list_sessions() == ["child", "parent"]


@pytest.mark.asyncio
async def test_session_service_fork_create_failure_rolls_back(
    tmp_path: Path,
) -> None:
    """Verifies a fork whose child id already exists on disk rolls back cleanly."""
    _stage_log(
        tmp_path,
        "child",
        _v4_log(
            "child",
            [
                {
                    "type": "user/message",
                    "seq": 0,
                    "time": 100000,
                    "data": {"text": "staged"},
                    "surfaceOp": "append",
                }
            ],
        ),
    )
    staged = find_session_dir(tmp_path, "child")
    assert staged is not None
    before = (staged / "session.v4.jsonl").read_bytes()
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    parent = service.create("parent")
    service.log_event(parent, "turn/start", {"turn": 1})
    await service.flush(parent)
    with pytest.raises(SessionAlreadyExistsError):
        service.fork("parent", child_session_id="child")
    assert service.get("child") is None
    assert (staged / "session.v4.jsonl").read_bytes() == before


@pytest.mark.asyncio
async def test_session_service_fork_append_failure_rolls_back(
    tmp_path: Path,
) -> None:
    """Verifies a fork whose first write fails leaves no store entry or directory.

    The child's generation path is a directory, so the seed append fails
    on open; the rollback closes the handle, removes the directory, and
    detaches the store entry, so a retry succeeds.
    """
    child_dir = session_dir(tmp_path, None, "child")
    (child_dir / "session.v4.jsonl").mkdir(parents=True)
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    parent = service.create("parent")
    service.log_event(parent, "turn/start", {"turn": 1})
    await service.flush(parent)
    with pytest.raises(OSError, match="Is a directory"):
        service.fork("parent", child_session_id="child")
    assert service.get("child") is None
    assert find_session_dir(tmp_path, "child") is None
    child = service.fork("parent", child_session_id="child")
    assert child.session_id == "child"


@pytest.mark.asyncio
async def test_session_service_fork_blocked_directory_rolls_back(
    tmp_path: Path,
) -> None:
    """Verifies a fork whose session directory path is a file rolls back.

    The blocking file is not a session directory, so nothing is removed;
    only the store entry is detached.
    """
    project_dir = session_dir(tmp_path, None, "child").parent
    project_dir.mkdir(parents=True, exist_ok=True)
    blocker = project_dir / "child"
    blocker.write_text("not a directory", encoding="utf-8")
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    parent = service.create("parent")
    service.log_event(parent, "turn/start", {"turn": 1})
    await service.flush(parent)
    with pytest.raises(OSError):
        service.fork("parent", child_session_id="child")
    assert service.get("child") is None
    assert blocker.read_text(encoding="utf-8") == "not a directory"


@pytest.mark.asyncio
async def test_session_service_load_store_only_session(tmp_path: Path) -> None:
    """Verifies load returns a session that is live in the store but has no handle."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.store.create("store-only")
    assert service.load("store-only") is session


@pytest.mark.asyncio
async def test_session_service_load_veto_rolls_back_handle(tmp_path: Path) -> None:
    """Verifies a vetoed load releases its write handle, so a retry succeeds."""
    _stage_log(
        tmp_path,
        "vetoed-load",
        _v4_log(
            "vetoed-load",
            [
                {
                    "type": "user/message",
                    "seq": 0,
                    "time": 100000,
                    "data": {"text": "cold"},
                    "surfaceOp": "append",
                }
            ],
        ),
    )
    ctx = Context()

    def veto(session) -> None:
        raise RuntimeError("no entry")

    dispose = ctx.on("session/created", veto)
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    with pytest.raises(RuntimeError, match="no entry"):
        service.load("vetoed-load")
    assert service.get("vetoed-load") is None
    dispose()
    loaded = service.load("vetoed-load")
    assert loaded is not None
    assert len(loaded.events) == 1


def test_session_service_read_store_only_and_without_backend(
    tmp_path: Path,
) -> None:
    """Verifies read returns a store-live session and None with no backend."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.store.create("read-only")
    assert service.read("read-only") is session
    assert service.read("missing") is None
    in_memory = SessionService(Context())
    assert in_memory.read("missing") is None


def test_session_service_log_event_broadcasts_detached_session() -> None:
    """Verifies log_event broadcasts for a detached session with no hook."""
    ctx = Context()
    events_broadcast: list = []
    ctx.on(
        "session/event",
        lambda sess, ev: events_broadcast.append((sess.session_id, ev.type)),
    )
    service = SessionService(ctx)
    session = Session.create("detached")
    ev = service.log_event(session, EventType.USER_MESSAGE, {"text": "hi"})
    assert ev.type == EventType.USER_MESSAGE
    assert events_broadcast == [("detached", EventType.USER_MESSAGE)]


@pytest.mark.asyncio
async def test_session_service_create_round_trip_through_backend(
    tmp_path: Path,
) -> None:
    """Verifies create/log/flush writes a v4 log a fresh service reloads."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("roundtrip")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "first"})
    service.log_event(session, EventType.USER_MESSAGE, {"text": "second"})
    await service.flush(session)

    location = service.location("roundtrip")
    assert location is not None
    assert location.exists()
    lines = location.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    header = json.loads(lines[0])
    assert header["type"] == "session"
    assert header["version"] == 4
    assert header["id"] == "roundtrip"
    assert header["isSeeded"] is False
    assert header["delegationDepth"] == 0
    assert isinstance(header["createdAt"], int)

    assert service.close("roundtrip") is True
    ctx2 = Context()
    service2 = SessionService(ctx2, {"base_storage_dir": tmp_path})
    reloaded = service2.load("roundtrip")
    assert reloaded is not None
    assert reloaded.session_id == "roundtrip"
    assert [event.payload for event in reloaded.events] == [
        {"text": "first"},
        {"text": "second"},
    ]
    assert [event.seq for event in reloaded.events] == [0, 1]


@pytest.mark.asyncio
async def test_session_service_load_list_and_delete(tmp_path: Path) -> None:
    """Verifies loading persisted sessions, listing session IDs, and deleting sessions."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})

    # 1. Create and persist session 1
    s1 = service.create("sess_1")
    service.log_event(s1, EventType.USER_MESSAGE, {"text": "hello 1"})
    await service.flush(s1)

    # 2. Stage session 2 directly on disk to test cold loading
    _stage_log(
        tmp_path,
        "sess_2",
        _v4_log(
            "sess_2",
            [
                {
                    "type": "user/message",
                    "seq": 0,
                    "time": 100000,
                    "data": {"text": "cold"},
                    "surfaceOp": "append",
                }
            ],
        ),
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
    _stage_log(tmp_path, "sess_corrupt", '{"bad json\n')
    with pytest.raises(SessionPersistenceCorruptionError):
        service.load("sess_corrupt")

    # 7. Delete s1 (both active and on disk): close first, then delete.
    assert service.close("sess_1") is True
    assert service.delete("sess_1") is True
    assert service.get("sess_1") is None
    assert find_session_dir(tmp_path, "sess_1") is None

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


@pytest.mark.asyncio
async def test_session_service_create_over_historical_log_migrates_on_load(
    tmp_path: Path,
) -> None:
    """Verifies create over an existing log is refused, while load migrates it.

    Creating over a v3 log raises AlreadyExists and leaves the file
    byte-identical; loading the same id publishes the migration and the
    session accepts new events.
    """
    staged = _stage_log(tmp_path, "probe-bypass1", _v3_log("probe-bypass1"), version=3)
    before = staged.read_bytes()
    ctx = Context()
    broadcast = []
    ctx.on("session/event", lambda sess, ev: broadcast.append(ev.type))
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    with pytest.raises(SessionAlreadyExistsError):
        service.create("probe-bypass1")
    assert staged.read_bytes() == before
    assert broadcast == []

    loaded = service.load("probe-bypass1")
    assert loaded is not None
    assert len(loaded.events) == 2
    service.log_event(loaded, EventType.USER_MESSAGE, {"text": "after"})
    await service.flush(loaded)
    assert service.close("probe-bypass1") is True
    directory = find_session_dir(tmp_path, "probe-bypass1")
    assert directory is not None
    assert not (directory / "session.v3.jsonl").exists()
    assert (directory / "session.v4.jsonl").exists()

    ctx2 = Context()
    service2 = SessionService(ctx2, {"base_storage_dir": tmp_path})
    reloaded = service2.load("probe-bypass1")
    assert reloaded is not None
    assert len(reloaded.events) == 3
    assert reloaded.events[2].payload == {"text": "after"}


@pytest.mark.asyncio
async def test_session_service_load_refuses_mixed_session_id(
    tmp_path: Path,
) -> None:
    """Verifies a log whose header names another id is corruption, never adopted."""
    _stage_log(
        tmp_path,
        "requested",
        _v4_log(
            "other",
            [
                {
                    "type": "user/message",
                    "seq": 0,
                    "time": 100000,
                    "data": {"text": "cold"},
                    "surfaceOp": "append",
                }
            ],
        ),
    )
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    with pytest.raises(SessionPersistenceCorruptionError, match="does not match"):
        service.load("requested")
    assert service.load("other") is None


@pytest.mark.asyncio
async def test_session_service_load_refuses_newer_and_malformed_logs(
    tmp_path: Path,
) -> None:
    """Verifies load fails loud on a newer-version log and a corrupt header."""
    _stage_log(tmp_path, "future", _v4_header_line("future", version=5) + "\n")
    _stage_log(tmp_path, "bad_header", '{"bad json\n')
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    with pytest.raises(SessionFormatUnsupportedError, match="uses log format v5"):
        service.load("future")
    with pytest.raises(
        SessionPersistenceCorruptionError, match="corrupt session header"
    ):
        service.load("bad_header")


@pytest.mark.asyncio
async def test_session_service_load_empty_generation_is_corruption(
    tmp_path: Path,
) -> None:
    """Verifies an empty generation file is corruption, not an empty session."""
    _stage_log(tmp_path, "empty", "")
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    with pytest.raises(
        SessionPersistenceCorruptionError, match="first line is not a session header"
    ):
        service.load("empty")


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


@pytest.mark.asyncio
async def test_session_service_delete_live_session_refuses_then_succeeds(
    tmp_path: Path,
) -> None:
    """Verifies deleting a live session is refused until it is closed."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("deletable")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "hello"})
    await service.flush(session)
    assert find_session_dir(tmp_path, "deletable") is not None
    with pytest.raises(SessionAlreadyOwnedError):
        service.delete("deletable")
    assert service.close("deletable") is True
    assert service.delete("deletable") is True
    assert service.get("deletable") is None
    assert find_session_dir(tmp_path, "deletable") is None
    assert service.delete("deletable") is False


@pytest.mark.asyncio
async def test_session_service_delete_cold_directory(tmp_path: Path) -> None:
    """Verifies deleting a session that was never opened removes its directory."""
    _stage_log(
        tmp_path,
        "cold-del",
        _v4_log(
            "cold-del",
            [
                {
                    "type": "user/message",
                    "seq": 0,
                    "time": 100000,
                    "data": {"text": "cold"},
                    "surfaceOp": "append",
                }
            ],
        ),
    )
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    assert service.delete("cold-del") is True
    assert find_session_dir(tmp_path, "cold-del") is None


@pytest.mark.asyncio
async def test_session_service_log_event_broadcasts_independent_session(
    tmp_path: Path,
) -> None:
    """Verifies log_event broadcasts for a cold-loaded session.

    The first service closes the session, releasing its write handle,
    so a second service can cold-load the same log; the loaded session
    is store-live, so log_event broadcasts through its hook exactly once.
    """
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
    await service.flush(created)
    assert service.close("cold") is True

    # A fresh service cold-loads the same log: the loaded session is
    # store-live, so log_event broadcasts it through its hook.
    service2 = SessionService(ctx, {"base_storage_dir": tmp_path})
    loaded = service2.load("cold")
    assert loaded is not None
    assert len(loaded.events) == 1
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


@pytest.mark.asyncio
async def test_session_service_live_tracker_routes_event_within_window(
    tmp_path: Path,
) -> None:
    """Verifies the live tracker routes a session/event to the file
    within the batching window, without waiting for an explicit flush."""
    ctx = Context()
    service = SessionService(
        ctx, {"base_storage_dir": tmp_path, "live_batch_delay": 0.01}
    )
    session = service.create("live")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "batched text"})
    deadline = time.monotonic() + 2.0
    found = False
    while time.monotonic() < deadline:
        location = service.location("live")
        if location is not None and "batched text" in location.read_text(
            encoding="utf-8"
        ):
            found = True
            break
        await asyncio.sleep(0.01)
    assert found


@pytest.mark.asyncio
async def test_session_service_flush_is_barrier_naming_failed_sessions(
    tmp_path: Path,
) -> None:
    """Verifies the service-wide flush aggregates failures naming each
    failed session, while the other sessions still flush."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    good = service.create("good")
    broken = service.create("broken")
    service.log_event(good, EventType.USER_MESSAGE, {"text": "ok"})
    service.log_event(broken, EventType.USER_MESSAGE, {"text": "doomed"})
    await service.flush()
    good_location = service.location("good")
    assert good_location is not None and good_location.exists()

    broken_dir = find_session_dir(tmp_path, "broken")
    assert broken_dir is not None
    shutil.rmtree(broken_dir)
    with pytest.raises(ExceptionGroup) as exc_info:
        await service.flush()
    assert "broken" in str(exc_info.value)
    # The healthy session still flushed.
    assert good_location.exists()
    await service.flush(good)
    assert service.close("good") is True


@pytest.mark.asyncio
async def test_session_service_second_writer_is_refused(
    tmp_path: Path,
) -> None:
    """Verifies a second service cannot write-open a session the first
    still holds; closing the first releases it."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("shared")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "held"})
    await service.flush(session)

    ctx2 = Context()
    service2 = SessionService(ctx2, {"base_storage_dir": tmp_path})
    with pytest.raises(SessionAlreadyOwnedError):
        service2.load("shared")
    assert service.close("shared") is True
    loaded = service2.load("shared")
    assert loaded is not None
    assert len(loaded.events) == 1
    assert loaded.events[0].payload == {"text": "held"}


@pytest.mark.asyncio
async def test_session_service_close_releases_and_reload(
    tmp_path: Path,
) -> None:
    """Verifies close detaches and releases, and the session reloads after."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("reopen")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "once"})
    await service.flush(session)
    assert service.close("reopen") is True
    assert service.close("reopen") is False
    assert service.get("reopen") is None
    reloaded = service.load("reopen")
    assert reloaded is not None
    assert len(reloaded.events) == 1
    assert reloaded.events[0].payload == {"text": "once"}


def test_session_service_create_veto_leaves_no_trace(tmp_path: Path) -> None:
    """Verifies a vetoed session/created leaves no store entry or log behind."""
    ctx = Context()

    def veto(session) -> None:
        raise RuntimeError("no creation")

    dispose = ctx.on("session/created", veto)
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    with pytest.raises(RuntimeError, match="no creation"):
        service.create("vetoed")
    assert service.list_sessions() == []
    dispose()
    session = service.create("vetoed")
    assert session.session_id == "vetoed"
    assert service.list_sessions() == ["vetoed"]


@pytest.mark.asyncio
async def test_session_service_read_inspects_without_owning(
    tmp_path: Path,
) -> None:
    """Verifies read inspects a stored log without claiming it or
    publishing a migration."""
    _stage_log(tmp_path, "peek3", _v3_log("peek3"), version=3)
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    assert service.read("missing") is None
    peeked = service.read("peek3")
    assert peeked is not None
    assert len(peeked.events) == 2
    # No migration was published by the read.
    directory = find_session_dir(tmp_path, "peek3")
    assert directory is not None
    assert (directory / "session.v3.jsonl").exists()
    assert not (directory / "session.v4.jsonl").exists()
    # No write claim was taken: a live load still succeeds.
    loaded = service.load("peek3")
    assert loaded is not None
    assert len(loaded.events) == 2


@pytest.mark.asyncio
async def test_session_service_load_append_continues_seq(tmp_path: Path) -> None:
    """Verifies append to a loaded v4 log derives seq from the stored log."""
    _stage_log(
        tmp_path,
        "adopted",
        _v4_log(
            "adopted",
            [
                {
                    "type": "user/message",
                    "seq": 0,
                    "time": 100000,
                    "data": {"text": "a"},
                    "surfaceOp": "append",
                },
                {
                    "type": "user/message",
                    "seq": 1,
                    "time": 100001,
                    "data": {"text": "b"},
                    "surfaceOp": "append",
                },
            ],
        ),
    )
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.load("adopted")
    assert session is not None
    event = service.log_event(session, "user/message", {"text": "c"})
    assert event.seq == 2
    await service.flush(session)
    assert service.close("adopted") is True
    ctx2 = Context()
    service2 = SessionService(ctx2, {"base_storage_dir": tmp_path})
    reloaded = service2.load("adopted")
    assert reloaded is not None
    assert [event.seq for event in reloaded.events] == [0, 1, 2]
    assert reloaded.events[2].payload == {"text": "c"}


@pytest.mark.asyncio
async def test_session_service_load_preserves_surface_op(tmp_path: Path) -> None:
    """Verifies surfaceOp and sourceEventSeqs survive a backend round-trip."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("surface")
    service.log_event(session, "user/message", {"text": "a"})
    service.log_event(
        session,
        "user/message",
        {"text": "b"},
        surface_op={"op": "replace", "startSeq": 0, "endSeq": 0},
        source_event_seqs=[0],
    )
    await service.flush(session)
    assert service.close("surface") is True
    ctx2 = Context()
    service2 = SessionService(ctx2, {"base_storage_dir": tmp_path})
    reloaded = service2.load("surface")
    assert reloaded is not None
    assert reloaded.events[0].surface_op == "append"
    assert reloaded.events[1].surface_op == {
        "op": "replace",
        "startSeq": 0,
        "endSeq": 0,
    }
    assert reloaded.events[1].source_event_seqs == [0]
    assert reloaded.surface_nodes() == [1]


@pytest.mark.asyncio
async def test_session_service_fork_persists_inherited_prefix(
    tmp_path: Path,
) -> None:
    """Verifies a forked child's inherited prefix is durable before its
    first live event, so the log stays contiguous across the fork."""
    ctx = Context()
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    parent = service.create("parent")
    service.log_event(parent, "turn/start", {"turn": 1})
    service.log_event(parent, "user/message", {"text": "hi"})
    await service.flush(parent)
    child = service.fork("parent", child_session_id="child")
    assert child.inherited_event_count == 2
    service.log_event(child, "user/message", {"text": "child first"})
    await service.flush(child)
    assert service.close("child") is True
    assert service.close("parent") is True
    ctx2 = Context()
    service2 = SessionService(ctx2, {"base_storage_dir": tmp_path})
    reloaded = service2.load("child")
    assert reloaded is not None
    assert [event.type for event in reloaded.events] == [
        event.type for event in child.events
    ]
    assert [event.payload for event in reloaded.events] == [
        event.payload for event in child.events
    ]
    assert reloaded.inherited_event_count == 2
