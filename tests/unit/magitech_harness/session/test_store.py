"""Unit tests for Session and derive_messages projection."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.store import (
    Session,
    _lock_exclusive,
    _normalize_adjacent_messages,
    _unlock,
    derive_messages,
)


def test_derive_messages_ignores_unknown_and_handles_roles() -> None:
    """Verifies that derive_messages correctly projects user, assistant, and tool messages."""
    events = [
        SessionEvent(id="1", session_id="s", type=EventType.TURN_START, payload={}),
        SessionEvent(
            id="2",
            session_id="s",
            type=EventType.USER_MESSAGE,
            payload={"text": "hello"},
        ),
        SessionEvent(
            id="3",
            session_id="s",
            type=EventType.ASSISTANT_MESSAGE,
            payload={
                "text": "I can help",
                "tool_calls": [{"call_id": "c1", "name": "test", "arguments": {}}],
            },
        ),
        SessionEvent(
            id="4",
            session_id="s",
            type=EventType.TOOL_RESULT,
            payload={"call_id": "c1", "name": "test", "content": "result_123"},
        ),
    ]

    msgs = derive_messages(events)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "I can help"
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "c1"
    assert msgs[2]["content"] == "result_123"


def test_session_append_and_load(tmp_path: Path) -> None:
    """Verifies Session persistence, sequential numbering, and loading from JSONL file."""
    file_path = tmp_path / "sess_save.jsonl"
    session = Session(session_id="sess_custom", storage_path=file_path)

    event1 = SessionEvent(
        id="ev_1",
        session_id="sess_custom",
        type=EventType.USER_MESSAGE,
        payload={"text": "persisted text 1"},
    )
    event2 = SessionEvent(
        id="ev_2",
        session_id="sess_custom",
        type=EventType.USER_MESSAGE,
        payload={"text": "persisted text 2"},
        seq=10,  # Explicit seq
    )
    session.append(event1)
    session.append(event2)
    assert event1.seq == 1
    assert event2.seq == 10
    assert event1.schema_version == 1
    assert file_path.exists()

    # Append blank line to verify line.strip() branch
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n\n")

    # Load from disk
    loaded_session = Session.load(file_path)
    assert loaded_session.session_id == "sess_custom"
    assert len(loaded_session.events) == 2
    assert loaded_session.events[0].payload == {"text": "persisted text 1"}
    assert loaded_session.events[0].seq == 1
    assert loaded_session.events[1].seq == 10


def test_session_load_repair_on_open(tmp_path: Path) -> None:
    """Verifies Session.load skips corrupt/torn lines when repair_on_open=True."""
    file_path = tmp_path / "sess_torn.jsonl"
    valid_ev = SessionEvent(
        id="1",
        session_id="torn_sess",
        type=EventType.USER_MESSAGE,
        payload={"text": "ok"},
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(valid_ev.model_dump_json() + "\n")
        f.write('{"id": "2", "session_id": "torn_sess", "incomp\n')  # Torn write

    # Default repair_on_open=True should load valid line and ignore corrupt line
    sess = Session.load(file_path, repair_on_open=True)
    assert len(sess.events) == 1
    assert sess.events[0].id == "1"

    # repair_on_open=False should raise JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        Session.load(file_path, repair_on_open=False)


def test_session_append_fcntl_fallback(tmp_path: Path) -> None:
    """Verifies Session.append behaves gracefully when fcntl is unavailable or throws."""
    file_path = tmp_path / "sess_nofcntl.jsonl"
    session = Session(session_id="sess_nofcntl", storage_path=file_path)

    with patch.dict("sys.modules", {"fcntl": None}):
        event = SessionEvent(
            id="ev_nofcntl",
            session_id="sess_nofcntl",
            type=EventType.USER_MESSAGE,
            payload={"text": "no fcntl"},
        )
        session.append(event)
        assert len(session.events) == 1


def _fake_msvcrt() -> SimpleNamespace:
    """A stub msvcrt recording locking calls (no Windows in the test env)."""
    return SimpleNamespace(locking=MagicMock(), LK_LOCK=1, LK_UNLCK=0)


def test_lock_exclusive_and_unlock_windows_use_msvcrt(tmp_path: Path) -> None:
    """Verifies the Windows locking path byte-locks offset 0 via msvcrt."""
    fake_msvcrt = _fake_msvcrt()
    file_path = tmp_path / "sess_win.jsonl"
    with (
        patch("sys.platform", "win32"),
        patch.dict("sys.modules", {"msvcrt": fake_msvcrt}),
        open(file_path, "a", encoding="utf-8") as f,
    ):
        _lock_exclusive(f)
        fake_msvcrt.locking.assert_called_once_with(f.fileno(), 1, 1)
        fake_msvcrt.locking.reset_mock()
        _unlock(f)
        fake_msvcrt.locking.assert_called_once_with(f.fileno(), 0, 1)


def test_session_append_windows_locking(tmp_path: Path) -> None:
    """Verifies Session.append locks, writes, and unlocks via msvcrt on Windows."""
    fake_msvcrt = _fake_msvcrt()
    file_path = tmp_path / "sess_win_append.jsonl"
    session = Session(session_id="sess_win", storage_path=file_path)
    with (
        patch("sys.platform", "win32"),
        patch.dict("sys.modules", {"msvcrt": fake_msvcrt}),
    ):
        session.append(
            SessionEvent(
                id="ev_win",
                session_id="sess_win",
                type=EventType.USER_MESSAGE,
                payload={"text": "windows lock"},
            )
        )
    assert len(session.events) == 1
    assert fake_msvcrt.locking.call_count == 2
    lock_call, unlock_call = fake_msvcrt.locking.call_args_list
    assert lock_call.args[1:] == (1, 1)
    assert unlock_call.args[1:] == (0, 1)
    assert "windows lock" in file_path.read_text(encoding="utf-8")


def _fake_fcntl() -> SimpleNamespace:
    """A stub fcntl recording flock calls (drives the POSIX branch on Windows)."""
    return SimpleNamespace(flock=MagicMock(), LOCK_EX=2, LOCK_UN=8)


def test_lock_exclusive_and_unlock_posix_use_fcntl(tmp_path: Path) -> None:
    """Verifies the POSIX locking path uses flock for lock and unlock."""
    fake_fcntl = _fake_fcntl()
    file_path = tmp_path / "sess_posix.jsonl"
    with (
        patch("sys.platform", "linux"),
        patch.dict("sys.modules", {"fcntl": fake_fcntl}),
        open(file_path, "a", encoding="utf-8") as f,
    ):
        _lock_exclusive(f)
        fake_fcntl.flock.assert_called_once_with(f.fileno(), 2)
        fake_fcntl.flock.reset_mock()
        _unlock(f)
        fake_fcntl.flock.assert_called_once_with(f.fileno(), 8)


def test_session_append_lock_failure_is_best_effort(tmp_path: Path) -> None:
    """Verifies Session.append still persists when locking raises."""
    fake_msvcrt = SimpleNamespace(
        locking=MagicMock(side_effect=OSError("lock failed")), LK_LOCK=1, LK_UNLCK=0
    )
    file_path = tmp_path / "sess_lock_fail.jsonl"
    session = Session(session_id="sess_lock_fail", storage_path=file_path)
    with (
        patch("sys.platform", "win32"),
        patch.dict("sys.modules", {"msvcrt": fake_msvcrt}),
    ):
        session.append(
            SessionEvent(
                id="ev_lock_fail",
                session_id="sess_lock_fail",
                type=EventType.USER_MESSAGE,
                payload={"text": "persisted despite lock failure"},
            )
        )
    assert len(session.events) == 1
    assert fake_msvcrt.locking.call_count == 2
    assert "persisted despite lock failure" in file_path.read_text(encoding="utf-8")


def test_session_load_nonexistent_or_empty(tmp_path: Path) -> None:
    """Verifies Session.load on nonexistent or empty file."""
    nonexistent = tmp_path / "nonexistent.jsonl"
    s1 = Session.load(nonexistent)
    assert s1.session_id == "nonexistent"
    assert s1.events == []

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    s2 = Session.load(empty_file)
    assert s2.session_id == "empty"
    assert s2.events == []


def test_derive_messages_with_compaction_and_adjacent_fold() -> None:
    """Verifies derive_messages folds prior history into compaction summary and merges consecutive user turns."""
    events = [
        SessionEvent(
            id="1",
            session_id="s",
            type=EventType.USER_MESSAGE,
            payload={"text": "Initial task"},
        ),
        SessionEvent(
            id="2",
            session_id="s",
            type=EventType.ASSISTANT_MESSAGE,
            payload={"text": "Step 1 done"},
        ),
        SessionEvent(
            id="3",
            session_id="s",
            type=EventType.COMPACTION,
            payload={"summary": "Initial task: Step 1 is done."},
        ),
        SessionEvent(
            id="4",
            session_id="s",
            type=EventType.USER_MESSAGE,
            payload={"text": "Now step 2"},
        ),
        SessionEvent(
            id="5",
            session_id="s",
            type=EventType.ASSISTANT_MESSAGE,
            payload={"text": "Step 2 done"},
        ),
    ]

    msgs = derive_messages(events)
    # The compaction checkpoint user message and "Now step 2" user message fold into 1 user message
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert (
        "[Conversation summary up to checkpoint]:\nInitial task: Step 1 is done.\n\nNow step 2"
        in msgs[0]["content"]
    )
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Step 2 done"


def test_derive_messages_with_compaction_and_compacted_count() -> None:
    """Verifies derive_messages uses compacted_count to keep non-compacted history."""
    events = [
        SessionEvent(
            id="1",
            session_id="s",
            type=EventType.USER_MESSAGE,
            payload={"text": "Old 1"},
        ),
        SessionEvent(
            id="2",
            session_id="s",
            type=EventType.ASSISTANT_MESSAGE,
            payload={"text": "Old 2"},
        ),
        SessionEvent(
            id="3",
            session_id="s",
            type=EventType.USER_MESSAGE,
            payload={"text": "Kept 3"},
        ),
        SessionEvent(
            id="4",
            session_id="s",
            type=EventType.COMPACTION,
            payload={"summary": "Summary of 1 & 2", "compacted_count": 2},
        ),
    ]
    msgs = derive_messages(events)
    # Compaction summary (user) + Kept 3 (user) fold together
    assert len(msgs) == 1
    assert (
        "[Conversation summary up to checkpoint]:\nSummary of 1 & 2\n\nKept 3"
        == msgs[0]["content"]
    )


def test_normalize_adjacent_messages_branches() -> None:
    """Verifies all branch edge cases in _normalize_adjacent_messages."""
    assert _normalize_adjacent_messages([]) == []

    # Single message
    assert len(_normalize_adjacent_messages([{"role": "user", "content": "hi"}])) == 1

    # Consecutive assistant messages without tool calls
    msgs_asst = [
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]
    norm_asst = _normalize_adjacent_messages(msgs_asst)
    assert len(norm_asst) == 1
    assert norm_asst[0]["content"] == "Hello\n\nWorld"

    # Consecutive assistant with None content
    msgs_none = [
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "Text"},
    ]
    norm_none = _normalize_adjacent_messages(msgs_none)
    assert len(norm_none) == 1
    assert norm_none[0]["content"] == "Text"

    # Assistant with tool calls should NOT merge
    msgs_tool_calls = [
        {"role": "assistant", "content": "Text", "tool_calls": [{"id": "1"}]},
        {"role": "assistant", "content": "Text 2"},
    ]
    assert len(_normalize_adjacent_messages(msgs_tool_calls)) == 2

    # Tool role messages should NOT merge
    msgs_tools = [
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]
    assert len(_normalize_adjacent_messages(msgs_tools)) == 2


def test_session_derive_messages_method() -> None:
    """Verifies session.derive_messages() forwards to derive_messages."""
    session = Session(session_id="test")
    session.append(
        SessionEvent(
            id="1",
            session_id="test",
            type=EventType.USER_MESSAGE,
            payload={"text": "hello"},
        )
    )
    msgs = session.derive_messages()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "hello"
