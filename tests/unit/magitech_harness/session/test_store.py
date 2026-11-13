"""Unit tests for Session and derive_messages projection."""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from anima import Context

from magitech_harness.session.events import (
    KNOWN_SESSION_EVENT_TYPES,
    SURFACE_EVENT_TYPES,
    EventType,
    SessionEvent,
)
from magitech_harness.session.format import (
    SessionFormatError,
    SessionFormatUnsupportedError,
    SessionPersistenceCorruptionError,
)
from magitech_harness.session.store import (
    Session,
    SessionForkError,
    SessionPreparation,
    SessionStore,
    _lock_exclusive,
    _normalize_adjacent_messages,
    _to_session_event,
    _unlock,
    append_plugin_record,
    derive_messages,
    plugin_record_of,
)
from magitech_harness.session.surface import SurfaceError, fold_surface


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

    event1 = session.append(
        EventType.USER_MESSAGE,
        {"text": "persisted text 1"},
        surface_op="append",
    )
    event2 = session.append(
        EventType.USER_MESSAGE,
        {"text": "persisted text 2"},
        surface_op="append",
    )
    assert event1.seq == 0
    assert event2.seq == 1
    assert file_path.exists()

    # The first line is the v4 physical header, not an event row.
    lines = file_path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["type"] == "session"
    assert header["version"] == 4
    assert header["id"] == "sess_custom"
    assert header["isSeeded"] is False
    assert header["delegationDepth"] == 0
    assert isinstance(header["createdAt"], int)

    # Append blank line to verify line.strip() branch
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n\n")

    # Load from disk
    loaded_session = Session.load(file_path)
    assert loaded_session.session_id == "sess_custom"
    assert len(loaded_session.events) == 2
    # Rows carry no event id; the id is synthesized from session and seq.
    assert loaded_session.events[0].id == "sess_custom:0"
    assert loaded_session.events[1].id == "sess_custom:1"
    assert loaded_session.events[0].payload == {"text": "persisted text 1"}
    assert loaded_session.events[0].seq == 0
    assert loaded_session.events[1].seq == 1


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


def test_session_load_refuses_bad_envelope_row(tmp_path: Path) -> None:
    """Verifies a v4 row that fails envelope checks is corruption, not a skip."""
    file_path = tmp_path / "sess_bad_row.jsonl"
    file_path.write_text(
        _v4_header_line("bad_row_sess")
        + "\n"
        + json.dumps({"type": "user/message", "seq": 1, "data": {"text": "no time"}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionPersistenceCorruptionError):
        Session.load(file_path)


def test_session_load_migrated_unsupported_is_unsupported(tmp_path: Path) -> None:
    """Verifies a historical file that cannot migrate is unsupported, not corrupt."""
    file_path = tmp_path / "sess_unmig.jsonl"
    file_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session",
                        "version": 0,
                        "id": "unmig",
                        "createdAt": 1000,
                        "delegationDepth": 0,
                    }
                ),
                json.dumps(
                    {"type": "mode/set", "seq": 0, "time": 1000, "data": {"mode": "x"}}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionFormatUnsupportedError, match="raw log"):
        Session.load(file_path)


def test_session_load_migrated_malformed_is_corruption(tmp_path: Path) -> None:
    """Verifies a historical file with a seq gap is corruption, not unsupported."""
    file_path = tmp_path / "sess_gap.jsonl"
    file_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session",
                        "version": 3,
                        "id": "gap",
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
                        "seq": 2,
                        "time": 1001,
                        "data": {"turn": 1, "reason": {"kind": "done"}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionPersistenceCorruptionError):
        Session.load(file_path)


def test_session_load_migrates_ignorable_unknown_as_skip(tmp_path: Path) -> None:
    """Verifies a migrated ignorable unknown row is skipped, not refused."""
    file_path = tmp_path / "mig_skip.jsonl"
    file_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session",
                        "version": 3,
                        "id": "mig_skip",
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
                        "type": "vendor/widget",
                        "seq": 1,
                        "time": 1001,
                        "data": {"x": 1},
                        "ignorable": True,
                    }
                ),
                json.dumps(
                    {
                        "type": "turn/end",
                        "seq": 2,
                        "time": 1002,
                        "data": {"turn": 1, "reason": {"kind": "done"}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = Session.load(file_path)
    assert [event.type for event in loaded.events] == [
        EventType.TURN_START,
        # The v3->v4 edge namespaces unknown ignorable events into plugin:.
        "plugin:vendor/widget",
        EventType.TURN_END,
    ]
    assert loaded.events[1].ignorable is True
    assert loaded.events[1].payload == {"x": 1}


def test_session_load_refuses_torn_row(tmp_path: Path) -> None:
    """Verifies Session.load refuses a torn row instead of silently truncating it."""
    file_path = tmp_path / "sess_torn.jsonl"
    file_path.write_text(
        _v4_header_line("torn_sess")
        + "\n"
        + json.dumps(
            {
                "type": "user/message",
                "seq": 1,
                "time": 100000,
                "data": {"text": "ok"},
            }
        )
        + "\n"
        + '{"type": "user/message", "seq": 2, "time": 100001, "data": {"text": "torn\n',
        encoding="utf-8",
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="corrupt session row"):
        Session.load(file_path)


def test_session_load_refuses_non_object_row(tmp_path: Path) -> None:
    """Verifies Session.load refuses a JSON row that is not an object."""
    file_path = tmp_path / "sess_array_row.jsonl"
    file_path.write_text(
        _v4_header_line("array_sess") + "\n" + "[]\n",
        encoding="utf-8",
    )
    with pytest.raises(
        SessionPersistenceCorruptionError, match="row must be a JSON object"
    ):
        Session.load(file_path)


def test_session_load_refuses_corrupt_header(tmp_path: Path) -> None:
    """Verifies Session.load refuses a header line that is not JSON."""
    file_path = tmp_path / "sess_bad_header.jsonl"
    file_path.write_text(
        '{"type": "session", "version": 4, "id": "bad\n', encoding="utf-8"
    )
    with pytest.raises(
        SessionPersistenceCorruptionError, match="corrupt session header"
    ):
        Session.load(file_path)


def test_session_load_refuses_newer_version_without_decoding_rows(
    tmp_path: Path,
) -> None:
    """Verifies a newer-version log is refused with the upgrade text, rows unread."""
    file_path = tmp_path / "sess_future.jsonl"
    file_path.write_text(
        _v4_header_line("future_sess", version=9)
        + "\n"
        + "this is not json at all\n"
        + '{"type": "user/message", "seq": 1}\n',
        encoding="utf-8",
    )
    with pytest.raises(SessionFormatUnsupportedError, match="upgrade the harness"):
        Session.load(file_path)


def test_session_load_refuses_bare_old_format_file(tmp_path: Path) -> None:
    """Verifies a headerless log (event row first) is malformed, not loadable."""
    file_path = tmp_path / "sess_bare.jsonl"
    file_path.write_text(
        json.dumps(
            {
                "type": "user/message",
                "seq": 1,
                "time": 100000,
                "data": {"text": "cold"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionFormatError, match="must be a non-negative safe integer"):
        Session.load(file_path)


def test_session_load_refuses_malformed_header(tmp_path: Path) -> None:
    """Verifies a header object without a version is malformed, not unsupported."""
    file_path = tmp_path / "sess_no_version.jsonl"
    file_path.write_text(
        json.dumps({"type": "session", "id": "no_version"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(SessionFormatError, match="must be a non-negative safe integer"):
        Session.load(file_path)


def test_session_load_migrates_v0_file(tmp_path: Path) -> None:
    """Verifies a v0 log loads through the migration chain as v4 events."""
    file_path = tmp_path / "mig_v0.jsonl"
    file_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session",
                        "version": 0,
                        "id": "mig_v0",
                        "createdAt": 1000,
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
                        "data": {"turn": 1, "reason": {"kind": "completed"}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = Session.load(file_path)
    assert loaded.session_id == "mig_v0"
    assert [event.type for event in loaded.events] == [
        EventType.TURN_START,
        EventType.TURN_END,
    ]
    assert loaded.events[0].payload == {"turn": 1}
    assert loaded.events[1].payload == {"turn": 1, "reason": {"kind": "completed"}}


def test_session_load_carries_ignorable_unknown_and_refuses_known_unknown(
    tmp_path: Path,
) -> None:
    """Verifies unknown ignorable rows are carried opaquely while unknown rows are refused."""
    file_path = tmp_path / "sess_unknown.jsonl"
    file_path.write_text(
        _v4_header_line("unknown_sess")
        + "\n"
        + json.dumps(
            {
                "type": "vendor/widget",
                "seq": 0,
                "time": 100000,
                "data": {"x": 1},
                "ignorable": True,
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "user/message",
                "seq": 1,
                "time": 100001,
                "data": {"text": "kept"},
                "surfaceOp": "append",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = Session.load(file_path)
    assert len(loaded.events) == 2
    assert loaded.events[0].type == "vendor/widget"
    assert loaded.events[0].ignorable is True
    assert loaded.events[0].payload == {"x": 1}
    assert loaded.events[1].payload == {"text": "kept"}

    file_path.write_text(
        _v4_header_line("unknown_sess")
        + "\n"
        + json.dumps(
            {"type": "vendor/widget", "seq": 0, "time": 100000, "data": {"x": 1}}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionFormatUnsupportedError, match="unknown event type"):
        Session.load(file_path)


def test_session_append_fcntl_fallback(tmp_path: Path) -> None:
    """Verifies Session.append behaves gracefully when fcntl is unavailable or throws."""
    file_path = tmp_path / "sess_nofcntl.jsonl"
    session = Session(session_id="sess_nofcntl", storage_path=file_path)

    with patch.dict("sys.modules", {"fcntl": None}):
        session.append(
            EventType.USER_MESSAGE, {"text": "no fcntl"}, surface_op="append"
        )
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
            EventType.USER_MESSAGE, {"text": "windows lock"}, surface_op="append"
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
            EventType.USER_MESSAGE,
            {"text": "persisted despite lock failure"},
            surface_op="append",
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
            type="session/compaction",
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
            type="session/compaction",
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
    session.append(EventType.USER_MESSAGE, {"text": "hello"}, surface_op="append")
    msgs = session.derive_messages()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "hello"


def _write_history_file(
    file_path: Path, version: int, session_id: str, rows: list[dict]
) -> None:
    """Write one historical log (physical header + source rows) as JSONL."""
    header: dict = {
        "type": "session",
        "version": version,
        "id": session_id,
        "createdAt": 1000,
        "delegationDepth": 0,
    }
    if version >= 2:
        header["isSeeded"] = False
    file_path.write_text(
        "\n".join([json.dumps(header)] + [json.dumps(row) for row in rows]) + "\n",
        encoding="utf-8",
    )


#: Source rows of the end-to-end fixtures in test_format.py
#: (test_e2e_v*_to_v4), paired with the migrated event types Session.load
#: must produce for each.
_E2E_HISTORY_CASES: list[tuple[int, str, list[dict], list[EventType]]] = [
    (
        0,
        "e2e-v0",
        [
            {
                "type": "turn/start",
                "seq": 0,
                "time": 1000,
                "data": {"turn": 1, "trigger": {"kind": "user"}},
            },
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "steering/message",
                "seq": 2,
                "time": 1002,
                "data": {
                    "turn": 1,
                    "content": [{"type": "text", "text": "steer"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "compact/start",
                "seq": 3,
                "time": 1003,
                "data": {"compactionId": "c1", "turn": 1},
            },
            {
                "type": "step/end",
                "seq": 4,
                "time": 1004,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 5,
                "time": 1005,
                "data": {"turn": 1, "reason": {"kind": "aborted"}},
            },
        ],
        [
            EventType.TURN_START,
            EventType.STEP_START,
            EventType.SYSTEM_MESSAGE,
            EventType.USER_MESSAGE,
            EventType.COMPACTION_START,
            EventType.STEP_END,
            EventType.TURN_END,
        ],
    ),
    (
        1,
        "e2e-v1",
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "user/message",
                "seq": 2,
                "time": 1002,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "assistant/chunk",
                "seq": 3,
                "time": 1003,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "chunk": {"type": "text-delta", "index": 0, "text": "Hello"},
                },
            },
            {
                "type": "assistant/chunk",
                "seq": 4,
                "time": 1004,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "chunk": {"type": "text-delta", "index": 0, "text": " world"},
                },
            },
            {
                "type": "assistant/message",
                "seq": 5,
                "time": 1005,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m2",
                        "role": "assistant",
                        "source": {"kind": "model"},
                        "content": [{"type": "text", "text": "Hello world"}],
                    },
                },
                "sourceEventSeqs": [3, 4],
                "surfaceOp": "append",
            },
            {
                "type": "step/end",
                "seq": 6,
                "time": 1006,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 7,
                "time": 1007,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
        [
            EventType.TURN_START,
            EventType.STEP_START,
            EventType.SYSTEM_MESSAGE,
            EventType.USER_MESSAGE,
            EventType.ASSISTANT_MESSAGE,
            EventType.STEP_END,
            EventType.TURN_END,
        ],
    ),
    (
        2,
        "e2e-v2",
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "request/header",
                "seq": 2,
                "time": 1002,
                "data": {
                    "header": {"system": "Be helpful."},
                    "reason": {"kind": "start"},
                },
            },
            {
                "type": "user/message",
                "seq": 3,
                "time": 1003,
                "data": {
                    "role": "user",
                    "id": "m1",
                    "content": [{"type": "text", "text": "hi"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "step/end",
                "seq": 4,
                "time": 1004,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 5,
                "time": 1005,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
        [
            EventType.TURN_START,
            EventType.STEP_START,
            EventType.SYSTEM_MESSAGE,
            EventType.SYSTEM_MESSAGE,
            EventType.REQUEST_HEADER,
            EventType.USER_MESSAGE,
            EventType.STEP_END,
            EventType.TURN_END,
        ],
    ),
    (
        3,
        "e2e-v3",
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "step/start",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "user/message",
                "seq": 2,
                "time": 1002,
                "data": {
                    "role": "user",
                    "id": "m-user-1",
                    "content": [{"type": "text", "text": "hello"}],
                    "source": {"kind": "user"},
                },
                "surfaceOp": "append",
            },
            {
                "type": "assistant/message",
                "seq": 3,
                "time": 1003,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m-asst-1",
                        "role": "assistant",
                        "source": {"kind": "model"},
                        "content": [{"type": "text", "text": "hi there"}],
                    },
                    "stream": [],
                },
                "surfaceOp": "append",
            },
            {
                "type": "tool/call",
                "seq": 4,
                "time": 1004,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "callId": "c1",
                    "name": "add",
                    "arguments": {"a": 1},
                },
            },
            {
                "type": "tool/result",
                "seq": 5,
                "time": 1005,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "id": "m-tool-1",
                        "role": "user",
                        "source": {"kind": "tool", "callId": "c1"},
                        "content": [
                            {
                                "type": "tool-result",
                                "toolCallId": "c1",
                                "content": [{"type": "text", "text": "3"}],
                            }
                        ],
                    },
                },
                "surfaceOp": "append",
            },
            {
                "type": "step/end",
                "seq": 6,
                "time": 1006,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "turn/end",
                "seq": 7,
                "time": 1007,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
        [
            EventType.TURN_START,
            EventType.STEP_START,
            EventType.USER_MESSAGE,
            EventType.ASSISTANT_MESSAGE,
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.STEP_END,
            EventType.TURN_END,
        ],
    ),
]


@pytest.mark.parametrize(
    ("version", "session_id", "rows", "expected_types"),
    _E2E_HISTORY_CASES,
    ids=["e2e-v0", "e2e-v1", "e2e-v2", "e2e-v3"],
)
def test_session_load_e2e_history_fixtures(
    tmp_path: Path,
    version: int,
    session_id: str,
    rows: list[dict],
    expected_types: list[EventType],
) -> None:
    """Verifies each end-to-end history fixture loads through Session.load.

    The inputs mirror test_e2e_v*_to_v4 in test_format.py; this pins the
    store mapping against the migration's own validated output instead of
    only the format-level byte encoding.
    """
    file_path = tmp_path / f"{session_id}.jsonl"
    _write_history_file(file_path, version, session_id, rows)
    loaded = Session.load(file_path)
    assert loaded.session_id == session_id
    assert [event.type for event in loaded.events] == expected_types


def test_session_append_to_migrated_session_is_refused(tmp_path: Path) -> None:
    """Verifies append fails closed on a migrated log instead of bricking it.

    Appending a current-version row to a historical log corrupts the next
    load (missing surfaceOp marker, seq drift from promoted events), so
    append refuses until the session is republished as current-version v4.
    """
    file_path = tmp_path / "mig_append.jsonl"
    _write_history_file(
        file_path,
        3,
        "mig_append",
        [
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
            {
                "type": "turn/end",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            },
        ],
    )
    before = file_path.read_bytes()
    loaded = Session.load(file_path)
    assert len(loaded.events) == 2
    with pytest.raises(
        SessionFormatUnsupportedError,
        match="cannot append to a migrated session until it is republished",
    ):
        loaded.append(EventType.USER_MESSAGE, {"text": "after"}, surface_op="append")
    assert len(loaded.events) == 2
    assert file_path.read_bytes() == before


def test_session_append_to_current_v4_session_loads_back(tmp_path: Path) -> None:
    """Verifies append still works on a current-version log (load, append, reload)."""
    file_path = tmp_path / "current_append.jsonl"
    file_path.write_text(
        _v4_header_line("current_append")
        + "\n"
        + json.dumps(
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}}
        )
        + "\n"
        + json.dumps(
            {
                "type": "turn/end",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = Session.load(file_path)
    loaded.append(EventType.USER_MESSAGE, {"text": "after"}, surface_op="append")
    reloaded = Session.load(file_path)
    assert [event.type for event in reloaded.events] == [
        EventType.TURN_START,
        EventType.TURN_END,
        EventType.USER_MESSAGE,
    ]
    assert reloaded.events[2].payload == {"text": "after"}


def _v4_file_with_two_events(file_path: Path, session_id: str) -> None:
    """Write a current-version v4 log with two events (load, append, reload)."""
    file_path.write_text(
        _v4_header_line(session_id)
        + "\n"
        + json.dumps(
            {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}}
        )
        + "\n"
        + json.dumps(
            {
                "type": "turn/end",
                "seq": 1,
                "time": 1001,
                "data": {"turn": 1, "reason": {"kind": "done"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _v3_two_event_rows() -> list[dict]:
    """Two valid historical v3 rows (turn/start, turn/end)."""
    return [
        {"type": "turn/start", "seq": 0, "time": 1000, "data": {"turn": 1}},
        {
            "type": "turn/end",
            "seq": 1,
            "time": 1001,
            "data": {"turn": 1, "reason": {"kind": "done"}},
        },
    ]


def test_session_constructed_over_historical_file_refuses_append(
    tmp_path: Path,
) -> None:
    """Verifies constructing a Session over a historical log refuses append.

    The load-path guard alone is not enough: a Session built directly over
    an existing v3 log used to append a current-version row to it, bricking
    the next load with a seq gap. Construction classifies the on-disk
    header, so append refuses and the file stays byte-identical.
    """
    file_path = tmp_path / "probe-bypass2.jsonl"
    _write_history_file(file_path, 3, "probe-bypass2", _v3_two_event_rows())
    before = file_path.read_bytes()
    session = Session(session_id="probe-bypass2", storage_path=file_path)
    with pytest.raises(
        SessionFormatUnsupportedError,
        match="cannot append to a migrated session until it is republished",
    ):
        session.append(EventType.USER_MESSAGE, {"text": "after"}, surface_op="append")
    assert session.events == []
    assert file_path.read_bytes() == before
    assert len(Session.load(file_path).events) == 2


def test_session_constructed_over_header_only_v4_file_appends(
    tmp_path: Path,
) -> None:
    """Verifies constructing a Session over a header-only v4 log appends.

    Positive control for the construction-time classification: a current
    v4 header classifies as current, so append writes the first event row
    under the existing header and the log reloads with one event.
    """
    file_path = tmp_path / "header_only.jsonl"
    file_path.write_text(_v4_header_line("header_only") + "\n", encoding="utf-8")
    session = Session(session_id="header_only", storage_path=file_path)
    session.append(EventType.USER_MESSAGE, {"text": "first"}, surface_op="append")
    reloaded = Session.load(file_path)
    assert len(reloaded.events) == 1
    assert reloaded.events[0].payload == {"text": "first"}


def test_session_constructed_over_empty_file_writes_header(tmp_path: Path) -> None:
    """Verifies constructing a Session over an empty file still writes a header."""
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("", encoding="utf-8")
    session = Session(session_id="empty", storage_path=file_path)
    session.append(EventType.USER_MESSAGE, {"text": "first"}, surface_op="append")
    lines = file_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["version"] == 4
    assert len(Session.load(file_path).events) == 1


def test_session_constructed_over_newer_version_file_raises(tmp_path: Path) -> None:
    """Verifies constructing a Session over a newer-version log fails loud."""
    file_path = tmp_path / "newer.jsonl"
    file_path.write_text(_v4_header_line("newer", version=5) + "\n", encoding="utf-8")
    with pytest.raises(
        SessionFormatUnsupportedError,
        match="uses log format v5",
    ):
        Session(session_id="newer", storage_path=file_path)


def test_session_constructed_over_malformed_header_raises(tmp_path: Path) -> None:
    """Verifies constructing a Session over a corrupt header fails loud."""
    file_path = tmp_path / "bad_header.jsonl"
    file_path.write_text('{"bad json\n', encoding="utf-8")
    with pytest.raises(
        SessionPersistenceCorruptionError,
        match="corrupt session header",
    ):
        Session(session_id="bad_header", storage_path=file_path)


def test_session_append_to_blank_only_file_writes_header(tmp_path: Path) -> None:
    """Verifies append to a blank-lines-only file writes a header first."""
    file_path = tmp_path / "blank.jsonl"
    file_path.write_text("\n\n", encoding="utf-8")
    session = Session(session_id="blank", storage_path=file_path)
    session.append(EventType.USER_MESSAGE, {"text": "first"}, surface_op="append")
    lines = [
        line
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert json.loads(lines[0])["version"] == 4
    assert len(Session.load(file_path).events) == 1


def test_session_append_refuses_when_log_replaced_after_load(tmp_path: Path) -> None:
    """Verifies append refuses when the on-disk log is replaced after load.

    A session loaded from a current v4 log must not silently append to a
    historical log swapped in underneath it: append re-classifies the
    on-disk header before mutating, so the replacement stays untouched
    and the in-memory log is unchanged.
    """
    file_path = tmp_path / "stale.jsonl"
    _v4_file_with_two_events(file_path, "stale")
    loaded = Session.load(file_path)
    assert len(loaded.events) == 2
    _write_history_file(file_path, 3, "stale", _v3_two_event_rows())
    replacement = file_path.read_bytes()
    with pytest.raises(
        SessionFormatUnsupportedError,
        match="cannot append to a migrated session until it is republished",
    ):
        loaded.append(EventType.USER_MESSAGE, {"text": "after"}, surface_op="append")
    assert len(loaded.events) == 2
    assert file_path.read_bytes() == replacement


def test_session_append_refuses_when_log_replaced_with_newer_version(
    tmp_path: Path,
) -> None:
    """Verifies append refuses when the log is replaced by a newer version."""
    file_path = tmp_path / "stale_newer.jsonl"
    _v4_file_with_two_events(file_path, "stale_newer")
    loaded = Session.load(file_path)
    assert len(loaded.events) == 2
    file_path.write_text(
        _v4_header_line("stale_newer", version=5) + "\n", encoding="utf-8"
    )
    replacement = file_path.read_bytes()
    with pytest.raises(
        SessionFormatUnsupportedError,
        match="uses log format v5",
    ):
        loaded.append(EventType.USER_MESSAGE, {"text": "after"}, surface_op="append")
    assert len(loaded.events) == 2
    assert file_path.read_bytes() == replacement


def test_session_append_uses_header_of_file_created_after_construction(
    tmp_path: Path,
) -> None:
    """Verifies append adopts a v4 header written after construction.

    A session built while no log exists must not write a second header
    when a current-version log appears at its path before the first
    append: append re-classifies the on-disk header and writes only the
    event row under it.
    """
    file_path = tmp_path / "appears_late.jsonl"
    session = Session(session_id="appears_late", storage_path=file_path)
    file_path.write_text(_v4_header_line("appears_late") + "\n", encoding="utf-8")
    session.append(EventType.USER_MESSAGE, {"text": "first"}, surface_op="append")
    lines = [
        line
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    assert json.loads(lines[0])["version"] == 4
    assert len(Session.load(file_path).events) == 1


# ---------------------------------------------------------------------------
# P2-1: event vocabulary, Session core, SessionStore, fork
# ---------------------------------------------------------------------------


def _seed_event(
    seq: int,
    type: str,
    payload: dict | None = None,
    session_id: str = "seed",
    **kwargs: object,
) -> SessionEvent:
    return SessionEvent(
        id=f"{session_id}:{seq}",
        session_id=session_id,
        type=type,
        timestamp=1000.0 + seq,
        payload=payload if payload is not None else {},
        seq=seq,
        **kwargs,  # type: ignore[arg-type]
    )


def _user_event(seq: int, text: str = "hi") -> SessionEvent:
    return _seed_event(
        seq,
        "user/message",
        {"role": "user", "id": f"m{seq}", "content": [{"type": "text", "text": text}]},
        surface_op="append",
    )


def _minimal_payload(type: str) -> dict:
    """A payload that passes payload validation for one event type."""
    if type == "user/message":
        return {
            "role": "user",
            "id": "m",
            "content": [{"type": "text", "text": "x"}],
        }
    if type == "system/message":
        return {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "s",
                "role": "system",
                "content": [{"type": "text", "text": "x"}],
            },
        }
    if type == "developer/message":
        return {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "d",
                "role": "developer",
                "content": [{"type": "text", "text": "x"}],
            },
        }
    if type == "assistant/message":
        return {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "a",
                "role": "assistant",
                "content": [{"type": "text", "text": "x"}],
            },
            "stream": [],
        }
    if type == "tool/result":
        return {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "t",
                "role": "tool",
                "toolCallId": "c1",
                "isError": False,
                "source": {"kind": "tool", "callId": "c1"},
                "content": [{"type": "text", "text": "x"}],
            },
        }
    if type == "request/header":
        return {"header": {"config": {}}, "reason": {"kind": "initial"}}
    return {}


class _ImageOffloadProjection:
    """Test projection interpreting image/offload decisions."""

    type = "image/offload"

    def project(self, event: SessionEvent, context: object) -> dict:
        return {event.seq: {"role": "user", "content": "offloaded"}}


def _v4_row(type: str, seq: int, data: dict | None = None, **extra: object) -> str:
    row: dict = {"type": type, "seq": seq, "time": 100000 + seq, "data": data or {}}
    row.update(extra)
    return json.dumps(row)


def _write_v4_file(file_path: Path, session_id: str, rows: list[str]) -> None:
    file_path.write_text(
        _v4_header_line(session_id) + "\n" + "".join(row + "\n" for row in rows),
        encoding="utf-8",
    )


def test_session_appends_all_62_types_and_replays_in_order(
    tmp_path: Path,
) -> None:
    """Verifies a session appends every known type and replays them in order."""
    file_path = tmp_path / "all62.jsonl"
    session = Session(
        session_id="all62",
        storage_path=file_path,
        projections=[_ImageOffloadProjection()],  # type: ignore[list-item]
    )
    types = sorted(KNOWN_SESSION_EVENT_TYPES)
    assert len(types) == 62
    for seq, type in enumerate(types):
        event = session.append(
            type,
            _minimal_payload(type),
            surface_op="append" if type in SURFACE_EVENT_TYPES else None,
        )
        assert event.seq == seq
        assert event.type == type
    assert [event.type for event in session.events] == types
    assert [event.seq for event in session.events] == list(range(62))
    # Reloading a log holding image/offload needs its projection supplied.
    reloaded = Session.load(file_path, projections=[_ImageOffloadProjection()])  # type: ignore[list-item]
    assert [event.type for event in reloaded.events] == types
    assert [event.seq for event in reloaded.events] == list(range(62))


def test_session_load_refuses_gapped_seq(tmp_path: Path) -> None:
    """Verifies a gapped seq is refused on load."""
    file_path = tmp_path / "gapped.jsonl"
    _write_v4_file(
        file_path,
        "gapped",
        [
            _v4_row("user/message", 0, {"text": "a"}, surfaceOp="append"),
            _v4_row("user/message", 2, {"text": "b"}, surfaceOp="append"),
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="seq gap"):
        Session.load(file_path)


def test_session_load_refuses_duplicated_seq(tmp_path: Path) -> None:
    """Verifies a duplicated seq is refused on load."""
    file_path = tmp_path / "dup.jsonl"
    _write_v4_file(
        file_path,
        "dup",
        [
            _v4_row("user/message", 0, {"text": "a"}, surfaceOp="append"),
            _v4_row("user/message", 0, {"text": "b"}, surfaceOp="append"),
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="seq gap"):
        Session.load(file_path)


def test_unknown_non_ignorable_refused_at_all_three_checkpoints(
    tmp_path: Path,
) -> None:
    """Verifies an unknown non-ignorable event is refused at load,
    storage validation, and the surface fold; an unknown ignorable event
    is carried opaquely through all three."""
    # Checkpoint 1: Session.load / restore path.
    file_path = tmp_path / "unknown.jsonl"
    _write_v4_file(file_path, "unknown", [_v4_row("vendor/widget", 0, {"x": 1})])
    with pytest.raises(SessionFormatUnsupportedError, match="unknown event type"):
        Session.load(file_path)
    # Checkpoint 2: _to_session_event storage validation.
    with pytest.raises(SessionFormatUnsupportedError, match="unknown event type"):
        _to_session_event(
            {"type": "vendor/widget", "seq": 0, "time": 100000, "data": {"x": 1}},
            "unknown",
            file_path,
        )
    # Checkpoint 3: the surface fold's surfaceOpOf rule — an unknown event
    # carrying surface metadata is refused there.
    unknown = _seed_event(0, "vendor/widget", {"x": 1}, surface_op="append")
    with pytest.raises(SurfaceError, match="not surface-eligible"):
        fold_surface([unknown])
    # Unknown ignorable events are carried opaquely at every checkpoint.
    _write_v4_file(
        file_path,
        "unknown",
        [_v4_row("vendor/widget", 0, {"x": 1}, ignorable=True)],
    )
    loaded = Session.load(file_path)
    assert loaded.events[0].type == "vendor/widget"
    assert loaded.events[0].ignorable is True
    assert loaded.events[0].payload == {"x": 1}
    carried = _to_session_event(
        {
            "type": "vendor/widget",
            "seq": 0,
            "time": 100000,
            "data": {"x": 1},
            "ignorable": True,
        },
        "unknown",
        file_path,
    )
    assert carried.type == "vendor/widget"
    assert carried.ignorable is True
    assert fold_surface(
        [_user_event(0), _seed_event(1, "vendor/widget", {"x": 1}, ignorable=True)]
    ).nodes == [0]


def test_session_load_refuses_retired_shapes(tmp_path: Path) -> None:
    """Verifies retired shapes are refused on load."""
    cases = [
        (
            "fallback",
            [
                _v4_row(
                    "request/header",
                    0,
                    {"header": {"config": {}}, "reason": {"kind": "fallback"}},
                )
            ],
        ),
        (
            "header.system",
            [
                _v4_row(
                    "request/header",
                    0,
                    {
                        "header": {"system": "prompt", "config": {}},
                        "reason": {"kind": "initial"},
                    },
                )
            ],
        ),
        (
            "tool-result",
            [
                _v4_row(
                    "user/message",
                    0,
                    {
                        "role": "user",
                        "id": "m0",
                        "content": [{"type": "tool-result", "toolCallId": "c1"}],
                    },
                    surfaceOp="append",
                )
            ],
        ),
        (
            "tool/code-dispatch",
            [_v4_row("tool/code-dispatch", 0, {"x": 1})],
        ),
    ]
    for name, rows in cases:
        file_path = (
            tmp_path / f"retired_{name.replace('/', '_').replace('.', '_')}.jsonl"
        )
        _write_v4_file(file_path, "retired", rows)
        with pytest.raises(SessionFormatUnsupportedError, match=name.split("/")[0]):
            Session.load(file_path)


def test_session_create_snapshots_seed_while_from_restore_adopts() -> None:
    """Verifies create deep-copies borrowed seed events while fromRestore
    adopts independently owned events without copying."""
    seed = [_user_event(0, "borrowed")]
    header = {
        "version": 4,
        "id": "seeded",
        "createdAt": 1000,
        "isSeeded": True,
        "delegationDepth": 0,
    }
    created = Session.create(
        "seeded", seed=seed, header=header, inherited_event_count=1
    )
    seed[0].payload["content"][0]["text"] = "mutated"
    assert created.events[0].payload["content"][0]["text"] == "borrowed"
    assert created.events[0] is not seed[0]

    adopted = Session.from_restore("restored", seed, header, 1, "detached")
    assert adopted.events[0] is seed[0]
    assert adopted.events[0].payload["content"][0]["text"] == "mutated"


def test_session_seeded_constructor_rules() -> None:
    """Verifies seeded construction: explicit seed and count, prefix or
    tagged marker, at most one tagged marker, and it must be final."""
    seeded_header = {
        "version": 4,
        "id": "s",
        "createdAt": 1000,
        "isSeeded": True,
        "delegationDepth": 0,
    }
    seed = [_user_event(0)]
    # Seeded without an inherited count.
    with pytest.raises(ValueError, match="inherited"):
        Session.create("s", seed=seed, header=seeded_header)
    # Inherited count without a seed.
    with pytest.raises(ValueError, match="seed"):
        Session.create("s", header=seeded_header, inherited_event_count=1)
    # Inherited count beyond the log.
    with pytest.raises(ValueError, match="inherited"):
        Session.create("s", seed=seed, header=seeded_header, inherited_event_count=2)
    # Seed seqs must be contiguous from 0.
    bad_seq = [_user_event(1)]
    with pytest.raises(ValueError, match="seq"):
        Session.create("s", seed=bad_seq, header=seeded_header, inherited_event_count=1)
    # Seed data must be lossless JSON.
    bad_data = [_user_event(0)]
    bad_data[0].payload = {"x": {1, 2}}
    with pytest.raises(ValueError, match="lossless"):
        Session.create(
            "s", seed=bad_data, header=seeded_header, inherited_event_count=1
        )
    # Seed surface metadata must validate.
    bad_surface = [_seed_event(0, "user/message", {"text": "x"})]
    with pytest.raises(ValueError, match="surfaceOp"):
        Session.create(
            "s", seed=bad_surface, header=seeded_header, inherited_event_count=1
        )
    # Two tagged markers are refused.
    marker = _seed_event(1, "session/end-seed", {"inherited": True})
    with pytest.raises(ValueError, match="final inherited marker"):
        Session.create(
            "s",
            seed=[
                _user_event(0),
                marker,
                _seed_event(2, "session/end-seed", {"inherited": True}),
            ],
            header=seeded_header,
            inherited_event_count=1,
        )
    # A tagged marker at the cut followed by child-owned events is
    # accepted: the inherited count identifies the final tagged marker.
    accepted = Session.create(
        "s",
        seed=[_user_event(0), marker, _user_event(2)],
        header=seeded_header,
        inherited_event_count=1,
    )
    assert len(accepted.events) == 3
    assert accepted.first_lifecycle_seq == 1
    assert accepted.is_own_seq(1)
    assert accepted.is_own_seq(2)


def test_session_constructor_markers() -> None:
    """Verifies constructor markers: a seeded child gets a tagged
    end-seed at its cut, an unseeded seed gets an untagged marker, and a
    restore appends none."""
    seeded_header = {
        "version": 4,
        "id": "child",
        "createdAt": 1000,
        "isSeeded": True,
        "delegationDepth": 0,
    }
    # Seeded child whose seed lacks a marker gets one at the cut.
    child = Session.create(
        "child",
        seed=[_user_event(0)],
        header=seeded_header,
        inherited_event_count=1,
    )
    assert len(child.events) == 2
    assert child.events[1].type == "session/end-seed"
    assert child.events[1].payload == {"inherited": True}
    assert child.events[1].seq == 1
    assert child.inherited_event_count == 1
    assert child.first_live_seq == 1
    assert not child.is_own_seq(0)
    assert child.is_own_seq(1)
    # Seeded child whose seed already ends in a tagged marker gets none.
    tagged = [_user_event(0), _seed_event(1, "session/end-seed", {"inherited": True})]
    child2 = Session.create(
        "child2",
        seed=tagged,
        header={**seeded_header, "id": "child2"},
        inherited_event_count=1,
    )
    assert len(child2.events) == 2
    assert child2.first_live_seq == 2
    assert child2.first_lifecycle_seq == 1
    # Unseeded seed whose last event is not an end-seed gets an untagged one.
    plain = Session.create("plain", seed=[_user_event(0)])
    assert len(plain.events) == 2
    assert plain.events[1].type == "session/end-seed"
    assert plain.events[1].payload == {}
    assert plain.inherited_event_count == 0
    # Unseeded seed already ending in an end-seed gets none.
    ended = Session.create(
        "ended", seed=[_user_event(0), _seed_event(1, "session/end-seed", {})]
    )
    assert len(ended.events) == 2
    # Restore adopts the stored log exactly: no constructor marker.
    header = {
        "version": 4,
        "id": "restored",
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    restored = Session.from_restore(
        "restored", [_user_event(0)], header, 0, "shared-frozen"
    )
    assert len(restored.events) == 1
    assert restored.events[0].type == "user/message"


def test_session_append_rules() -> None:
    """Verifies append rules: surface types require surfaceOp,
    non-surface types forbid surface metadata, data must be lossless
    JSON, and the deprecated accessors do not exist."""
    session = Session(session_id="rules")
    with pytest.raises(ValueError, match="requires a surfaceOp marker"):
        session.append("user/message", {"text": "x"})
    with pytest.raises(ValueError, match="cannot carry surface metadata"):
        session.append("turn/start", {"turn": 1}, surface_op="append")
    with pytest.raises(ValueError, match="cannot carry surface metadata"):
        session.append("turn/start", {"turn": 1}, source_event_seqs=[0])
    with pytest.raises(ValueError, match="cannot carry sourceEventSeqs"):
        session.append(
            "assistant/message",
            _minimal_payload("assistant/message"),
            surface_op="append",
            source_event_seqs=[0],
        )
    with pytest.raises(ValueError, match="non-JSON-serializable"):
        session.append("turn/start", {"turn": 1, "bad": {1, 2}})
    with pytest.raises(ValueError, match="unknown session event type"):
        session.append("session/compaction", {})
    assert not hasattr(Session, "eventAt")
    assert not hasattr(Session, "snapshotEvents")
    assert not hasattr(Session, "ownEvents")


def test_session_append_derives_seq_from_adopted_v4_log(tmp_path: Path) -> None:
    """Verifies append to an adopted v4 log derives seq from the on-disk
    log, not from the in-memory event count alone."""
    file_path = tmp_path / "adopted.jsonl"
    _write_v4_file(
        file_path,
        "adopted",
        [
            _v4_row("user/message", 0, {"text": "a"}, surfaceOp="append"),
            _v4_row("user/message", 1, {"text": "b"}, surfaceOp="append"),
        ],
    )
    session = Session(session_id="adopted", storage_path=file_path)
    event = session.append("user/message", {"text": "c"}, surface_op="append")
    assert event.seq == 2
    reloaded = Session.load(file_path)
    assert [event.seq for event in reloaded.events] == [0, 1, 2]
    assert reloaded.events[2].payload == {"text": "c"}


def test_session_append_writes_surface_op(tmp_path: Path) -> None:
    """Verifies append writes surfaceOp and sourceEventSeqs to the log."""
    file_path = tmp_path / "surface.jsonl"
    session = Session(session_id="surface", storage_path=file_path)
    session.append("user/message", {"text": "a"}, surface_op="append")
    session.append(
        "user/message",
        {"text": "b"},
        surface_op={"op": "replace", "startSeq": 0, "endSeq": 0},
        source_event_seqs=[0],
    )
    reloaded = Session.load(file_path)
    assert reloaded.events[0].surface_op == "append"
    assert reloaded.events[1].surface_op == {
        "op": "replace",
        "startSeq": 0,
        "endSeq": 0,
    }
    assert reloaded.events[1].source_event_seqs == [0]
    assert reloaded.surface_nodes() == [1]


def test_session_plugin_records() -> None:
    """Verifies plugin: records are always ignorable, owner-validated,
    and never members of the known vocabulary."""
    session = Session(session_id="plugins")
    seq = session.append_plugin_record("plugin:vendor/widget", {"x": 1})
    assert seq == 0
    event = session.events[0]
    assert event.type == "plugin:vendor/widget"
    assert event.ignorable is True
    record = plugin_record_of(event)
    assert record is not None
    assert record.type == "plugin:vendor/widget"
    assert record.data == {"x": 1}
    assert plugin_record_of(_user_event(0)) is None
    seq2 = append_plugin_record(session, "plugin:vendor/other", {"y": 2})
    assert seq2 == 1
    with pytest.raises(ValueError, match="outside the grammar"):
        session.append_plugin_record("vendor/widget", {})
    with pytest.raises(ValueError, match="unknown session event type"):
        session.append("plugin:vendor/widget", {"x": 1})
    with pytest.raises(ValueError, match="non-JSON-serializable"):
        session.append_plugin_record("plugin:vendor/widget", {"bad": {1}})


def test_session_plugin_record_carried_on_load(tmp_path: Path) -> None:
    """Verifies a plugin: record row loads opaquely."""
    file_path = tmp_path / "plugin.jsonl"
    _write_v4_file(
        file_path,
        "plugin",
        [_v4_row("plugin:vendor/widget", 0, {"x": 1}, ignorable=True)],
    )
    loaded = Session.load(file_path)
    assert loaded.events[0].type == "plugin:vendor/widget"
    assert loaded.events[0].ignorable is True


def test_session_preparation_dispose_is_idempotent() -> None:
    """Verifies preparation disposal is synchronous and idempotent, and
    abandoning a preparation detaches its session."""
    store = SessionStore()
    preparation = store.prepare("prep1")
    session = preparation.session
    assert store.get("prep1") is None
    preparation.dispose()
    preparation.dispose()
    assert session._detached is True
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        session.append("turn/start", {"turn": 1})


def test_session_preparation_context_manager() -> None:
    """Verifies a preparation works as a context manager."""
    store = SessionStore()
    with store.prepare("prep2") as preparation:
        session = preparation.session
        assert session.session_id == "prep2"
    assert session._detached is True
    assert store.get("prep2") is None


def test_store_prepare_mints_ids_and_validates_meta(tmp_path: Path) -> None:
    """Verifies prepare mints session-<n> ids, validates meta.cwd, and
    builds the immutable header."""
    store = SessionStore()
    first = store.create()
    second = store.create()
    assert first.session_id == "session-1"
    assert second.session_id == "session-2"
    assert store.get("session-1") is first
    assert store.get("missing") is None
    assert [s.session_id for s in store.list()] == ["session-1", "session-2"]
    with pytest.raises(ValueError, match="already exists"):
        store.create("session-1")
    with pytest.raises(ValueError, match="must be an absolute path"):
        store.prepare("bad", options={"meta": {"cwd": "relative/path"}})
    with pytest.raises(TypeError, match="must be an object"):
        store.prepare("bad2", options={"meta": "not-a-dict"})
    with pytest.raises(ValueError, match="unknown restore event state"):
        store.prepare(
            "bad3",
            options={
                "event_state": "frozen",
                "seed": [],
                "inherited_event_count": 0,
            },
        )
    with pytest.raises(ValueError, match="requires a seed"):
        store.prepare("bad4", options={"event_state": "detached"})
    prepared = store.prepare("headed", options={"meta": {"cwd": str(tmp_path)}})
    header = prepared.session.header
    assert header["version"] == 4
    assert header["id"] == "headed"
    assert header["cwd"] == str(tmp_path)
    assert isinstance(header["createdAt"], int)
    assert header["isSeeded"] is False


def test_store_prepare_restores_via_from_restore() -> None:
    """Verifies prepare with eventState restores without snapshotting."""
    store = SessionStore()
    seed = [_user_event(0)]
    preparation = store.prepare(
        "restored",
        options={
            "event_state": "shared-frozen",
            "seed": seed,
            "inherited_event_count": 1,
            "meta": {"isSeeded": True},
        },
    )
    assert preparation.session.events[0] is seed[0]
    assert preparation.session.inherited_event_count == 1


def test_store_append_gating() -> None:
    """Verifies store-owned append gating: writes stay disabled until
    announce completes, and removal detaches the session."""
    store = SessionStore()
    preparation = store.prepare("gated")
    session = preparation.session
    with pytest.raises(ValueError, match="before session/created completes"):
        session.append("turn/start", {"turn": 1})
    store.enter(preparation)
    with pytest.raises(ValueError, match="before session/created completes"):
        session.append("turn/start", {"turn": 1})
    store.announce(session)
    session.append("turn/start", {"turn": 1})
    assert len(session.events) == 1
    assert store.remove(session) is True
    assert store.remove(session) is False
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        session.append("turn/start", {"turn": 2})
    # Independent sessions append freely.
    free = Session(session_id="free")
    free.append("turn/start", {"turn": 1})
    assert len(free.events) == 1


def test_store_announce_exactly_once_and_veto() -> None:
    """Verifies announce emits session/created exactly once and a
    synchronous throw vetoes publication."""
    ctx = Context()
    seen: list[Session] = []
    ctx.on("session/created", lambda session: seen.append(session))
    store = SessionStore(ctx)
    session = store.create("once")
    assert seen == [session]
    with pytest.raises(ValueError, match="already announced"):
        store.announce(session)

    ctx2 = Context()

    def veto(session: Session) -> None:
        raise RuntimeError("veto")

    ctx2.on("session/created", veto)
    store2 = SessionStore(ctx2)
    with pytest.raises(RuntimeError, match="veto"):
        store2.create("vetoed")
    assert store2.get("vetoed") is None
    # A failed announcement leaves the session read-only and unretryable:
    # the entry stays, but announce cannot run twice and appends stay
    # gated on the never-completed announcement.
    preparation = store2.prepare("vetoed2")
    store2.enter(preparation)
    with pytest.raises(RuntimeError, match="veto"):
        store2.announce(preparation.session)
    assert store2.get("vetoed2") is preparation.session
    with pytest.raises(ValueError, match="already announced"):
        store2.announce(preparation.session)
    with pytest.raises(ValueError, match="before session/created completes"):
        preparation.session.append("turn/start", {"turn": 1})


def test_store_announce_async_rejection_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies an async session/created rejection is logged, not vetoing."""
    ctx = Context()

    async def reject(session: Session) -> None:
        raise RuntimeError("async veto")

    ctx.on("session/created", reject)
    store = SessionStore(ctx)
    with caplog.at_level(logging.WARNING):
        session = store.create("async-ok")
    assert store.get("async-ok") is session
    assert "rejected" in caplog.text
    assert "async veto" in caplog.text


def test_store_enter_rechecks_duplicates() -> None:
    """Verifies a stale prepared session cannot overwrite a live entry."""
    store = SessionStore()
    first = store.prepare("dup-enter")
    second = store.prepare("dup-enter")
    store.enter(first)
    with pytest.raises(ValueError, match="already exists"):
        store.enter(second)


def test_store_enter_rejects_cross_store_attach() -> None:
    """Verifies a session prepared in one store cannot enter another."""
    store_a = SessionStore()
    store_b = SessionStore()
    preparation = store_a.prepare("cross")
    with pytest.raises(ValueError, match="already attached to a store"):
        store_b.enter(preparation)


def test_store_append_reentrancy_guard() -> None:
    """Verifies a reentrant append while publication is open throws."""
    ctx = Context()
    store = SessionStore(ctx)
    errors: list[Exception] = []
    attempted: list[bool] = []

    def reenter(session: Session, event: SessionEvent) -> None:
        attempted.append(True)
        try:
            session.append("turn/start", {"turn": 2})
        except ValueError as error:
            errors.append(error)

    ctx.on("session/event", reenter)
    session = store.create("reentrant")
    session.append("turn/start", {"turn": 1})
    assert attempted == [True]
    assert len(errors) == 1
    assert "cannot reenter" in str(errors[0])
    # The outer event still committed.
    assert len(session.events) == 1


def test_store_register_message_projection() -> None:
    """Verifies one projection per event type and that disposing it makes
    sessions that used it refuse further derivation."""

    class OffloadProjection:
        type = "image/offload"

        def project(self, event: SessionEvent, context: object) -> dict:
            return {0: {"role": "user", "content": "offloaded"}}

    store = SessionStore()
    projection = OffloadProjection()
    disposer = store.register_message_projection(projection)  # type: ignore[arg-type]
    assert store.message_projections == [projection]
    with pytest.raises(ValueError, match="already registered"):
        store.register_message_projection(projection)  # type: ignore[arg-type]
    session = store.create("projected")
    session.append("image/offload", {"seq": 0})
    assert session.events[0].type == "image/offload"
    disposer()
    assert store.message_projections == []
    disposer()
    with pytest.raises(SurfaceError, match="was removed or replaced"):
        session.surface_nodes()
    with pytest.raises(SurfaceError, match="was removed or replaced"):
        session.append("turn/start", {"turn": 1})


async def test_store_flush() -> None:
    """Verifies flush is the single durability entry point: all-settled,
    throw-first, and reports whether any listener participated."""
    store = SessionStore()
    session = store.create("flushable")
    assert await store.flush(session) is False

    ctx = Context()
    store2 = SessionStore(ctx)
    session2 = store2.create("flushable2")
    calls: list[str] = []

    def sync_listener(session: Session) -> None:
        calls.append("sync")

    async def async_listener(session: Session) -> None:
        calls.append("async")

    ctx.on("session/flush", sync_listener)
    ctx.on("session/flush", async_listener)
    assert await store2.flush(session2) is True
    assert calls == ["sync", "async"]

    ctx3 = Context()
    store3 = SessionStore(ctx3)
    session3 = store3.create("flushable3")
    order: list[str] = []

    def failing(session: Session) -> None:
        order.append("failing")
        raise RuntimeError("flush boom")

    def after(session: Session) -> None:
        order.append("after")

    ctx3.on("session/flush", failing)
    ctx3.on("session/flush", after)
    with pytest.raises(RuntimeError, match="flush boom"):
        await store3.flush(session3)
    assert order == ["failing", "after"]

    free = Session(session_id="free-flush")
    with pytest.raises(ValueError, match="is not live in this store"):
        await store3.flush(free)


def _live_source(store: SessionStore, session_id: str = "source") -> Session:
    """A live source session: one closed turn, then one open turn."""
    session = store.create(session_id, options={"meta": {"cwd": "/tmp"}})
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("user/message", {"text": "hi"}, surface_op="append")
    session.append("step/end", {"turn": 1, "step": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    session.append("turn/start", {"turn": 2})
    session.append("step/start", {"turn": 2, "step": 1})
    session.append("user/message", {"text": "go"}, surface_op="append")
    return session


def test_store_fork_mid_log_boundary() -> None:
    """Verifies a mid-log fork yields the exact prefix plus a tagged
    end-seed, with closed steps/turns preserved unchanged."""
    store = SessionStore()
    source = _live_source(store)
    child = store.fork(source, 4, "child")
    assert [event.type for event in child.events] == [
        "turn/start",
        "step/start",
        "user/message",
        "step/end",
        "turn/end",
        "session/end-seed",
    ]
    assert child.events[:5] == source.events[:5]
    marker = child.events[5]
    assert marker.type == "session/end-seed"
    assert marker.payload == {"inherited": True}
    assert marker.seq == 5
    assert child.inherited_event_count == 5
    assert child.header["isSeeded"] is True
    assert child.header["cwd"] == "/tmp"
    assert not child.is_own_seq(0)
    assert child.is_own_seq(5)
    assert store.get("child") is child
    # The child is live and appendable.
    child.append("turn/start", {"turn": 3})
    assert child.events[-1].seq == 6


def test_store_fork_default_boundary_and_open_turn() -> None:
    """Verifies the default boundary is the source's last event and an
    open tail receives synthetic closers."""
    store = SessionStore()
    _live_source(store)
    child = store.fork("source")
    assert [event.type for event in child.events] == [
        "turn/start",
        "step/start",
        "user/message",
        "step/end",
        "turn/end",
        "turn/start",
        "step/start",
        "user/message",
        "session/end-seed",
        "step/end",
        "turn/end",
    ]
    assert child.events[8].payload == {"inherited": True}
    assert child.events[9].payload == {"turn": 2, "step": 1}
    assert child.events[10].payload["reason"] == {"kind": "forked"}
    assert child.inherited_event_count == 8


def test_store_fork_empty_source() -> None:
    """Verifies forking an empty source yields an empty child with a
    tagged end-seed at seq 0."""
    store = SessionStore()
    source = store.create("empty-source")
    child = store.fork(source, child_session_id="empty-child")
    assert len(child.events) == 1
    assert child.events[0].type == "session/end-seed"
    assert child.events[0].payload == {"inherited": True}
    assert child.events[0].seq == 0
    assert child.inherited_event_count == 0
    assert child.header["isSeeded"] is True


def test_store_fork_errors() -> None:
    """Verifies fork error codes: not found, not live, invalid boundary,
    already exists, and a non-session source type."""
    store = SessionStore()
    source = _live_source(store)
    with pytest.raises(SessionForkError) as excinfo:
        store.fork("missing")
    assert excinfo.value.code == "SESSION_NOT_FOUND"
    free = Session(session_id="free-fork")
    with pytest.raises(SessionForkError) as excinfo:
        store.fork(free)
    assert excinfo.value.code == "SESSION_NOT_LIVE"
    with pytest.raises(SessionForkError) as excinfo:
        store.fork(source, 99)
    assert excinfo.value.code == "INVALID_BOUNDARY"
    with pytest.raises(SessionForkError) as excinfo:
        store.fork(source, -1)
    assert excinfo.value.code == "INVALID_BOUNDARY"
    store.fork(source, 4, "taken")
    with pytest.raises(SessionForkError) as excinfo:
        store.fork(source, 4, "taken")
    assert excinfo.value.code == "SESSION_ALREADY_EXISTS"
    with pytest.raises(TypeError, match="must be a live session"):
        store.fork(42)  # type: ignore[arg-type]


def test_store_fork_preserves_failed_calls_unchanged() -> None:
    """Verifies a fork from a boundary after a failed call with no result
    keeps the failed call exactly as recorded."""
    store = SessionStore()
    source = store.create("failed-source")
    source.append("turn/start", {"turn": 1})
    source.append("step/start", {"turn": 1, "step": 1})
    source.append(
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "a2",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool-call",
                        "id": "c1",
                        "name": "search",
                        "arguments": "{}",
                    }
                ],
            },
            "stream": [],
        },
        surface_op="append",
    )
    source.append("step/end", {"turn": 1, "step": 1})
    source.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    child = store.fork(source, 4, "failed-child")
    # Closed turn: no synthetic closers, failed call left as recorded.
    assert len(child.events) == 6
    assert child.events[2].type == "assistant/message"
    assert child.events[2].payload["message"]["content"][0]["id"] == "c1"
    assert child.events[5].type == "session/end-seed"


def test_session_header_origin_and_cwd_validation() -> None:
    """Verifies a header with a bad origin or a relative cwd fails loud."""
    bad_origin = {
        "version": 4,
        "id": "s",
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
        "origin": "other",
    }
    with pytest.raises(SessionFormatError, match="origin must be"):
        Session.create("s", header=bad_origin)
    bad_cwd = {
        "version": 4,
        "id": "s",
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
        "cwd": "relative/path",
    }
    with pytest.raises(SessionFormatError, match="cwd must be absolute"):
        Session.create("s", header=bad_cwd)


def test_session_unseeded_inherited_count_must_be_zero() -> None:
    """Verifies an unseeded session cannot carry an inherited count."""
    with pytest.raises(ValueError, match="inherited event count must be 0"):
        Session.create("s", inherited_event_count=1)


def test_session_from_restore_rejects_bad_state_and_data() -> None:
    """Verifies fromRestore validates its event state and seed data."""
    header = {
        "version": 4,
        "id": "s",
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    with pytest.raises(ValueError, match="unknown restore event state"):
        Session.from_restore("s", [], header, 0, "frozen")
    bad_seed = [_user_event(0)]
    bad_seed[0].payload = {"x": {1, 2}}
    with pytest.raises(ValueError, match="losslessly"):
        Session.from_restore("s", bad_seed, header, 0, "detached")


def test_session_seeded_seed_must_match_prefix() -> None:
    """Verifies a seeded seed that neither equals its inherited prefix
    nor marks its cut is refused."""
    seeded_header = {
        "version": 4,
        "id": "s",
        "createdAt": 1000,
        "isSeeded": True,
        "delegationDepth": 0,
    }
    with pytest.raises(ValueError, match="must equal its inherited prefix"):
        Session.create(
            "s",
            seed=[_user_event(0), _user_event(1)],
            header=seeded_header,
            inherited_event_count=1,
        )


def test_session_id_and_seq_properties() -> None:
    """Verifies the id and seq properties track the session."""
    session = Session(session_id="props")
    assert session.id == "props"
    assert session.seq == 0
    session.append("turn/start", {"turn": 1})
    assert session.seq == 1


def test_session_load_blank_only_file(tmp_path: Path) -> None:
    """Verifies a blank-lines-only file loads as an empty session."""
    file_path = tmp_path / "blank_only.jsonl"
    file_path.write_text("\n\n", encoding="utf-8")
    loaded = Session.load(file_path)
    assert loaded.session_id == "blank_only"
    assert loaded.events == []


def test_session_load_refuses_corrupt_row(tmp_path: Path) -> None:
    """Verifies a corrupt JSON row is corruption, not truncation."""
    file_path = tmp_path / "corrupt_row.jsonl"
    file_path.write_text(
        _v4_header_line("corrupt_row") + "\n" + '{"bad json\n', encoding="utf-8"
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="corrupt session row"):
        Session.load(file_path)


def test_session_load_tagged_end_seed_inherited_count(tmp_path: Path) -> None:
    """Verifies a tagged end-seed marker sets the inherited count on load."""
    file_path = tmp_path / "tagged.jsonl"
    file_path.write_text(
        _v4_header_line("tagged", isSeeded=True)
        + "\n"
        + _v4_row("user/message", 0, {"text": "a"}, surfaceOp="append")
        + "\n"
        + _v4_row("session/end-seed", 1, {"inherited": True})
        + "\n"
        + _v4_row("user/message", 2, {"text": "b"}, surfaceOp="append")
        + "\n",
        encoding="utf-8",
    )
    loaded = Session.load(file_path)
    assert loaded.inherited_event_count == 1
    assert len(loaded.events) == 3
    assert loaded.events[1].payload == {"inherited": True}


def test_session_preparation_create_without_release() -> None:
    """Verifies a preparation without a release callback disposes cleanly."""
    session = Session(session_id="no-release")
    preparation = SessionPreparation.create(session)
    preparation.dispose()
    preparation.dispose()


def test_store_listeners_without_event_bus() -> None:
    """Verifies a store whose context has no event bus degrades to no
    listeners instead of failing."""
    store = SessionStore(object())
    session = store.create("no-bus")
    assert store.get("no-bus") is session
    store2 = SessionStore(SimpleNamespace(_event_bus=object()))
    session2 = store2.create("no-filter")
    assert store2.get("no-filter") is session2


async def test_store_flush_without_event_bus() -> None:
    """Verifies flush reports no participation without an event bus."""
    store = SessionStore(object())
    session = store.create("flush-no-bus")
    assert await store.flush(session) is False


def test_store_prepare_origin_validation() -> None:
    """Verifies prepare refuses a header origin other than subagent."""
    store = SessionStore()
    with pytest.raises(SessionFormatError, match="origin must be"):
        store.prepare("bad-origin", options={"meta": {"origin": "other"}})


def test_store_prepare_minted_id_skips_live() -> None:
    """Verifies minted session-<n> ids skip ids already live."""
    store = SessionStore()
    store.create("session-1")
    minted = store.create()
    assert minted.session_id == "session-2"


def test_store_prepare_release_noop_after_enter() -> None:
    """Verifies disposing a preparation after enter leaves the live
    session untouched."""
    store = SessionStore()
    preparation = store.prepare("entered")
    store.enter(preparation)
    preparation.dispose()
    assert store.get("entered") is preparation.session


def test_store_detach_twice_via_enter_disposer() -> None:
    """Verifies the enter disposer is idempotent: a second detach is a
    no-op, not an error."""
    store = SessionStore()
    preparation = store.prepare("twice")
    detach = store.enter(preparation)
    detach()
    detach()
    assert store.get("twice") is None


def test_store_remove_wrong_entry_returns_false() -> None:
    """Verifies remove returns False when the store holds a different
    entry for the id, or the session belongs to another store."""
    store = SessionStore()
    first = store.prepare("shared-id")
    second = store.prepare("shared-id")
    store.enter(first)
    assert store.remove(second.session) is False
    assert store.get("shared-id") is first.session
    other = SessionStore()
    live = other.create("elsewhere")
    assert store.remove(live) is False
    assert other.get("elsewhere") is live


def test_store_disposed_event_emitted() -> None:
    """Verifies removing a live session emits its paired disposal."""
    ctx = Context()
    disposed: list[Session] = []
    ctx.on("session/disposed", lambda session: disposed.append(session))
    store = SessionStore(ctx)
    session = store.create("disposable")
    assert store.remove(session) is True
    assert disposed == [session]


def test_store_detach_during_announce() -> None:
    """Verifies a session/created listener that removes the session
    defers the detach until announce completes."""
    ctx = Context()
    store = SessionStore(ctx)

    def remove_on_created(session: Session) -> None:
        store.remove(session)

    ctx.on("session/created", remove_on_created)
    session = store.create("detach-on-created")
    assert store.get("detach-on-created") is None
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        session.append("turn/start", {"turn": 1})


def test_store_detach_during_append() -> None:
    """Verifies a session/event listener that removes the session defers
    the detach until the append commits."""
    ctx = Context()
    store = SessionStore(ctx)
    removed: list[bool] = []

    def remove_on_event(session: Session, event: SessionEvent) -> None:
        removed.append(store.remove(session))

    ctx.on("session/event", remove_on_event)
    session = store.create("detach-on-append")
    session.append("turn/start", {"turn": 1})
    assert removed == [True]
    assert len(session.events) == 1
    assert store.get("detach-on-append") is None
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        session.append("turn/start", {"turn": 2})


def test_store_announce_not_live_raises() -> None:
    """Verifies announce on a session that is not live raises."""
    store = SessionStore()
    free = Session(session_id="free-announce")
    with pytest.raises(ValueError, match="is not live in this store"):
        store.announce(free)


async def test_store_announce_async_rejection_with_running_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies an async session/created rejection is logged when a
    loop is already running."""
    ctx = Context()

    async def reject(session: Session) -> None:
        raise RuntimeError("async veto in loop")

    ctx.on("session/created", reject)
    store = SessionStore(ctx)
    with caplog.at_level(logging.WARNING):
        session = store.create("async-loop")
        # One turn lets the listener task run and fail; a second lets its
        # done callback log the rejection.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert store.get("async-loop") is session
    assert "rejected" in caplog.text
    assert "async veto in loop" in caplog.text


async def test_store_announce_async_success_with_running_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies a succeeding async session/created listener logs nothing
    when a loop is already running."""
    ctx = Context()
    seen: list[Session] = []

    async def accept(session: Session) -> None:
        seen.append(session)

    ctx.on("session/created", accept)
    store = SessionStore(ctx)
    with caplog.at_level(logging.WARNING):
        session = store.create("async-success")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert seen == [session]
    assert store.get("async-success") is session
    assert "rejected" not in caplog.text


def test_store_announce_non_coroutine_awaitable() -> None:
    """Verifies a session/created listener returning a non-coroutine
    awaitable is left un-awaited when no loop is running."""
    ctx = Context()
    awaited: list[bool] = []

    class Awaitable:
        def __await__(self):  # type: ignore[no-untyped-def]
            awaited.append(True)
            yield from ()

    def listener(session: Session) -> Awaitable:
        return Awaitable()

    ctx.on("session/created", listener)
    store = SessionStore(ctx)
    session = store.create("awaitable")
    assert store.get("awaitable") is session
    assert awaited == []


def test_session_append_non_json_surface_metadata() -> None:
    """Verifies non-JSON surface metadata is refused at append."""
    session = Session(session_id="bad-meta")
    with pytest.raises(ValueError, match="non-JSON-serializable surface metadata"):
        session.append(
            "user/message",
            {"text": "x"},
            surface_op={"op": "append", "bad": {1, 2}},  # type: ignore[dict-item]
        )
