"""Unit tests for Session and derive_messages projection."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.format import (
    SessionFormatError,
    SessionFormatUnsupportedError,
    SessionPersistenceCorruptionError,
)
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
    assert loaded_session.events[0].id == "sess_custom:1"
    assert loaded_session.events[1].id == "sess_custom:10"
    assert loaded_session.events[0].payload == {"text": "persisted text 1"}
    assert loaded_session.events[0].seq == 1
    assert loaded_session.events[1].seq == 10


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
        EventType.TURN_END,
    ]


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


def test_session_load_skips_ignorable_unknown_and_refuses_known_unknown(
    tmp_path: Path,
) -> None:
    """Verifies unknown ignorable rows are skipped while unknown rows are refused."""
    file_path = tmp_path / "sess_unknown.jsonl"
    file_path.write_text(
        _v4_header_line("unknown_sess")
        + "\n"
        + json.dumps(
            {
                "type": "vendor/widget",
                "seq": 1,
                "time": 100000,
                "data": {"x": 1},
                "ignorable": True,
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "user/message",
                "seq": 2,
                "time": 100001,
                "data": {"text": "kept"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = Session.load(file_path)
    assert len(loaded.events) == 1
    assert loaded.events[0].payload == {"text": "kept"}

    file_path.write_text(
        _v4_header_line("unknown_sess")
        + "\n"
        + json.dumps(
            {"type": "vendor/widget", "seq": 1, "time": 100000, "data": {"x": 1}}
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
        loaded.append(
            SessionEvent(
                id="ev_after",
                session_id="mig_append",
                type=EventType.USER_MESSAGE,
                payload={"text": "after"},
            )
        )
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
    loaded.append(
        SessionEvent(
            id="ev_after",
            session_id="current_append",
            type=EventType.USER_MESSAGE,
            payload={"text": "after"},
        )
    )
    reloaded = Session.load(file_path)
    assert [event.type for event in reloaded.events] == [
        EventType.TURN_START,
        EventType.TURN_END,
        EventType.USER_MESSAGE,
    ]
    assert reloaded.events[2].payload == {"text": "after"}
