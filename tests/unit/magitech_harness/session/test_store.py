"""Unit tests for Session and derive_messages projection."""

from pathlib import Path
from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.store import Session, derive_messages


def test_derive_messages_ignores_unknown_and_handles_roles() -> None:
    """Verifies that derive_messages correctly projects user, assistant, and tool messages."""
    events = [
        SessionEvent(
            id="1", session_id="s", type=EventType.TURN_START, payload={}
        ),
        SessionEvent(
            id="2", session_id="s", type=EventType.USER_MESSAGE, payload={"text": "hello"}
        ),
        SessionEvent(
            id="3",
            session_id="s",
            type=EventType.ASSISTANT_MESSAGE,
            payload={"text": "I can help", "tool_calls": [{"call_id": "c1", "name": "test", "arguments": {}}]},
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
    """Verifies Session persistence and loading from JSONL file."""
    file_path = tmp_path / "sess_save.jsonl"
    session = Session(session_id="sess_custom", storage_path=file_path)

    event = SessionEvent(
        id="ev_1",
        session_id="sess_custom",
        type=EventType.USER_MESSAGE,
        payload={"text": "persisted text"},
    )
    session.append(event)
    assert file_path.exists()

    # Append blank line to verify line.strip() branch (88->87)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n\n")

    # Load from disk
    loaded_session = Session.load(file_path)
    assert loaded_session.session_id == "sess_custom"
    assert len(loaded_session.events) == 1
    assert loaded_session.events[0].payload == {"text": "persisted text"}


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
