"""Unit tests for session query semantic text extraction."""

from __future__ import annotations

from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session_query.extraction import extract_semantic_text


def _event(event_type: str, payload: dict) -> SessionEvent:
    return SessionEvent(
        id="s1:0",
        session_id="s1",
        type=event_type,
        timestamp=1000.0,
        payload=payload,
        seq=0,
    )


def test_extract_user_message_text() -> None:
    event = _event(
        EventType.USER_MESSAGE,
        {"message": {"content": [{"type": "text", "text": "hello world"}]}},
    )
    assert extract_semantic_text(event) == "hello world"


def test_extract_user_message_plain_content() -> None:
    event = _event(
        EventType.USER_MESSAGE,
        {"message": {"content": "plain text"}},
    )
    assert extract_semantic_text(event) == "plain text"


def test_extract_user_message_no_content() -> None:
    event = _event(
        EventType.USER_MESSAGE,
        {"message": {}},
    )
    assert extract_semantic_text(event) == ""


def test_extract_system_message() -> None:
    event = _event(
        EventType.SYSTEM_MESSAGE,
        {"message": {"content": [{"type": "text", "text": "system note"}]}},
    )
    assert extract_semantic_text(event) == "system note"


def test_extract_assistant_message() -> None:
    event = _event(
        EventType.ASSISTANT_MESSAGE,
        {"message": {"content": [{"type": "text", "text": "assistant reply"}]}},
    )
    assert extract_semantic_text(event) == "assistant reply"


def test_extract_tool_result() -> None:
    event = _event(
        EventType.TOOL_RESULT,
        {"message": {"content": [{"type": "text", "text": "tool output"}]}},
    )
    assert extract_semantic_text(event) == "tool output"


def test_extract_request_header() -> None:
    event = _event(
        EventType.REQUEST_HEADER,
        {"config": {"provider": "deepseek", "model": "deepseek-chat"}},
    )
    text = extract_semantic_text(event)
    assert "deepseek" in text
    assert "deepseek-chat" in text


def test_extract_request_context() -> None:
    event = _event(
        EventType.REQUEST_CONTEXT,
        {"provider": "deepseek", "model": "deepseek-chat"},
    )
    text = extract_semantic_text(event)
    assert "deepseek" in text
    assert "deepseek-chat" in text


def test_extract_reasoning_block_no_text() -> None:
    """Reasoning blocks contribute no semantic text."""
    event = _event(
        EventType.ASSISTANT_MESSAGE,
        {"message": {"content": [{"type": "reasoning", "text": "thinking..."}]}},
    )
    assert extract_semantic_text(event) == ""


def test_extract_empty_event() -> None:
    event = _event(EventType.USER_MESSAGE, {})
    assert extract_semantic_text(event) == ""
