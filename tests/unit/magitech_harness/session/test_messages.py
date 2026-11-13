"""Unit tests for message factories."""

from magitech_harness.session.messages import (
    create_assistant_message,
    create_developer_message,
    create_message,
    create_system_message,
    create_tool_result_message,
    create_user_message,
)


def test_create_message_mints_hex_id_and_detaches_inputs() -> None:
    content = [{"type": "text", "text": "hi"}]
    source = {"kind": "user"}
    first = create_message(role="user", content=content, source=source)
    second = create_message(role="user", content=content, source=source)
    assert first["id"] != second["id"]
    assert len(first["id"]) == 32
    assert all(char in "0123456789abcdef" for char in first["id"])
    content[0]["text"] = "changed"
    source["kind"] = "other"
    assert first["content"] == [{"type": "text", "text": "hi"}]
    assert first["source"] == {"kind": "user"}


def test_create_system_message_empty_and_text() -> None:
    empty = create_system_message("")
    assert empty["role"] == "system"
    assert empty["content"] == []
    assert empty["source"] == {"kind": "system-prompt"}
    text = create_system_message("Be helpful.")
    assert text["content"] == [{"type": "text", "text": "Be helpful."}]


def test_create_user_and_developer_messages() -> None:
    user = create_user_message(
        content=[{"type": "text", "text": "hi"}], source={"kind": "user"}
    )
    assert user["role"] == "user"
    assert user["source"] == {"kind": "user"}
    dev = create_developer_message(
        content=[{"type": "tool-addition", "toolName": "t"}],
        source={"kind": "tool-registry"},
    )
    assert dev["role"] == "developer"
    assert dev["source"] == {"kind": "tool-registry"}


def test_create_assistant_message_source() -> None:
    message = create_assistant_message(
        content=[{"type": "text", "text": "answer"}],
        provider="mock",
        model="mock-model",
    )
    assert message["role"] == "assistant"
    assert message["source"] == {
        "kind": "model",
        "provider": "mock",
        "model": "mock-model",
    }


def test_create_tool_result_message_fields() -> None:
    message = create_tool_result_message(
        call_id="c1",
        content=[{"type": "text", "text": "42"}],
        is_error=True,
    )
    assert message["role"] == "tool"
    assert message["toolCallId"] == "c1"
    assert message["source"] == {"kind": "tool", "callId": "c1"}
    assert message["isError"] is True
    default = create_tool_result_message(
        call_id="c2", content=[{"type": "text", "text": "ok"}]
    )
    assert default["isError"] is False
