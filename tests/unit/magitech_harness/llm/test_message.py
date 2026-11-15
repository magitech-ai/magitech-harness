"""Unit tests for immutable conversation message factories."""

from __future__ import annotations

import pytest

from magitech_harness.llm.brand import MessageId, ToolCallId
from magitech_harness.llm.message import (
    CONTEXT_SUMMARY_MAX_CHARS,
    bound_context_summary,
    create_assistant_message,
    create_developer_message,
    create_message,
    create_system_message,
    create_tool_result_message,
    create_user_message,
    freeze_message,
)
from magitech_harness.llm.values import is_frozen


def test_message_factories_immutability_and_detachment() -> None:
    content = [{"type": "tool-addition", "toolName": "search"}]
    dev = create_developer_message({"content": content, "source": {"kind": "test"}})
    content[0]["toolName"] = "changed"
    assert dev["role"] == "developer"
    assert dev["content"] == [{"type": "tool-addition", "toolName": "search"}]
    assert is_frozen(dev["content"][0])
    assert freeze_message(dev)["id"] == dev["id"]

    dev_kw = create_developer_message(
        content=[{"type": "text", "text": "kw"}], source={"kind": "kw"}
    )
    assert dev_kw["source"] == {"kind": "kw"}

    exact = "x" * CONTEXT_SUMMARY_MAX_CHARS
    assert bound_context_summary(exact) == exact
    assert bound_context_summary(exact + "x") == f"{exact[:-1]}\u2026"

    sys_msg = create_system_message("rule")
    assert sys_msg["role"] == "system"
    assert sys_msg["content"] == [{"type": "text", "text": "rule"}]
    assert sys_msg["source"] == {"kind": "system-prompt"}
    assert is_frozen(sys_msg)
    assert is_frozen(sys_msg["source"])
    assert create_system_message("")["content"] == []

    user_input = {
        "content": [{"type": "text", "text": "original"}],
        "source": {"kind": "test"},
    }
    user_msg = create_user_message(user_input)
    assert user_msg["id"]
    assert user_msg["role"] == "user"
    assert user_msg is not user_input
    assert is_frozen(user_msg)
    assert is_frozen(user_msg["content"])
    assert is_frozen(user_msg["content"][0])
    user_input["content"][0]["text"] = "caller mutation"
    assert user_msg["content"] == [{"type": "text", "text": "original"}]
    with pytest.raises(TypeError):
        user_msg["content"][0]["text"] = "observer mutation"

    user_kw = create_user_message(
        content=[{"type": "text", "text": "kw"}], source={"kind": "kw"}
    )
    assert user_kw["role"] == "user"

    generic = create_message(
        role="user",
        content=[{"type": "text", "text": "g"}],
        source={"kind": "g"},
    )
    assert generic["role"] == "user"

    existing_id = MessageId("existing")
    frozen_existing = freeze_message(
        {
            "id": existing_id,
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "source": {"kind": "model", "provider": "test", "model": "test"},
        }
    )
    assert frozen_existing["id"] == existing_id
    assert is_frozen(frozen_existing)

    asst = create_assistant_message(
        {
            "content": [{"type": "text", "text": "answer"}],
            "source": {
                "provider": "test-provider",
                "model": "test-model",
                "replayState": {"request": 1},
            },
        }
    )
    assert asst["role"] == "assistant"
    assert asst["source"] == {
        "kind": "model",
        "provider": "test-provider",
        "model": "test-model",
        "replayState": {"request": 1},
    }
    asst_kw = create_assistant_message(
        content=[{"type": "text", "text": "hi"}],
        source={"replayState": {"r": 2}},
        provider="p1",
        model="m1",
    )
    assert asst_kw["source"] == {
        "kind": "model",
        "replayState": {"r": 2},
        "provider": "p1",
        "model": "m1",
    }

    call_id = ToolCallId("call-1")
    tool_res = create_tool_result_message(
        {
            "callId": call_id,
            "content": [{"type": "text", "text": "result"}],
            "isError": False,
        }
    )
    assert tool_res["role"] == "tool"
    assert tool_res["toolCallId"] == call_id
    assert tool_res["isError"] is False
    assert tool_res["source"] == {"kind": "tool", "callId": call_id}

    tool_kw = create_tool_result_message(
        call_id="c2", content=[{"type": "text", "text": "err"}], is_error=True
    )
    assert tool_kw["isError"] is True
    assert tool_kw["toolCallId"] == "c2"
