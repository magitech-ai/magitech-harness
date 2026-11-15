"""Unit tests for BlockAssembler."""

from __future__ import annotations

import pytest

from magitech_harness.llm.assembler import BlockAssembler
from magitech_harness.llm.brand import ToolCallId


def test_assembler_text_reasoning_and_tool_calls() -> None:
    asm = BlockAssembler()
    assert asm.finish == {"kind": "stop"}
    assert asm.usage is None
    assert asm.replay_state is None
    assert asm.replayState is None

    asm.push({"type": "block-start", "index": 0, "blockType": "reasoning"})
    asm.push(
        {"type": "block-start", "index": 0, "blockType": "reasoning"}
    )  # duplicate index ignored
    asm.push({"type": "reasoning-delta", "index": 0, "text": "think"})
    asm.push({"type": "reasoning-delta", "index": 0, "text": "ing"})
    asm.push(
        {"type": "text-delta", "index": 1, "text": "hello "}
    )  # missing block-start allowed
    asm.push(
        {
            "type": "block-end",
            "index": 1,
            "block": {"type": "text", "text": "hello world"},
        }
    )
    asm.push(
        {"type": "text-delta", "index": 1, "text": "straggler"}
    )  # ignored after block-end
    asm.push(
        {"type": "block-end", "index": 1, "block": {"type": "text", "text": "ignored"}}
    )
    asm.push(
        {
            "type": "tool-call-delta",
            "index": 2,
            "id": ToolCallId("c1"),
            "name": "bash",
            "argumentsDelta": '{"cmd":',
        }
    )
    asm.push(
        {
            "type": "tool-call-delta",
            "index": 2,
            "id": ToolCallId("c1"),
            "argumentsDelta": '"ls"}',
        }
    )
    asm.push(
        {
            "type": "tool-call-delta",
            "index": 3,
            "id": ToolCallId(""),
            "argumentsDelta": "{}",
        }
    )
    asm.push({"type": "usage", "usage": {"inputTokens": 10, "outputTokens": 5}})
    asm.push(
        {
            "type": "finish",
            "reason": {"kind": "tool-calls"},
            "replayState": {
                "response": {"r": 1},
                "blocks": [{"b": 0}, {"b": 1}, {"b": 2}, {"b": 3}],
            },
        }
    )

    assert asm.usage == {"inputTokens": 10, "outputTokens": 5}
    assert asm.finish == {"kind": "tool-calls"}
    blocks = asm.blocks()
    assert blocks == [
        {"type": "reasoning", "text": "thinking"},
        {"type": "text", "text": "hello world"},
        {"type": "tool-call", "id": "c1", "name": "bash", "arguments": '{"cmd":"ls"}'},
        {"type": "tool-call", "id": "call-3", "name": "", "arguments": "{}"},
    ]
    assert asm.replay_state == {
        "response": {"r": 1},
        "blocks": [{"b": 0}, {"b": 1}, {"b": 2}, {"b": 3}],
    }
    assert asm.replayState == asm.replay_state

    msg = asm.message({"provider": "p", "model": "m"})
    assert msg["role"] == "assistant"
    assert msg["content"] == blocks

    interrupted = asm.interrupted_blocks()
    assert interrupted == [
        {"type": "reasoning", "text": "thinking"},
        {"type": "text", "text": "hello world"},
    ]
    assert asm.interruptedBlocks() == interrupted


def test_assembler_max_tokens_drops_tool_calls_and_aligns_replay_state() -> None:
    asm = BlockAssembler()
    asm.push({"type": "block-start", "index": 0, "blockType": "text"})
    asm.push({"type": "text-delta", "index": 0, "text": "   "})
    asm.push(
        {
            "type": "block-end",
            "index": 1,
            "block": {
                "type": "tool-call",
                "id": ToolCallId("t1"),
                "name": "search",
                "arguments": "{}",
            },
        }
    )
    asm.push(
        {
            "type": "finish",
            "reason": {"kind": "max-tokens"},
            "replayState": {"response": {"r": 1}, "blocks": [{"idx": 0}, {"idx": 1}]},
        }
    )

    assert asm.blocks() == [{"type": "text", "text": "   "}]
    assert asm.replay_state == {"response": {"r": 1}, "blocks": [{"idx": 0}]}
    # Whitespace-only text block omitted from interrupted_blocks.
    assert asm.interrupted_blocks() == []

    # max-tokens with no tool calls leaves envelope intact; mismatched replay lengths return None.
    asm_no_tools = BlockAssembler()
    asm_no_tools.push({"type": "text-delta", "index": 0, "text": "ok"})
    asm_no_tools.push(
        {
            "type": "finish",
            "reason": {"kind": "max-tokens"},
            "replayState": {"response": {"r": 2}, "blocks": [{"idx": 0}]},
        }
    )
    assert asm_no_tools.replay_state == {"response": {"r": 2}, "blocks": [{"idx": 0}]}

    asm_mismatch = BlockAssembler()
    asm_mismatch.push({"type": "text-delta", "index": 0, "text": "ok"})
    asm_mismatch.push(
        {
            "type": "finish",
            "reason": {"kind": "stop"},
            "replayState": {"response": {"r": 3}, "blocks": []},
        }
    )
    assert asm_mismatch.replay_state is None


def test_assembler_rejects_unknown_chunk_and_incomplete_unknown_block() -> None:
    asm = BlockAssembler()
    with pytest.raises(ValueError, match="BlockAssembler.push"):
        asm.push({"type": "unknown"})

    asm_bad = BlockAssembler()
    asm_bad.push({"type": "block-start", "index": 0, "blockType": "custom"})
    with pytest.raises(
        ValueError, match='cannot assemble incomplete block of type "custom"'
    ):
        asm_bad.blocks()

    asm_closed = BlockAssembler()
    asm_closed.push(
        {"type": "block-end", "index": 0, "block": {"type": "text", "text": "done"}}
    )
    asm_closed.push(
        {
            "type": "tool-call-delta",
            "index": 0,
            "id": "ignored",
            "name": "tool",
            "argumentsDelta": "{}",
        }
    )
    assert asm_closed.blocks() == [{"type": "text", "text": "done"}]
