"""Unit tests for AssistantStreamAttempt frame emission and durable settlement."""

from __future__ import annotations

import pytest

from magitech_harness.agent.assistant_stream import AssistantStreamAttempt
from magitech_harness.agent.constants import DEFAULT_MAX_PARALLEL_TOOL_CALLS


def test_default_max_parallel_tool_calls() -> None:
    assert DEFAULT_MAX_PARALLEL_TOOL_CALLS == 10


def test_assistant_stream_attempt_lifecycle_commit_and_settlement() -> None:
    frames: list[dict[str, object]] = []
    revision = 0

    def next_rev() -> int:
        nonlocal revision
        revision += 1
        return revision

    attempt = AssistantStreamAttempt(
        "s-1",
        3,
        next_rev,
        1,
        2,
        lambda frame: frames.append(dict(frame)),
    )
    assert attempt.attempt_id == "s-1:3"
    assert attempt.attemptId == "s-1:3"
    assert attempt.ended is False
    assert attempt.finish == {"kind": "stop"}
    assert attempt.usage is None
    assert attempt.replay_state is None
    assert attempt.replayState is None

    attempt.start()
    attempt.push({"type": "text-delta", "index": 0, "text": "Hello "})
    attempt.push({"type": "text-delta", "index": 0, "text": "world"})
    attempt.push({"type": "usage", "usage": {"inputTokens": 4, "outputTokens": 2}})
    attempt.push(
        {
            "type": "finish",
            "reason": {"kind": "stop"},
            "replayState": {"response": {"r": 1}},
        }
    )

    assert attempt.blocks() == [{"type": "text", "text": "Hello world"}]
    assert attempt.interrupted_blocks() == [{"type": "text", "text": "Hello world"}]
    assert attempt.interruptedBlocks() == [{"type": "text", "text": "Hello world"}]
    assert attempt.usage == {"inputTokens": 4, "outputTokens": 2}
    assert attempt.finish == {"kind": "stop"}
    assert attempt.replay_state == {"response": {"r": 1}}
    assert len(attempt.stream) > 0

    attempt.settle("assistant/message", lambda: 7)
    assert attempt.ended is True
    assert frames[0] == {
        "type": "start",
        "attemptId": "s-1:3",
        "revision": 1,
        "turn": 1,
        "step": 2,
    }
    assert frames[1]["type"] == "chunk"
    assert frames[1]["index"] == 0
    assert frames[-1] == {
        "type": "end",
        "attemptId": "s-1:3",
        "revision": 6,
        "index": 4,
        "outcome": {"kind": "committed", "eventType": "assistant/message", "seq": 7},
    }


def test_assistant_stream_attempt_settle_failure_abandons_and_reraises() -> None:
    frames: list[dict[str, object]] = []
    revision = 0

    def next_rev() -> int:
        nonlocal revision
        revision += 1
        return revision

    attempt = AssistantStreamAttempt(
        "s-2",
        1,
        next_rev,
        1,
        1,
        lambda frame: frames.append(dict(frame)),
    )
    attempt.start()

    def failing_append() -> int:
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        attempt.settle("assistant/attempt", failing_append)

    assert attempt.ended is True
    assert frames[-1] == {
        "type": "end",
        "attemptId": "s-2:1",
        "revision": 2,
        "index": 0,
        "outcome": {"kind": "abandoned"},
    }
