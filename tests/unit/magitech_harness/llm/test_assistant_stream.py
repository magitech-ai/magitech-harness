"""Unit tests for AssistantStreamAccumulator and compact stream helpers."""

from __future__ import annotations

import pytest

from magitech_harness.llm.assistant_stream import (
    AssistantStreamAccumulator,
    assemble_assistant_stream,
    assistant_stream_chunks,
    assistant_stream_first_token_time,
    assistant_stream_has_visible_content,
    assistant_stream_has_visible_text,
    chunk_has_visible_text,
    expand_assistant_stream,
    is_token_delta,
    is_visible_chunk,
    join_assistant_stream_text,
    last_assistant_stream_chunk,
    run_first_token_time,
    run_first_visible_time,
)
from magitech_harness.llm.types import ModelChunk, UsageData
from magitech_harness.llm.values import MAX_SAFE_INTEGER, deep_freeze
from magitech_harness.session.events import ToolCallData


def test_accumulator_records_single_chunk_records_in_order() -> None:
    accumulator = AssistantStreamAccumulator()
    assert not accumulator
    assert len(accumulator) == 0
    accumulator.push(ModelChunk(text="hello "))
    accumulator.push(
        ModelChunk(
            text="world",
            tool_calls=[ToolCallData(call_id="c1", name="t", arguments={})],
            usage=UsageData(total_tokens=3),
        )
    )
    assert accumulator
    assert len(accumulator) == 2
    records = accumulator.snapshot()
    assert [record["type"] for record in records] == ["chunk", "chunk"]
    assert isinstance(records[0]["time"], int)
    assert records[0]["chunk"]["text"] == "hello "
    assert records[1]["chunk"]["tool_calls"] == [
        {"call_id": "c1", "name": "t", "arguments": {}}
    ]
    assert records[1]["chunk"]["usage"]["total_tokens"] == 3


def test_accumulator_snapshot_is_detached() -> None:
    accumulator = AssistantStreamAccumulator()
    accumulator.push(ModelChunk(text="x"))
    snapshot = accumulator.snapshot()
    snapshot[0]["chunk"]["text"] = "mutated"
    assert accumulator.snapshot()[0]["chunk"]["text"] == "x"


def test_accumulator_compacts_parity_runs_and_expands_cleanly() -> None:
    acc = AssistantStreamAccumulator()
    assert acc.snapshot() == []

    # Compact reasoning-delta, text-delta (including overflow gap split), tool-call-delta, raw chunks.
    acc.push(
        {
            "time": 10,
            "chunk": {"type": "block-start", "index": 0, "blockType": "reasoning"},
        }
    )
    acc.push(
        {"time": 11, "chunk": {"type": "reasoning-delta", "index": 0, "text": "a"}}
    )
    acc.push(
        {"time": 15, "chunk": {"type": "reasoning-delta", "index": 0, "text": "b"}}
    )
    acc.push({"time": 20, "chunk": {"type": "text-delta", "index": 1, "text": "hi "}})
    acc.push({"time": 25, "chunk": {"type": "text-delta", "index": 1, "text": "there"}})
    acc.push(
        {
            "time": 30,
            "chunk": {
                "type": "tool-call-delta",
                "index": 2,
                "id": "c1",
                "name": "run",
                "argumentsDelta": '{"x":',
            },
        }
    )
    acc.push(
        {
            "time": 35,
            "chunk": {
                "type": "tool-call-delta",
                "index": 2,
                "id": "c1",
                "name": "run",
                "argumentsDelta": "1}",
            },
        }
    )
    # Empty id or empty name keeps raw chunk instead of compact run.
    acc.push(
        {
            "time": 36,
            "chunk": {
                "type": "tool-call-delta",
                "index": 3,
                "id": "",
                "argumentsDelta": "{}",
            },
        }
    )
    acc.push(
        {
            "time": 37,
            "chunk": {
                "type": "tool-call-delta",
                "index": 4,
                "id": "c2",
                "argumentsDelta": "{}",
            },
        }
    )
    acc.push(
        {
            "time": 38,
            "chunk": {"type": "usage", "usage": {"inputTokens": 1, "outputTokens": 2}},
        }
    )
    acc.push({"time": 40, "chunk": {"type": "finish", "reason": {"kind": "stop"}}})

    # Large negative jump that cannot keep safe integer gap starts a fresh run.
    acc_gap = AssistantStreamAccumulator()
    acc_gap.push(
        {
            "time": MAX_SAFE_INTEGER,
            "chunk": {"type": "text-delta", "index": 0, "text": "a"},
        }
    )
    acc_gap.push(
        {"time": -10, "chunk": {"type": "text-delta", "index": 0, "text": "b"}}
    )
    assert len(acc_gap.snapshot()) == 2

    records = acc.snapshot()
    expanded = expand_assistant_stream(records)
    assert [c["time"] for c in expanded] == [10, 11, 15, 20, 25, 30, 35, 36, 37, 38, 40]
    assert [c["chunk"]["type"] for c in expanded] == [
        "block-start",
        "reasoning-delta",
        "reasoning-delta",
        "text-delta",
        "text-delta",
        "tool-call-delta",
        "tool-call-delta",
        "tool-call-delta",
        "tool-call-delta",
        "usage",
        "finish",
    ]
    assert join_assistant_stream_text(records) == "hi there"
    assert last_assistant_stream_chunk(records, "finish") == {
        "type": "finish",
        "reason": {"kind": "stop"},
    }
    assert last_assistant_stream_chunk(records, "block-end") is None
    assert assistant_stream_chunks(records, "usage") == [
        {"type": "usage", "usage": {"inputTokens": 1, "outputTokens": 2}}
    ]

    asm = assemble_assistant_stream(records)
    assert [b["type"] for b in asm.blocks()] == [
        "reasoning",
        "text",
        "tool-call",
        "tool-call",
        "tool-call",
    ]


def test_stream_validation_errors_and_inspection_helpers() -> None:
    acc = AssistantStreamAccumulator()
    with pytest.raises(TypeError, match="safe integer"):
        acc.push({"time": 1.5, "chunk": {"type": "finish", "reason": {"kind": "stop"}}})
    with pytest.raises(TypeError, match="non-negative safe integer"):
        acc.push({"time": 0, "chunk": {"type": "text-delta", "index": -1, "text": "x"}})
    with pytest.raises(TypeError, match="losslessly JSON-serializable"):
        acc.push({"time": 0, "chunk": "bad"})
    with pytest.raises(TypeError, match="text must be a string"):
        acc.push({"time": 0, "chunk": {"type": "text-delta", "index": 0, "text": 1}})
    with pytest.raises(TypeError, match="id must be a string"):
        acc.push(
            {
                "time": 0,
                "chunk": {
                    "type": "tool-call-delta",
                    "index": 0,
                    "id": 1,
                    "argumentsDelta": "",
                },
            }
        )
    with pytest.raises(TypeError, match="name must be a string"):
        acc.push(
            {
                "time": 0,
                "chunk": {
                    "type": "tool-call-delta",
                    "index": 0,
                    "id": "c",
                    "name": 1,
                    "argumentsDelta": "",
                },
            }
        )
    with pytest.raises(TypeError, match="argumentsDelta must be a string"):
        acc.push(
            {
                "time": 0,
                "chunk": {
                    "type": "tool-call-delta",
                    "index": 0,
                    "id": "c",
                    "argumentsDelta": 1,
                },
            }
        )
    with pytest.raises(ValueError, match="AssistantStreamAccumulator.push"):
        acc.push({"time": 0, "chunk": {"type": "weird"}})

    with pytest.raises(TypeError, match="record must be an object"):
        expand_assistant_stream(["bad"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Unsupported Assistant stream record"):
        expand_assistant_stream([{"type": "weird"}])
    with pytest.raises(TypeError, match="must contain exactly"):
        expand_assistant_stream([{"type": "text-chunks", "time0": 0}])
    with pytest.raises(TypeError, match="string array"):
        expand_assistant_stream(
            [{"type": "text-chunks", "time0": 0, "index": 0, "dt": [], "texts": [1]}]
        )
    with pytest.raises(TypeError, match="non-empty"):
        expand_assistant_stream(
            [{"type": "text-chunks", "time0": 0, "index": 0, "dt": [], "texts": []}]
        )
    with pytest.raises(TypeError, match="dt must contain safe integers"):
        expand_assistant_stream(
            [
                {
                    "type": "text-chunks",
                    "time0": 0,
                    "index": 0,
                    "dt": ["x"],
                    "texts": ["a"],
                }
            ]
        )
    with pytest.raises(TypeError, match="dt length must be one less"):
        expand_assistant_stream(
            [{"type": "text-chunks", "time0": 0, "index": 0, "dt": [1], "texts": ["a"]}]
        )
    with pytest.raises(TypeError, match="member times must stay safe integers"):
        expand_assistant_stream(
            [
                {
                    "type": "text-chunks",
                    "time0": MAX_SAFE_INTEGER,
                    "index": 0,
                    "dt": [10],
                    "texts": ["a", "b"],
                }
            ]
        )
    with pytest.raises(TypeError, match="tool-call-chunks args must be non-empty"):
        expand_assistant_stream(
            [
                {
                    "type": "tool-call-chunks",
                    "time0": 0,
                    "index": 0,
                    "dt": [],
                    "id": "c",
                    "args": [],
                }
            ]
        )
    with pytest.raises(
        TypeError, match="tool-call-chunks id must be a non-empty string"
    ):
        expand_assistant_stream(
            [
                {
                    "type": "tool-call-chunks",
                    "time0": 0,
                    "index": 0,
                    "dt": [],
                    "id": "",
                    "args": ["{}"],
                }
            ]
        )
    with pytest.raises(
        TypeError, match="tool-call-chunks name must be a non-empty string"
    ):
        expand_assistant_stream(
            [
                {
                    "type": "tool-call-chunks",
                    "time0": 0,
                    "index": 0,
                    "dt": [],
                    "id": "c",
                    "name": None,
                    "args": ["{}"],
                }
            ]
        )
    with pytest.raises(TypeError, match="raw chunk must be a lossless JSON object"):
        expand_assistant_stream([{"type": "chunk", "time": 0, "chunk": "nope"}])
    with pytest.raises(TypeError, match="raw chunk must be a lossless JSON object"):
        expand_assistant_stream([{"type": "chunk", "time": 0, "chunk": {"x": -0.0}}])

    # Cover FrozenDict record validation cleanly too.
    assert (
        len(
            expand_assistant_stream(
                [
                    deep_freeze(
                        {
                            "type": "tool-call-chunks",
                            "time0": 0,
                            "index": 0,
                            "dt": [],
                            "id": "c1",
                            "args": ["{}"],
                        }
                    )
                ]
            )
        )
        == 1
    )

    # Visibility & timing helpers.
    assert is_token_delta({"type": "text-delta", "text": "a"})
    assert is_token_delta(
        {"type": "tool-call-delta", "argumentsDelta": "", "name": "f"}
    )
    assert not is_token_delta({"type": "usage", "usage": {}})

    assert is_visible_chunk({"type": "text-delta", "text": "ok"})
    assert not is_visible_chunk({"type": "block-start", "blockType": "text"})
    assert is_visible_chunk({"type": "block-start", "blockType": "image"})
    assert not is_visible_chunk({"type": "block-end", "block": {"type": "tool-call"}})
    assert is_visible_chunk({"type": "block-end", "block": {"type": "image"}})
    assert not is_visible_chunk({"type": "finish", "reason": {"kind": "stop"}})

    assert chunk_has_visible_text({"type": "text-delta", "text": "x"})
    assert chunk_has_visible_text(
        {"type": "block-end", "block": {"type": "text", "text": "x"}}
    )
    assert not chunk_has_visible_text({"type": "usage"})

    text_run = {
        "type": "text-chunks",
        "time0": 100,
        "index": 0,
        "dt": [5],
        "texts": ["  ", "hi"],
    }
    named_tc_run = {
        "type": "tool-call-chunks",
        "time0": 10,
        "index": 0,
        "dt": [],
        "id": "c",
        "name": "t",
        "args": [""],
    }
    anon_tc_run = {
        "type": "tool-call-chunks",
        "time0": 10,
        "index": 0,
        "dt": [],
        "id": "c",
        "args": [""],
    }
    assert run_first_token_time(text_run) == 100
    assert run_first_token_time(named_tc_run) == 10
    assert run_first_token_time(anon_tc_run) is None
    assert run_first_visible_time(text_run) == 105
    assert run_first_visible_time(named_tc_run) is None

    stream = [
        {
            "type": "chunk",
            "time": 1,
            "chunk": {"type": "usage", "usage": {"inputTokens": 0, "outputTokens": 0}},
        },
        text_run,
    ]
    assert assistant_stream_first_token_time(stream) == 100
    assert (
        assistant_stream_first_token_time(
            [
                {
                    "type": "chunk",
                    "time": 5,
                    "chunk": {"type": "text-delta", "index": 0, "text": "z"},
                }
            ]
        )
        == 5
    )
    assert assistant_stream_first_token_time([stream[0]]) is None
    assert assistant_stream_has_visible_content(stream)
    assert assistant_stream_has_visible_content(
        [
            {
                "type": "chunk",
                "time": 1,
                "chunk": {"type": "text-delta", "index": 0, "text": "v"},
            }
        ]
    )
    assert assistant_stream_has_visible_text(stream)
    assert assistant_stream_has_visible_text(
        [
            {
                "type": "chunk",
                "time": 1,
                "chunk": {"type": "text-delta", "index": 0, "text": "v"},
            }
        ]
    )
    assert not assistant_stream_has_visible_text(
        [{"type": "chunk", "time": 1, "chunk": {"type": "usage", "usage": {}}}]
    )
    assert assistant_stream_has_visible_content(
        [
            {
                "type": "chunk",
                "time": 1,
                "chunk": {
                    "type": "block-end",
                    "index": 0,
                    "block": {"type": "reasoning", "text": "thought"},
                },
            }
        ]
    )
    assert (
        join_assistant_stream_text(
            [
                {
                    "type": "chunk",
                    "time": 1,
                    "chunk": {"type": "text-delta", "index": 0, "text": "raw"},
                }
            ]
        )
        == "raw"
    )

    with pytest.raises(ValueError, match="assemble_assistant_stream"):
        assemble_assistant_stream([{"type": "weird"}])
