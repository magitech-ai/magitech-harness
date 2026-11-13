"""Unit tests for AssistantStreamAccumulator."""

from magitech_harness.llm.assistant_stream import AssistantStreamAccumulator
from magitech_harness.llm.types import ModelChunk, UsageData
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
