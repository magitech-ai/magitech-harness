"""Unit tests for LLM types."""

from magitech_harness.llm.types import ModelChunk, ModelResponse
from magitech_harness.session.events import ToolCallData


def test_model_response_defaults() -> None:
    """Verifies default values and serialization of ModelResponse."""
    resp = ModelResponse()
    assert resp.text == ""
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.raw_response == {}


def test_model_response_with_tool_calls() -> None:
    """Verifies ModelResponse with tool calls."""
    tc = ToolCallData(call_id="call_99", name="grep", arguments={"pattern": "test"})
    resp = ModelResponse(
        text="Searching...", tool_calls=[tc], finish_reason="tool_calls"
    )
    assert resp.text == "Searching..."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "grep"
    assert resp.finish_reason == "tool_calls"


def test_model_chunk_defaults_and_fields() -> None:
    """Verifies default values and custom fields of ModelChunk."""
    chunk = ModelChunk()
    assert chunk.text == ""
    assert chunk.tool_calls == []
    assert chunk.finish_reason is None

    tc = ToolCallData(call_id="call_chunk", name="bash", arguments={})
    chunk2 = ModelChunk(text="delta", tool_calls=[tc], finish_reason="stop")
    assert chunk2.text == "delta"
    assert len(chunk2.tool_calls) == 1
    assert chunk2.finish_reason == "stop"
