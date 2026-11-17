"""Unit tests for the shared pi-ai protocol helpers."""

from __future__ import annotations

import httpx
import pytest

from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.protocols.common import (
    StreamEmitter,
    _extract_error_message,
    _failure_code,
    build_headers,
    contains_image,
    finish_chunk,
    flatten_text,
    iter_sse_events,
    map_finish_reason,
    parse_json_data,
    raise_for_status,
    token_usage,
    usage_chunk,
)


def _response(lines: list[str]) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content="\n".join(lines)
    )


async def _collect(response: httpx.Response) -> list[dict[str, str]]:
    return [event async for event in iter_sse_events(response)]


def test_build_headers_carries_attribution_and_authentication() -> None:
    headers = build_headers(api_key="sk-test", auth_header="authorization")
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["content-type"] == "application/json"
    assert "user-agent" in headers
    raw = build_headers(
        api_key="sk-test", auth_header="x-api-key", auth_scheme="", extra={"x-a": "1"}
    )
    assert raw["x-api-key"] == "sk-test"
    assert raw["x-a"] == "1"
    headers = build_headers(api_key=None, auth_header="authorization")
    assert "authorization" not in headers


@pytest.mark.asyncio
async def test_iter_sse_events_parses_frames() -> None:
    response = _response(
        [
            ": keepalive",
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            "data: second",
            "data: third",
            "",
            "data: trailing",
        ]
    )
    events = await _collect(response)
    assert events == [
        {"event": "message_start", "data": '{"type":"message_start"}'},
        {"event": "message", "data": "second\nthird"},
        {"event": "message", "data": "trailing"},
    ]


@pytest.mark.asyncio
async def test_iter_sse_events_handles_blank_and_garbage_lines() -> None:
    response = _response(
        [
            "",
            "",
            "data: only",
            "",
            "garbage line",
            "",
            ":comment",
            "",
        ]
    )
    events = await _collect(response)
    assert events == [
        {"event": "message", "data": "only"},
    ]


def test_parse_json_data_handles_terminal_and_broken_frames() -> None:
    assert parse_json_data("") is None
    assert parse_json_data("[DONE]") is None
    assert parse_json_data("{oops") is None
    assert parse_json_data('{"a":1}') == {"a": 1}


def test_usage_and_finish_chunks() -> None:
    usage = token_usage(input_tokens=3, output_tokens=4, cache_read_tokens=1)
    assert usage["inputTokens"] == 3
    assert usage["outputTokens"] == 4
    assert usage["totalTokens"] == 7
    assert usage["cacheReadTokens"] == 1
    assert usage_chunk(usage) == {"type": "usage", "usage": usage}
    assert finish_chunk({"kind": "stop"}) == {
        "type": "finish",
        "reason": {"kind": "stop"},
    }


def test_token_usage_reports_reasoning_and_cache_writes() -> None:
    usage = token_usage(
        input_tokens=1,
        output_tokens=2,
        total_tokens=9,
        cache_write_tokens=4,
        reasoning_tokens=5,
    )
    assert usage["totalTokens"] == 9
    assert usage["cacheWriteTokens"] == 4
    assert usage["reasoningTokens"] == 5
    assert "cacheReadTokens" not in usage


def test_map_finish_reason_covers_the_provider_vocabulary() -> None:
    assert map_finish_reason(None) == {"kind": "stop"}
    assert map_finish_reason("stop") == {"kind": "stop"}
    assert map_finish_reason("end_turn") == {"kind": "stop"}
    assert map_finish_reason("STOP") == {"kind": "stop"}
    assert map_finish_reason("completed") == {"kind": "stop"}
    assert map_finish_reason("length") == {"kind": "max-tokens"}
    assert map_finish_reason("max_tokens") == {"kind": "max-tokens"}
    assert map_finish_reason("tool_use") == {"kind": "tool-calls"}
    assert map_finish_reason("tool_calls") == {"kind": "tool-calls"}
    assert map_finish_reason("SAFETY")["kind"] == "error"
    assert map_finish_reason("something-else") == {"kind": "stop"}


@pytest.mark.asyncio
async def test_raise_for_status_maps_provider_failures() -> None:
    ok = httpx.Response(200)
    await raise_for_status(ok)

    unauthorized = httpx.Response(
        401,
        json={"error": {"message": "bad key"}},
        headers={"x-request-id": "req-1"},
    )
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(unauthorized)
    assert excinfo.value.code == "AUTH"
    assert excinfo.value.failure["status"] == 401
    assert excinfo.value.failure["requestId"] == "req-1"

    rate_limited = httpx.Response(429, json={"error": {"message": "slow down"}})
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(rate_limited)
    assert excinfo.value.code == "RATE_LIMIT"

    too_large = httpx.Response(413, json={"error": {"message": "too big"}})
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(too_large)
    assert excinfo.value.code == "INVALID_REQUEST"

    invalid = httpx.Response(400, json={"error": {"message": "bad request"}})
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(invalid)
    assert excinfo.value.code == "INVALID_REQUEST"

    server = httpx.Response(500, text="boom")
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(server)
    assert excinfo.value.code == "SERVER"

    timeout = httpx.Response(504, text="gateway timeout")
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(timeout)
    assert excinfo.value.code == "SERVER"

    plain = httpx.Response(418, text="teapot")
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(plain)
    assert excinfo.value.code == "PROVIDER_ERROR"

    plain_message = httpx.Response(400, json={"message": "flat message"})
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(plain_message)
    assert "flat message" in excinfo.value.failure["message"]

    empty_body = httpx.Response(500)
    with pytest.raises(LlmError) as excinfo:
        await raise_for_status(empty_body)
    assert excinfo.value.code == "SERVER"


def test_flatten_text_and_image_detection() -> None:
    assert flatten_text("plain") == "plain"
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert flatten_text(content) == "ab"
    assert flatten_text([{"type": "image"}]) == ""
    assert flatten_text(None) == ""
    assert contains_image([{"type": "image"}]) is True
    assert contains_image([{"type": "text", "text": "x"}]) is False
    assert contains_image("text") is False


def test_stream_emitter_assembles_text_reasoning_and_tool_calls() -> None:
    emitter = StreamEmitter()
    assert emitter.text("") == []
    assert emitter.reasoning("") == []
    start = emitter.text("Hello")
    assert start[0] == {"type": "block-start", "index": 0, "blockType": "text"}
    assert start[1] == {"type": "text-delta", "index": 0, "text": "Hello"}
    delta = emitter.text(" world")
    assert delta == [{"type": "text-delta", "index": 0, "text": " world"}]
    reasoning = emitter.reasoning("think")
    assert reasoning[0]["blockType"] == "reasoning"
    assert reasoning[0]["index"] == 1
    call = emitter.tool_call("0", "call-1", "bash", '{"a":')
    assert call[0] == {"type": "block-start", "index": 2, "blockType": "tool-call"}
    assert call[1]["id"] == "call-1"
    assert call[1]["name"] == "bash"
    call += emitter.tool_call("0", "", "", "1}")
    assert call[-1]["argumentsDelta"] == "1}"
    final = emitter.finalize()
    assert final[0] == {
        "type": "block-end",
        "index": 0,
        "block": {"type": "text", "text": "Hello world"},
    }
    assert final[1]["block"] == {"type": "reasoning", "text": "think"}
    assert final[2]["block"] == {
        "type": "tool-call",
        "id": "call-1",
        "name": "bash",
        "arguments": '{"a":1}',
    }


def test_stream_emitter_replaces_final_tool_arguments() -> None:
    emitter = StreamEmitter()
    emitter.tool_call("0", "call-1", "bash", "")
    emitter.tool_call("0", "", "", '{"a":1}', replace=True)
    final = emitter.finalize()
    assert final[0]["block"]["arguments"] == '{"a":1}'


def test_stream_emitter_repeats_block_kinds() -> None:
    emitter = StreamEmitter()
    emitter.reasoning("one")
    second = emitter.reasoning("two")
    assert second == [{"type": "reasoning-delta", "index": 0, "text": "two"}]
    final = emitter.finalize()
    assert final[0]["block"] == {"type": "reasoning", "text": "onetwo"}


def test_stream_emitter_tool_call_replace_and_name() -> None:
    emitter = StreamEmitter()
    first = emitter.tool_call("0", "call-1", "bash", "", replace=True)
    assert first[1]["id"] == "call-1"
    second = emitter.tool_call("0", "", "", '{"a":1}', replace=True)
    assert second[-1]["argumentsDelta"] == '{"a":1}'
    third = emitter.tool_call("0", "", "", "x", replace=True)
    assert third[-1]["argumentsDelta"] == "x"
    final = emitter.finalize()
    assert final[0]["block"]["arguments"] == "x"


def test_stream_emitter_retains_tool_name() -> None:
    emitter = StreamEmitter()
    emitter.tool_call("0", "call-1", "bash", "")
    emitter.tool_call("0", "", "", "args")
    final = emitter.finalize()
    assert final[0]["block"]["name"] == "bash"


def test_stream_emitter_tool_call_without_name() -> None:
    emitter = StreamEmitter()
    emitter.tool_call("0", "call-1", "", "")
    emitter.tool_call("0", "", "", "args")
    final = emitter.finalize()
    assert final[0]["block"]["name"] == ""


def test_extract_error_message_covers_body_shapes() -> None:

    assert _extract_error_message({"error": {"message": "nested"}}) == "nested"
    assert _extract_error_message({"error": {"type": "t"}}) == "t"
    assert _extract_error_message({"error": {"type": "t"}, "message": "flat"}) == ("t")
    assert _extract_error_message({"error": {"message": ""}, "message": "flat"}) == (
        "flat"
    )
    assert _extract_error_message({"message": "top"}) == "top"
    assert _extract_error_message({"message": ""}) == "provider request failed"
    assert _extract_error_message("plain text") == "plain text"
    assert _extract_error_message(None) == "provider request failed"
    assert _extract_error_message(123) == "provider request failed"


def test_failure_code_covers_timeout() -> None:

    assert _failure_code(408, "upstream timeout") == "TIMEOUT"
    assert _failure_code(500, "internal error") == "SERVER"


def test_usage_chunk_carries_all_counters() -> None:
    usage = usage_chunk(
        token_usage(
            input_tokens=1,
            output_tokens=2,
            total_tokens=4,
            cache_read_tokens=3,
            cache_write_tokens=5,
            reasoning_tokens=7,
        )
    )
    assert usage == {
        "type": "usage",
        "usage": {
            "inputTokens": 1,
            "outputTokens": 2,
            "totalTokens": 4,
            "cacheReadTokens": 3,
            "cacheWriteTokens": 5,
            "reasoningTokens": 7,
        },
    }
