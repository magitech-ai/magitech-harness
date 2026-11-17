"""Unit tests for the OpenAI Responses protocol client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.catalog import ModelSpec
from magitech_harness.llm.pi_ai.protocols.openai_responses import (
    stream_openai_responses,
)


def _spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "provider": "openai",
        "id": "gpt-5",
        "name": "GPT-5",
        "api": "openai-responses",
        "base_url": "https://api.openai.com/v1",
        "input": ("text",),
        "context_window": 400000,
        "max_tokens": 128000,
        "reasoning": True,
        "thinking_level_map": {"high": "high"},
    }
    base.update(overrides)
    return ModelSpec(**base)


async def _stream(
    spec: ModelSpec,
    options: dict[str, Any],
    handler: Any,
    captured: dict[str, Any],
) -> list[dict[str, Any]]:
    def _handle(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handle))
    return [
        chunk
        async for chunk in stream_openai_responses(
            spec, options, api_key="sk-test", client=client
        )
    ]


@pytest.mark.asyncio
async def test_openai_responses_streams_text_reasoning_and_tool_calls() -> None:
    """Verifies the responses stream reaches the harness vocabulary."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "event: response.output_text.delta\n"
                'data: {"type":"response.output_text.delta","delta":"Hello"}\n'
                "\n"
                "event: response.reasoning_summary_text.delta\n"
                'data: {"type":"response.reasoning_summary_text.delta",'
                '"delta":"think"}\n'
                "\n"
                "event: response.output_item.added\n"
                'data: {"type":"response.output_item.added","output_index":0,'
                '"item":{"type":"function_call","call_id":"call-1",'
                '"name":"bash"}}\n'
                "\n"
                "event: response.function_call_arguments.delta\n"
                'data: {"type":"response.function_call_arguments.delta",'
                '"output_index":0,"delta":"{\\"command\\":\\"ls\\"}"}\n'
                "\n"
                "event: response.completed\n"
                'data: {"type":"response.completed","response":{"status":'
                '"completed","usage":{"input_tokens":10,"output_tokens":5,'
                '"total_tokens":15,"input_tokens_details":'
                '{"cached_tokens":2},"output_tokens_details":'
                '{"reasoning_tokens":3}}}}\n'
                "\n"
            ),
        )

    chunks = await _stream(
        _spec(),
        {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "developer", "content": [{"type": "text", "text": "dev"}]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "earlier"},
                        {
                            "type": "tool-call",
                            "id": "call-0",
                            "name": "read",
                            "arguments": '{"path":"a"}',
                        },
                    ],
                },
                {
                    "role": "tool",
                    "toolCallId": "call-0",
                    "content": [{"type": "text", "text": "done"}],
                },
            ],
            "system": "top",
            "tools": [{"name": "bash", "description": "run", "parameters": {}}],
            "temperature": 0.2,
            "maxTokens": 64,
            "reasoningEffort": "high",
        },
        handler,
        captured,
    )
    types = [chunk["type"] for chunk in chunks]
    assert types[0] == "block-start"
    assert "reasoning-delta" in types
    assert "tool-call-delta" in types
    assert types[-1] == "finish"
    assert chunks[-1]["reason"] == {"kind": "stop"}

    body = captured["body"]
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert body["model"] == "gpt-5"
    assert body["instructions"] == "top\n\nsys"
    assert body["stream"] is True
    assert body["temperature"] == 0.2
    assert body["max_output_tokens"] == 64
    assert body["reasoning"] == {"effort": "high"}
    assert body["input"][0] == {
        "role": "developer",
        "content": [{"type": "input_text", "text": "dev"}],
    }
    assert body["input"][1]["role"] == "user"
    assert body["input"][2]["type"] == "message"
    assert body["input"][3]["type"] == "function_call"
    assert body["input"][4]["type"] == "function_call_output"
    assert body["tools"][0]["name"] == "bash"
    assert captured["headers"]["authorization"] == "Bearer sk-test"

    usage = next(chunk for chunk in chunks if chunk["type"] == "usage")
    assert usage["usage"]["inputTokens"] == 10
    assert usage["usage"]["cacheReadTokens"] == 2
    assert usage["usage"]["reasoningTokens"] == 3


@pytest.mark.asyncio
async def test_openai_responses_reports_an_incomplete_response() -> None:
    """Verifies an incomplete response reports the truncation reason."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "event: response.incomplete\n"
                'data: {"type":"response.incomplete","response":'
                '{"status":"incomplete","incomplete_details":'
                '{"reason":"max_output_tokens"}}}\n'
                "\n"
            ),
        )

    chunks = await _stream(_spec(), {"messages": []}, handler, captured)
    assert chunks[-1] == {"type": "finish", "reason": {"kind": "max-tokens"}}


@pytest.mark.asyncio
async def test_openai_responses_reports_a_failed_response() -> None:
    """Verifies a failed response raises with the provider message."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "event: response.failed\n"
                'data: {"type":"response.failed","response":{"status":'
                '"failed","error":{"code":"server_error","message":"boom"}}}\n'
                "\n"
            ),
        )

    with pytest.raises(RuntimeError, match="boom"):
        await _stream(_spec(), {"messages": []}, handler, captured)


@pytest.mark.asyncio
async def test_openai_responses_rejects_image_input() -> None:
    """Verifies an image request fails loudly on this protocol."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    with pytest.raises(RuntimeError, match="image input unsupported"):
        await _stream(
            _spec(),
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "attachment": {}}],
                    }
                ]
            },
            handler,
            captured,
        )


@pytest.mark.asyncio
async def test_openai_responses_maps_provider_failures() -> None:
    """Verifies a provider error becomes a harness failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    captured: dict[str, Any] = {}
    with pytest.raises(LlmError) as excinfo:
        await _stream(_spec(), {"messages": []}, handler, captured)
    assert excinfo.value.code == "RATE_LIMIT"


@pytest.mark.asyncio
async def test_openai_responses_honours_an_aborted_caller() -> None:
    """Verifies an aborted caller stops the stream."""
    signal = AbortSignal()
    signal.abort()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"type":"response.output_text.delta","delta":"x"}\n\n',
        )

    captured: dict[str, Any] = {}
    with pytest.raises(AbortError):
        await _stream(_spec(), {"messages": [], "signal": signal}, handler, captured)


@pytest.mark.asyncio
async def test_openai_responses_ignores_unparsable_frames() -> None:
    """Verifies malformed frames do not break the stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=('data: {oops\n\ndata: {"type":"unknown"}\n\n'),
        )

    captured: dict[str, Any] = {}
    chunks = await _stream(_spec(), {"messages": []}, handler, captured)
    assert [chunk["type"] for chunk in chunks] == ["finish"]


@pytest.mark.asyncio
async def test_openai_responses_handles_history_and_extra_events() -> None:
    """Verifies assistant history, developer handling, and extra events."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "event: response.output_item.added\n"
                'data: {"type":"response.output_item.added","output_index":0,'
                '"item":{"type":"message"}}\n'
                "\n"
                "event: response.function_call_arguments.done\n"
                'data: {"type":"response.function_call_arguments.done",'
                '"output_index":1,"arguments":"{\\"a\\":1}"}\n'
                "\n"
                "event: response.failed\n"
                'data: {"type":"response.failed","response":{"status":'
                '"failed","error":"boom"}}\n'
                "\n"
            ),
        )

    with pytest.raises(RuntimeError, match="response failed"):
        await _stream(
            _spec(),
            {
                "messages": [
                    {"role": "system", "content": []},
                    {"role": "developer", "content": []},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "earlier"},
                            {
                                "type": "tool-call",
                                "id": "call-1",
                                "name": "bash",
                                "arguments": {"command": "ls"},
                            },
                            {"type": "reasoning", "text": "prior"},
                        ],
                    },
                    {"role": "alien", "content": [{"type": "text", "text": "ignored"}]},
                ],
                "system": "",
            },
            handler,
            captured,
        )
    body = captured["body"]
    assert body["input"][0]["type"] == "message"
    assert body["input"][1]["type"] == "function_call"
    assert body["input"][1]["arguments"] == '{"command": "ls"}'
    assert "instructions" not in body


@pytest.mark.asyncio
async def test_openai_responses_owns_the_client_it_creates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the protocol closes a client it creates."""
    closed = False

    class _Client(httpx.AsyncClient):
        async def aclose(self) -> None:
            nonlocal closed
            closed = True
            await super().aclose()

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return _Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    try:
        [
            chunk
            async for chunk in stream_openai_responses(
                _spec(), {"messages": []}, api_key="sk-test"
            )
        ]
    finally:
        pass
    assert closed is True


@pytest.mark.asyncio
async def test_openai_responses_skips_non_mapping_assistant_blocks() -> None:
    """Verifies a non-mapping assistant block is skipped during conversion."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"type":"response.completed","response":'
            '{"status":"completed"}}\n\n',
        )

    await _stream(
        _spec(),
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [123, {"type": "text", "text": "hi"}],
                }
            ],
        },
        handler,
        captured,
    )
    assert captured["body"]["input"][0]["type"] == "message"
