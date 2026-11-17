"""Unit tests for the Anthropic Messages protocol client."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import httpx
import pytest

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.catalog import ModelSpec
from magitech_harness.llm.pi_ai.protocols.anthropic_messages import (
    stream_anthropic_messages,
)


def _spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "provider": "anthropic",
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "api": "anthropic-messages",
        "base_url": "https://api.anthropic.com",
        "input": ("text",),
        "context_window": 200000,
        "max_tokens": 64000,
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
        async for chunk in stream_anthropic_messages(
            spec, options, api_key="sk-test", client=client
        )
    ]


@pytest.mark.asyncio
async def test_anthropic_messages_streams_text_thinking_and_tool_use() -> None:
    """Verifies the messages stream reaches the harness vocabulary."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "event: message_start\n"
                'data: {"type":"message_start","message":{"usage":'
                '{"input_tokens":9,"cache_read_input_tokens":2,'
                '"cache_creation_input_tokens":1}}}\n'
                "\n"
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":0,'
                '"content_block":{"type":"text","text":""}}\n'
                "\n"
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"Hello"}}\n'
                "\n"
                "event: content_block_stop\n"
                'data: {"type":"content_block_stop","index":0}\n'
                "\n"
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":1,'
                '"content_block":{"type":"thinking","thinking":""}}\n'
                "\n"
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":1,'
                '"delta":{"type":"thinking_delta","thinking":"think"}}\n'
                "\n"
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":2,'
                '"content_block":{"type":"tool_use","id":"tool-1","name":"bash"}}\n'
                "\n"
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":2,'
                '"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":1}"}}\n'
                "\n"
                "event: message_delta\n"
                'data: {"type":"message_delta","delta":{"stop_reason":'
                '"tool_use"},"usage":{"output_tokens":4}}\n'
                "\n"
                "event: message_stop\n"
                'data: {"type":"message_stop"}\n'
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
                        {"type": "reasoning", "text": "prior thought"},
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
                    "isError": True,
                    "content": [{"type": "text", "text": "failed"}],
                },
            ],
            "system": "top",
            "tools": [{"name": "bash", "description": "run", "parameters": {}}],
            "temperature": 0.4,
            "maxTokens": 32,
            "reasoningEffort": "high",
        },
        handler,
        captured,
    )
    types = [chunk["type"] for chunk in chunks]
    assert "text-delta" in types
    assert "reasoning-delta" in types
    assert "tool-call-delta" in types
    assert types[-1] == "finish"
    assert chunks[-1]["reason"] == {"kind": "tool-calls"}

    body = captured["body"]
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-haiku-4-5"
    assert body["stream"] is True
    assert body["max_tokens"] == 32
    assert body["temperature"] == 0.4
    assert body["system"] == "top\n\nsys\n\ndev"
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert body["tools"][0]["name"] == "bash"
    assert body["messages"][0] == {"role": "user", "content": "hi"}
    assert body["messages"][1]["content"][0]["type"] == "text"
    assert body["messages"][1]["content"][1]["type"] == "text"
    assert body["messages"][1]["content"][2]["type"] == "tool_use"
    assert body["messages"][2]["content"][0]["is_error"] is True

    usage = next(chunk for chunk in chunks if chunk["type"] == "usage")
    assert usage["usage"]["inputTokens"] == 9
    assert usage["usage"]["outputTokens"] == 4
    assert usage["usage"]["cacheReadTokens"] == 2
    assert usage["usage"]["cacheWriteTokens"] == 1


@pytest.mark.asyncio
async def test_anthropic_messages_uses_the_declared_version_path() -> None:
    """Verifies an endpoint already ending in the version keeps its path."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='event: message_stop\ndata: {"type":"message_stop"}\n\n',
        )

    await _stream(
        _spec(base_url="https://proxy.example/v1"),
        {"messages": []},
        handler,
        captured,
    )
    assert captured["url"] == "https://proxy.example/v1/messages"


@pytest.mark.asyncio
async def test_anthropic_messages_rejects_image_input() -> None:
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
async def test_anthropic_messages_maps_provider_failures() -> None:
    """Verifies a provider error becomes a harness failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    captured: dict[str, Any] = {}
    with pytest.raises(LlmError) as excinfo:
        await _stream(_spec(), {"messages": []}, handler, captured)
    assert excinfo.value.code == "AUTH"


@pytest.mark.asyncio
async def test_anthropic_messages_honours_an_aborted_caller() -> None:
    """Verifies an aborted caller stops the stream."""
    signal = AbortSignal()
    signal.abort()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='event: message_stop\ndata: {"type":"message_stop"}\n\n',
        )

    captured: dict[str, Any] = {}
    with pytest.raises(AbortError):
        await _stream(_spec(), {"messages": [], "signal": signal}, handler, captured)


@pytest.mark.asyncio
async def test_anthropic_messages_ignores_unparsable_frames() -> None:
    """Verifies malformed frames do not break the stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=("data: {oops\n\nevent: ping\ndata: {}\n\n"),
        )

    captured: dict[str, Any] = {}
    chunks = await _stream(_spec(), {"messages": []}, handler, captured)
    assert [chunk["type"] for chunk in chunks] == ["usage", "finish"]


@pytest.mark.asyncio
async def test_anthropic_messages_handles_history_and_extra_events() -> None:
    """Verifies assistant history, empty text, and extra event types."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "event: message_start\n"
                'data: {"type":"message_start","message":{"usage":'
                '{"input_tokens":4}}}\n'
                "\n"
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":0,'
                '"content_block":{"type":"tool_use","id":"t-1","name":"bash"}}\n'
                "\n"
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"input_json_delta",'
                '"partial_json":"{\\"a\\":1}"}}\n'
                "\n"
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":1,'
                '"content_block":{"type":"image"}}\n'
                "\n"
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":1,'
                '"delta":{"type":"weird_delta","text":"x"}}\n'
                "\n"
                "event: message_delta\n"
                'data: {"type":"message_delta","delta":{"stop_reason":'
                '"end_turn"},"usage":{}}\n'
                "\n"
                "event: message_stop\n"
                'data: {"type":"message_stop"}\n'
                "\n"
            ),
        )

    chunks = await _stream(
        _spec(),
        {
            "messages": [
                {"role": "system", "content": []},
                {"role": "developer", "content": []},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                {
                    "role": "assistant",
                    "content": [
                        123,
                        {"type": "text", "text": "earlier"},
                        {"type": "reasoning", "text": ""},
                        {
                            "type": "tool-call",
                            "id": "call-1",
                            "name": "bash",
                            "arguments": "{bad",
                        },
                        {
                            "type": "tool-call",
                            "id": "call-2",
                            "name": "read",
                            "arguments": {"path": "a"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "toolCallId": "call-1",
                    "content": [{"type": "text", "text": "ok"}],
                },
                {"role": "alien", "content": [{"type": "text", "text": "ignored"}]},
            ],
            "system": "",
        },
        handler,
        captured,
    )
    types = [chunk["type"] for chunk in chunks]
    assert "tool-call-delta" in types
    assert types[-1] == "finish"
    body = captured["body"]
    assert body["messages"][1]["content"][1]["type"] == "tool_use"
    assert body["messages"][1]["content"][1]["input"] == {"raw": "{bad"}
    assert body["messages"][2]["content"][0].get("is_error") is None


@pytest.mark.asyncio
async def test_anthropic_messages_owns_the_client_it_creates(
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
    # Drain to exhaustion: whatever error the mock transport raises after
    # close is irrelevant here -- the assertion is that the client closed.
    with contextlib.suppress(Exception):
        [
            chunk
            async for chunk in stream_anthropic_messages(
                _spec(), {"messages": []}, api_key="sk-test"
            )
        ]
    assert closed is True
