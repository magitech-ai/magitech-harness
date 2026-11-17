"""Unit tests for the OpenAI Chat Completions protocol client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.catalog import ModelSpec
from magitech_harness.llm.pi_ai.protocols.openai_completions import (
    stream_openai_completions,
)


def _spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "provider": "groq",
        "id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B",
        "api": "openai-completions",
        "base_url": "https://api.groq.com/openai/v1",
        "input": ("text",),
        "context_window": 131072,
        "max_tokens": 131072,
        "reasoning": False,
    }
    base.update(overrides)
    return ModelSpec(**base)


def _client(handler: Any, captured: dict[str, Any]) -> httpx.AsyncClient:
    def _handle(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(_handle))


async def _stream(
    spec: ModelSpec,
    options: dict[str, Any],
    handler: Any,
    captured: dict[str, Any],
) -> list[dict[str, Any]]:
    client = _client(handler, captured)
    return [
        chunk
        async for chunk in stream_openai_completions(
            spec, options, api_key="sk-test", client=client
        )
    ]


@pytest.mark.asyncio
async def test_openai_completions_streams_text_and_tool_calls() -> None:
    """Verifies the chat completions stream reaches the harness vocabulary."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
                "\n"
                'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n'
                "\n"
                "data: "
                '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
                '"function":{"name":"bash","arguments":"{\\"command\\":\\"ls\\"}"}}]}}]}\n'
                "\n"
                'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n'
                "\n"
                'data: {"usage":{"prompt_tokens":3,"completion_tokens":4,'
                '"total_tokens":7,"completion_tokens_details":'
                '{"reasoning_tokens":2}}}\n'
                "\n"
                "data: [DONE]\n"
                "\n"
            ),
        )

    chunks = await _stream(
        _spec(),
        {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "earlier"},
                        {
                            "type": "tool-call",
                            "id": "call-0",
                            "name": "read",
                            "arguments": {"path": "a"},
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
            "tools": [
                {"name": "bash", "description": "run", "parameters": {"type": "object"}}
            ],
            "temperature": 0.5,
            "maxTokens": 128,
        },
        handler,
        captured,
    )
    types = [chunk["type"] for chunk in chunks]
    assert types[0] == "block-start"
    assert "text-delta" in types
    assert "reasoning-delta" in types
    assert "tool-call-delta" in types
    assert "usage" in types
    assert types[-1] == "finish"
    assert chunks[-1]["reason"] == {"kind": "tool-calls"}

    body = captured["body"]
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert body["model"] == "llama-3.1-8b-instant"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 128
    assert body["messages"][0] == {"role": "system", "content": "top"}
    assert body["messages"][1] == {"role": "system", "content": "sys"}
    assert body["messages"][2] == {"role": "user", "content": "hi"}
    assert body["messages"][3]["tool_calls"][0]["id"] == "call-0"
    assert body["messages"][3]["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "a"}'
    )
    assert body["messages"][4] == {
        "role": "tool",
        "tool_call_id": "call-0",
        "content": "done",
    }
    assert body["tools"][0]["function"]["name"] == "bash"
    assert captured["headers"]["authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_openai_completions_sends_reasoning_effort() -> None:
    """Verifies a reasoning request carries the wire effort spelling."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"})

    spec = _spec(reasoning=True, thinking_level_map={"high": "ultra"})
    await _stream(
        spec,
        {"messages": [], "reasoningEffort": "high"},
        handler,
        captured,
    )
    assert captured["body"]["reasoning_effort"] == "ultra"


@pytest.mark.asyncio
async def test_openai_completions_rejects_image_input() -> None:
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
async def test_openai_completions_maps_provider_failures() -> None:
    """Verifies a provider error becomes a harness failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    captured: dict[str, Any] = {}
    with pytest.raises(LlmError) as excinfo:
        await _stream(_spec(), {"messages": []}, handler, captured)
    assert excinfo.value.code == "AUTH"


@pytest.mark.asyncio
async def test_openai_completions_honours_an_aborted_caller() -> None:
    """Verifies an aborted caller stops the stream."""
    signal = AbortSignal()
    signal.abort()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
        )

    captured: dict[str, Any] = {}
    with pytest.raises(AbortError):
        await _stream(_spec(), {"messages": [], "signal": signal}, handler, captured)


@pytest.mark.asyncio
async def test_openai_completions_owns_the_client_it_creates() -> None:
    """Verifies the client the protocol creates is closed after the stream."""
    closed = False

    class _Client(httpx.AsyncClient):
        async def aclose(self) -> None:
            nonlocal closed
            closed = True
            await super().aclose()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
        )

    original = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        transport = httpx.MockTransport(handler)
        return _Client(transport=transport)

    httpx.AsyncClient = factory  # type: ignore[misc]
    try:
        chunks = [
            chunk
            async for chunk in stream_openai_completions(
                _spec(), {"messages": []}, api_key="sk-test"
            )
        ]
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]
    assert chunks
    assert closed is True


@pytest.mark.asyncio
async def test_openai_completions_ignores_unparsable_frames() -> None:
    """Verifies malformed frames do not break the stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "data: {oops\n"
                "\n"
                'data: {"choices":[]}\n'
                "\n"
                'data: {"choices":[{"delta":{}}]}\n'
                "\n"
            ),
        )

    captured: dict[str, Any] = {}
    chunks = await _stream(_spec(), {"messages": []}, handler, captured)
    assert [chunk["type"] for chunk in chunks] == ["finish"]


@pytest.mark.asyncio
async def test_openai_completions_handles_assistant_history_and_empty_system() -> None:
    """Verifies assistant history and an empty system message convert correctly."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
        )

    await _stream(
        _spec(),
        {
            "messages": [
                {"role": "system", "content": []},
                {"role": "developer", "content": []},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "only text"}],
                },
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
                        {
                            "type": "tool-call",
                            "id": "call-2",
                            "name": "read",
                            "arguments": '{"path":"a"}',
                        },
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
    assert body["messages"][0]["role"] == "assistant"
    assert body["messages"][0]["content"] == "only text"
    assert "tool_calls" not in body["messages"][0]
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"command": "ls"}'
    )
    assert body["messages"][1]["tool_calls"][1]["function"]["arguments"] == (
        '{"path":"a"}'
    )
    assert "system" not in body or body.get("system") is None


@pytest.mark.asyncio
async def test_openai_completions_skips_non_mapping_tool_call_deltas() -> None:
    """Verifies a malformed tool-call delta is skipped."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices":[{"delta":{"tool_calls":[123]}}]}\n'
                "\n"
                'data: {"choices":[{"delta":{"content":"x"}}]}\n'
                "\n"
            ),
        )

    captured: dict[str, Any] = {}
    chunks = await _stream(_spec(), {"messages": []}, handler, captured)
    types = [chunk["type"] for chunk in chunks]
    assert "tool-call-delta" not in types
    assert "text-delta" in types
