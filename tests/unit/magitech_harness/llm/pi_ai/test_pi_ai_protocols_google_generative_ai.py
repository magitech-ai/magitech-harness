"""Unit tests for the Google Generative AI protocol client."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import httpx
import pytest

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.catalog import ModelSpec
from magitech_harness.llm.pi_ai.protocols.google_generative_ai import (
    stream_google_generative_ai,
)


def _spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "provider": "google",
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "api": "google-generative-ai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "input": ("text",),
        "context_window": 1048576,
        "max_tokens": 65536,
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
        async for chunk in stream_google_generative_ai(
            spec, options, api_key="sk-test", client=client
        )
    ]


@pytest.mark.asyncio
async def test_google_generative_ai_streams_text_thoughts_and_calls() -> None:
    """Verifies the generative AI stream reaches the harness vocabulary."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}\n'
                "\n"
                'data: {"candidates":[{"content":{"parts":[{"text":'
                '"thought","thought":true}]}}]}\n'
                "\n"
                'data: {"candidates":[{"content":{"parts":[{"functionCall":'
                '{"name":"bash","args":{"command":"ls"}}}]}}]}\n'
                "\n"
                'data: {"candidates":[{"content":{"parts":[]},'
                '"finishReason":"STOP"}],"usageMetadata":'
                '{"promptTokenCount":5,"candidatesTokenCount":2,'
                '"totalTokenCount":7,"cachedContentTokenCount":1,'
                '"thoughtsTokenCount":3}}\n'
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
                    "toolName": "read",
                    "content": [{"type": "text", "text": "done"}],
                },
            ],
            "system": "top",
            "tools": [{"name": "bash", "description": "run", "parameters": {}}],
            "temperature": 0.3,
            "maxTokens": 16,
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
    assert chunks[-1]["reason"] == {"kind": "stop"}

    body = captured["body"]
    assert "streamGenerateContent" in captured["url"]
    assert captured["url"].endswith("alt=sse&key=sk-test")
    assert captured["headers"]["x-goog-api-key"] == "sk-test"
    assert body["systemInstruction"] == {"parts": [{"text": "top\n\nsys"}]}
    assert body["contents"][0] == {"role": "user", "parts": [{"text": "dev"}]}
    assert body["contents"][1] == {"role": "user", "parts": [{"text": "hi"}]}
    assert body["contents"][2]["role"] == "model"
    assert body["contents"][2]["parts"][1]["functionCall"]["name"] == "read"
    assert body["contents"][3]["parts"][0]["functionResponse"]["name"] == "read"
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "bash"
    assert body["generationConfig"]["temperature"] == 0.3
    assert body["generationConfig"]["maxOutputTokens"] == 16
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 4096}

    usage = next(chunk for chunk in chunks if chunk["type"] == "usage")
    assert usage["usage"]["inputTokens"] == 5
    assert usage["usage"]["outputTokens"] == 2
    assert usage["usage"]["cacheReadTokens"] == 1
    assert usage["usage"]["reasoningTokens"] == 3


@pytest.mark.asyncio
async def test_google_generative_ai_rejects_image_input() -> None:
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
async def test_google_generative_ai_maps_provider_failures() -> None:
    """Verifies a provider error becomes a harness failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad prompt"}})

    captured: dict[str, Any] = {}
    with pytest.raises(LlmError) as excinfo:
        await _stream(_spec(), {"messages": []}, handler, captured)
    assert excinfo.value.code == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_google_generative_ai_honours_an_aborted_caller() -> None:
    """Verifies an aborted caller stops the stream."""
    signal = AbortSignal()
    signal.abort()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"candidates":[{"content":{"parts":[{"text":"x"}]}}]}\n\n',
        )

    captured: dict[str, Any] = {}
    with pytest.raises(AbortError):
        await _stream(_spec(), {"messages": [], "signal": signal}, handler, captured)


@pytest.mark.asyncio
async def test_google_generative_ai_ignores_unparsable_frames() -> None:
    """Verifies malformed frames do not break the stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=('data: {oops\n\ndata: {"candidates":[]}\n\n'),
        )

    captured: dict[str, Any] = {}
    chunks = await _stream(_spec(), {"messages": []}, handler, captured)
    assert [chunk["type"] for chunk in chunks] == ["usage", "finish"]


@pytest.mark.asyncio
async def test_google_generative_ai_handles_history_and_extra_events() -> None:
    """Verifies assistant history, empty text, and extra event shapes."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"candidates":[{"content":{"parts":[{"text":""},'
                '{"foo":"bar"},123,{"text":"hi","thought":true},'
                '{"functionCall":{"name":"bash","args":"{bad"}},'
                '{"functionCall":{"name":"read","args":{"path":"a"}}}]}}]}\n'
                "\n"
                'data: {"candidates":[{"content":{"parts":"not-a-list"}}]}\n'
                "\n"
                'data: {"candidates":[{"content":{"parts":[]},'
                '"finishReason":"STOP"}],'
                '"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,'
                '"totalTokenCount":2}}\n'
                "\n"
            ),
        )

    chunks = await _stream(
        _spec(),
        {
            "messages": [
                {"role": "system", "content": []},
                {"role": "developer", "content": []},
                {
                    "role": "assistant",
                    "content": [
                        123,
                        {"type": "text", "text": ""},
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
                {"role": "alien", "content": [{"type": "text", "text": "ignored"}]},
            ],
            "system": "",
        },
        handler,
        captured,
    )
    types = [chunk["type"] for chunk in chunks]
    assert "reasoning-delta" in types
    assert "tool-call-delta" in types
    assert types[-1] == "finish"


@pytest.mark.asyncio
async def test_google_generative_ai_owns_the_client_it_creates(
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
            async for chunk in stream_google_generative_ai(
                _spec(), {"messages": []}, api_key=None
            )
        ]
    assert closed is True
