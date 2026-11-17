"""Integration tests for the pi-ai provider against the harness stream seam."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from magitech_harness import create_harness
from magitech_harness.llm.pi_ai.adapter import PiAiAdapter
from magitech_harness.llm.pi_ai.config import resolve_profiles


@pytest.mark.asyncio
async def test_end_to_end_pi_ai_streaming_turn(tmp_path: Path) -> None:
    """Verifies a complete harness stream turn through the pi-ai adapter."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices":[{"delta":{"content":"Run"}}]}\n'
                "\n"
                'data: {"choices":[{"delta":{"content":" echo"}}]}\n'
                "\n"
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
                "\n"
                'data: {"usage":{"prompt_tokens":2,"completion_tokens":1,'
                '"total_tokens":3}}\n'
                "\n"
                "data: [DONE]\n"
                "\n"
            ),
        )

    ctx = create_harness(storage_dir=tmp_path / "sessions")
    await ctx.flush()
    profiles = resolve_profiles(
        {"providers": {"groq": {"apiKeyEnv": "PI_AI_TEST_KEY"}}}
    )
    adapter = PiAiAdapter(
        profiles=lambda: profiles,
        client_factory=lambda _profile: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
    )
    ctx.llm.register_adapter(["groq"], adapter)

    os.environ["PI_AI_TEST_KEY"] = "sk-test"
    try:
        streamed = [
            chunk
            async for chunk in ctx.llm.stream(
                {
                    "provider": "groq",
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                    ],
                }
            )
        ]
    finally:
        os.environ.pop("PI_AI_TEST_KEY", None)

    assert streamed
    text_events = [c for c in streamed if c.get("type") == "text-delta"]
    assert any("echo" in str(ev.get("text")) for ev in text_events)
    finish = streamed[-1]
    assert finish["type"] == "finish"
    assert finish["reason"] == {"kind": "stop"}
    assert captured["body"]["model"] == "llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_end_to_end_pi_ai_credential_missing(tmp_path: Path) -> None:
    """Verifies a missing credential is surfaced as a terminal failure chunk."""
    ctx = create_harness(storage_dir=tmp_path / "sessions")
    await ctx.flush()
    profiles = resolve_profiles(
        {"providers": {"groq": {"apiKeyEnv": "PI_AI_TEST_KEY"}}}
    )
    adapter = PiAiAdapter(
        profiles=lambda: profiles,
        client_factory=lambda _profile: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        ),
    )
    ctx.llm.register_adapter(["groq"], adapter)

    streamed = [
        chunk
        async for chunk in ctx.llm.stream(
            {
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "messages": [],
            }
        )
    ]
    finish_events = [c for c in streamed if c.get("type") == "finish"]
    assert finish_events
    assert finish_events[-1]["reason"]["kind"] == "error"
