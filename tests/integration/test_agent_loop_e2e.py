"""End-to-end integration tests for AgentLoop through create_harness and JSONL persistence."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from magitech_harness import create_harness
from magitech_harness.agent.agent_loop import AgentLoop
from magitech_harness.agent.registry import AgentRegistry
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.service import LlmAdapter
from magitech_harness.llm.types import GenerateOptions, LlmModelInfo, StreamChunk
from magitech_harness.tools import define_tool


class _ScriptedE2EAdapter(LlmAdapter):
    """Stateful LlmAdapter driving tool call + multi-turn + resumed turn."""

    def __init__(self, requests_seen: list[GenerateOptions]) -> None:
        self.requests_seen = requests_seen
        self.call_count = 0

    def list_models(self, provider: str) -> list[LlmModelInfo]:
        return [{"id": "m", "name": "M", "provider": provider}]

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        self.requests_seen.append(options)
        self.call_count += 1
        if self.call_count == 1:
            yield {
                "type": "tool-call-delta",
                "index": 0,
                "id": "call_mul_1",
                "name": "multiply",
                "argumentsDelta": '{"a": 6, "b": 7}',
            }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
            return
        if self.call_count == 2:
            yield {"type": "text-delta", "index": 0, "text": "Product is 42."}
            yield {"type": "finish", "reason": {"kind": "stop"}}
            return
        if self.call_count == 3:
            yield {"type": "text-delta", "index": 0, "text": "Second turn done."}
            yield {"type": "finish", "reason": {"kind": "stop"}}
            return
        yield {"type": "text-delta", "index": 0, "text": "Resumed turn done."}
        yield {"type": "finish", "reason": {"kind": "stop"}}


@pytest.mark.asyncio
async def test_agent_loop_e2e_create_tools_multiturn_flush_and_resume(
    tmp_path: Path,
) -> None:
    ctx1 = create_harness(storage_dir=tmp_path)
    await ctx1.flush()
    loader = ctx1.get("loader", None, strict=False)
    assert loader is not None
    await loader.disable("session-title-first-prompt")

    assert isinstance(ctx1.agentLoop, AgentLoop)
    assert isinstance(ctx1.agents, AgentRegistry)

    executed_tools: list[tuple[int, int, int]] = []

    def multiply(args: dict[str, Any]) -> dict[str, int]:
        a = int(args["a"])
        b = int(args["b"])
        product = a * b
        executed_tools.append((a, b, product))
        return {"product": product}

    ctx1.tools.register(
        define_tool(
            {
                "name": "multiply",
                "description": "Multiply two integers.",
                "parameters": {
                    "a": {"type": "integer", "required": True},
                    "b": {"type": "integer", "required": True},
                },
                "output": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "integer", "required": True}
                        },
                        "additionalProperties": False,
                    },
                    "render": lambda out: json.dumps(out),
                },
                "execute": multiply,
            }
        ),
        {"when": lambda _c: True},
    )

    requests_seen: list[GenerateOptions] = []
    adapter1 = _ScriptedE2EAdapter(requests_seen)
    ctx1.llm.register_adapter(["e2e"], adapter1)

    handle1 = await ctx1.agents.create(
        {
            "sessionId": "e2e-agent-1",
            "meta": {"cwd": str(tmp_path)},
            "agentOptions": {"provider": "e2e", "model": "m"},
        }
    )
    agent1 = handle1["agent"]

    # Turn 1: user message triggers tool call -> multiply(6, 7) -> final assistant response
    agent1.followup(
        create_user_message(
            content=[{"type": "text", "text": "Multiply 6 by 7."}],
            source={"kind": "user"},
        )
    )
    await agent1.when_idle()
    assert executed_tools == [(6, 7, 42)]
    assert adapter1.call_count == 2

    # Turn 2: multi-turn follow-up on the same live agent
    agent1.followup(
        create_user_message(
            content=[{"type": "text", "text": "And confirm."}],
            source={"kind": "user"},
        )
    )
    await agent1.when_idle()
    assert adapter1.call_count == 3

    event_types_1 = [ev.type for ev in agent1.session.events]
    assert "tool/call" in event_types_1
    assert "tool/result" in event_types_1
    assert event_types_1.count("turn/end") == 2

    await handle1["dispose"]()
    await ctx1.fiber.dispose()

    # Reboot harness from same storage_dir and resume persisted session cleanly
    ctx2 = create_harness(storage_dir=tmp_path)
    await ctx2.flush()
    loader2 = ctx2.get("loader", None, strict=False)
    assert loader2 is not None
    await loader2.disable("session-title-first-prompt")

    resumed_requests: list[GenerateOptions] = []
    adapter2 = _ScriptedE2EAdapter(resumed_requests)
    adapter2.call_count = 3
    ctx2.llm.register_adapter(["e2e"], adapter2)

    handle2 = await ctx2.agents.resume(
        {
            "resumeSessionId": "e2e-agent-1",
            "agentOptions": {"provider": "e2e", "model": "m"},
        }
    )
    agent2 = handle2["agent"]
    assert ctx2.agents.turn_boundary_of(agent2.session)["lastTurn"] == 2

    agent2.followup(
        create_user_message(
            content=[{"type": "text", "text": "Turn 3 after resume."}],
            source={"kind": "user"},
        )
    )
    await agent2.when_idle()
    assert adapter2.call_count == 4
    assert ctx2.agents.turn_boundary_of(agent2.session)["lastTurn"] == 3
    derived = agent2.session.derive_messages()
    assert any(
        isinstance(part, dict) and "Resumed turn done." in str(part.get("text", ""))
        for msg in derived
        if msg.get("role") == "assistant"
        for part in msg.get("content", [])
    )

    await handle2["dispose"]()
    await ctx2.fiber.dispose()
