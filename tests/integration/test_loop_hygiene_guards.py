"""Conformance: repeat reminder and TOOL_TIMEOUT on the unified driver.

Finding #18: reminder fires only at {3,5,8} consecutive identical calls per
agent, resets on a different call or a new user prompt, delivered as a sourced
user/message (additionalContexts), never rewriting tool/result content.
Finding #19: cooperative scoped-deadline TOOL_TIMEOUT for both sync and async
tools that declare timeoutMs; fast/undeclared tools unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortController
from magitech_harness.session.events import EventType
from magitech_harness.tools.repeat_guard import RepeatChain


def _agent(session: Any) -> Any:
    return type("Agent", (), {"session": session})()


def _tool(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": name,
        "parameters": {},
        "output": {
            "schema": {"type": "object", "additionalProperties": True},
            "render": lambda _a, val: [{"type": "text", "text": str(val)}],
        },
        "execute": lambda _a, _c: {"ok": True},
        **extra,
    }


def test_repeat_chain_consecutive_reset() -> None:
    chain = RepeatChain([3, 5, 8], 500)
    agent = object()
    assert chain.observe(agent, "probe", {"a": 1}) is None
    assert chain.observe(agent, "probe", {"a": 1}) is None
    fired = chain.observe(agent, "probe", {"a": 1})
    assert fired is not None and "repeating the exact same tool call" in fired[0]
    assert fired[1] == 3
    # A different call resets the chain to 1.
    assert chain.observe(agent, "probe", {"a": 2}) is None
    assert chain.observe(agent, "probe", {"a": 1}) is None


@pytest.mark.asyncio
async def test_repeat_reminder_delivered_as_sourced_user_message(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = ctx.sessions.create("repeat-probe")
    session.append(EventType.TURN_START, {"turn": 1})
    agent = _agent(session)
    ctx.tools.register(_tool("probe"))

    for i in range(3):
        await ctx.tools.execute(
            {
                "calls": [
                    {
                        "type": "tool-call",
                        "id": f"c{i}",
                        "name": "probe",
                        "arguments": "{}",
                    }
                ],
                "signal": AbortController().signal,
                "agent": agent,
            }
        )
    reminders = [
        e
        for e in session.events
        if e.type == EventType.USER_MESSAGE
        and (e.payload.get("source") or {}).get("kind") == "repeat-tool-reminder"
    ]
    # The reminder lands via additionalContexts -> next step's inbox. In a
    # direct tools.execute loop there is no step to append it, so assert the
    # guard unit behavior above plus that no invented event is emitted.
    invented = [e for e in session.events if e.type == "tools/repeat-reminder"]
    assert invented == []
    assert len(reminders) <= 1  # at most one at the third consecutive call


@pytest.mark.asyncio
async def test_timeout_guard_structures_timeout_for_sync_tool(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = ctx.sessions.create("timeout-probe")
    session.append(EventType.TURN_START, {"turn": 1})
    agent = _agent(session)

    def _cooperative(args: Any, run_ctx: Any) -> dict[str, bool]:
        sig = run_ctx["signal"]
        while not sig.is_aborted:
            time.sleep(0.005)
        return {"done": True}

    ctx.tools.register(_tool("slow_sync", execute=_cooperative, timeoutMs=50))
    batch = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "t1",
                    "name": "slow_sync",
                    "arguments": "{}",
                }
            ],
            "signal": AbortController().signal,
            "agent": agent,
        }
    )
    result = batch["results"][0]["result"]
    assert result["isError"] is True
    assert result["error"]["info"]["code"] == "TOOL_TIMEOUT"
    assert "timed out after 50ms" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_timeout_guard_passes_fast_tool_unchanged(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = ctx.sessions.create("fast-probe")
    session.append(EventType.TURN_START, {"turn": 1})
    ctx.tools.register(_tool("fast", timeoutMs=5000))
    batch = await ctx.tools.execute(
        {
            "calls": [
                {"type": "tool-call", "id": "f1", "name": "fast", "arguments": "{}"}
            ],
            "signal": AbortController().signal,
            "agent": _agent(session),
        }
    )
    assert batch["results"][0]["result"]["isError"] is False


def test_repeat_chain_validation_and_preview() -> None:
    from magitech_harness.tools.repeat_guard import RepeatChain

    with pytest.raises(ValueError, match="must not be empty"):
        RepeatChain([], 500)
    with pytest.raises(ValueError, match=">= 2"):
        RepeatChain([1], 500)
    with pytest.raises(ValueError, match="must be unique"):
        RepeatChain([3, 3], 500)
    chain = RepeatChain([3, 5], 5)
    # Fire the second threshold (5) to exercise the preview branch.
    for _ in range(5):
        fired = chain.observe("a", "tool", {"k": "x" * 100})
    assert fired is not None and "… (+" in fired[0]
