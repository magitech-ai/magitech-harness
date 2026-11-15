"""Unit tests for agent-scoped event dispatcher and context assembly."""

from __future__ import annotations

import asyncio
from typing import Any

from anima import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.agent.dispatch import (
    agent_carrier,
    agent_events,
    assemble_context_for,
    emit_agent_event,
)
from magitech_harness.scope import create_scope


class _StubAgent:
    def __init__(self, agent_id: str, ctx: Context) -> None:
        self.id = agent_id
        self.ctx = ctx


async def test_agent_events_emit_contains_sync_throws_and_async_rejections() -> None:
    ctx = Context()
    heard: list[str] = []
    agent = _StubAgent("event", ctx)

    def _sync_fail(_payload: dict[str, Any]) -> None:
        raise RuntimeError("sync listener")

    async def _async_fail(_payload: dict[str, Any]) -> None:
        raise RuntimeError("async listener")

    ctx.on("agent/status", _sync_fail)
    ctx.on("agent/status", _async_fail)
    ctx.on("agent/status", lambda p: heard.append(p["status"]))

    agent_events(ctx, agent).emit("agent/status", {"status": "running"})  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert heard == ["running"]
    messages = "\n".join(
        " ".join(str(a) for a in msg.args) for msg in ctx.logger.buffer
    )
    assert 'agent event "agent/status" listener threw' in messages
    assert "sync listener" in messages
    assert 'agent event "agent/status" listener rejected' in messages
    assert "async listener" in messages


async def test_agent_events_serial_passes_fused_subject() -> None:
    ctx = Context()
    agent = _StubAgent("serial-event", ctx)
    signal = AbortSignal()
    heard: list[tuple[Any, int, AbortSignal]] = []

    async def _on_stopping(payload: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        heard.append((payload["agent"], payload["turn"], payload["signal"]))

    ctx.on("agent/turn-stopping", _on_stopping)
    await agent_events(ctx, agent).serial(  # type: ignore[arg-type]
        "agent/turn-stopping", {"turn": 3, "signal": signal}
    )

    assert heard == [(agent, 3, signal)]


def test_agent_events_injects_subject_over_conflicting_payload_agent_and_filters_by_scope() -> (
    None
):
    ctx = Context()
    agent = _StubAgent("fused-subject", ctx)
    other = _StubAgent("other", ctx)
    scope = create_scope(ctx, agent)
    other_scope = create_scope(ctx, other)
    heard: list[Any] = []
    other_heard: list[Any] = []

    scope.ctx.on("agent/status", lambda p: heard.append(p["agent"]))
    other_scope.ctx.on("agent/status", lambda p: other_heard.append(p["agent"]))

    emit_agent_event(
        ctx,
        agent,  # type: ignore[arg-type]
        "agent/status",
        {"status": "running", "agent": other},
    )
    assert heard == [agent]
    assert other_heard == []


def test_agent_carrier_and_assemble_context_for() -> None:
    ctx = Context()
    agent = _StubAgent("ctx-helper", ctx)
    carrier = agent_carrier(agent)  # type: ignore[arg-type]
    assert agent_events(ctx, agent, carrier=carrier) is not None  # type: ignore[arg-type]

    without_signal = assemble_context_for(agent)  # type: ignore[arg-type]
    assert without_signal == {"agent": agent, "scope": agent}

    sig = AbortSignal()
    with_signal = assemble_context_for(agent, sig)  # type: ignore[arg-type]
    assert with_signal == {"agent": agent, "scope": agent, "signal": sig}


def test_agent_events_emit_without_running_loop_contains_async_rejection() -> None:
    ctx = Context()
    agent = _StubAgent("sync-emit", ctx)

    async def _async_fail(_payload: dict[str, Any]) -> None:
        raise RuntimeError("no-loop async boom")

    ctx.on("agent/status", _async_fail)
    agent_events(ctx, agent).emit("agent/status", {"status": "idle"})  # type: ignore[arg-type]

    messages = "\n".join(
        " ".join(str(a) for a in msg.args) for msg in ctx.logger.buffer
    )
    assert 'agent event "agent/status" listener rejected' in messages
    assert "no-loop async boom" in messages
