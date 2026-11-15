"""Unit tests for turn archive admission on AgentRegistry."""

from __future__ import annotations

from typing import Any

from anima import Context

from magitech_harness.agent.archive_admission import install_turn_archive_admission
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.agent.registry import AgentRegistry
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.store import Session


class _LiveAgent:
    def __init__(self, ctx: Context, agent_id: str, status: str) -> None:
        self.id = agent_id
        self.options: dict[str, Any] = {}
        self.session = Session.create(agent_id)
        self.status = status
        self.ctx = ctx
        self.cancels: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    def cancel(
        self, cause: dict[str, Any], options: dict[str, Any] | None = None
    ) -> None:
        self.cancels.append((cause, options))


async def _register_live(ctx: Context, agent_id: str, initial: str) -> _LiveAgent:
    agent = _LiveAgent(ctx, agent_id, initial)
    await ctx.agents.register(agent)
    return agent


async def _ask(ctx: Context, session_id: str) -> list[dict[str, Any]]:
    return await ctx.waterfall(
        "workspace/session-activity",
        {"sessionId": session_id},
        default_handler=lambda *_args: [],
    )


async def _stop(ctx: Context, session_id: str) -> None:
    await ctx.parallel("workspace/session-stop", {"sessionId": session_id})


async def test_turn_archive_admission_reports_turn_only_when_running_and_prepends() -> (
    None
):
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    busy = await _register_live(ctx, "busy", "running")
    await _register_live(ctx, "quiet", "idle")

    async def _probe(_req: dict[str, Any], next_fn: Any) -> list[dict[str, Any]]:
        return [*(await next_fn()), {"kind": "probe"}]

    ctx.on("workspace/session-activity", _probe)

    assert await _ask(ctx, busy.id) == [{"kind": "turn"}, {"kind": "probe"}]
    assert await _ask(ctx, "quiet") == [{"kind": "probe"}]
    assert await _ask(ctx, "cold-or-unknown") == [{"kind": "probe"}]

    busy.status = "idle"
    assert await _ask(ctx, busy.id) == [{"kind": "probe"}]
    _ = AbortSignal()


async def test_turn_archive_admission_cancels_running_with_user_cause() -> None:
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    running = await _register_live(ctx, "stop-me", "running")
    idle = await _register_live(ctx, "already-idle", "idle")

    await _stop(ctx, running.id)
    await _stop(ctx, idle.id)
    await _stop(ctx, "nobody")

    assert running.cancels == [({"kind": "user"}, None)]
    assert idle.cancels == []


async def test_turn_archive_admission_stops_answering_after_registry_dispose() -> None:
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    fiber = ctx.use(AgentRegistry)
    await fiber
    agent = await _register_live(ctx, "watched", "running")
    assert await _ask(ctx, agent.id) == [{"kind": "turn"}]

    await fiber.dispose()
    assert await _ask(ctx, agent.id) == []
    await _stop(ctx, agent.id)
    assert agent.cancels == []


async def test_turn_archive_admission_manual_dispose_unregisters_hooks() -> None:
    ctx = Context()
    running = _LiveAgent(ctx, "manual-busy", "running")
    dispose = install_turn_archive_admission(
        ctx, lambda sid: running if sid == running.id else None
    )

    async def _transition_during_downstream(
        _req: dict[str, Any], next_fn: Any
    ) -> list[dict[str, Any]]:
        running.status = "idle"
        return await next_fn()

    dispose_mid = ctx.on("workspace/session-activity", _transition_during_downstream)
    assert await _ask(ctx, running.id) == [{"kind": "turn"}]
    dispose_mid()

    running.status = "running"
    assert await _ask(ctx, running.id) == [{"kind": "turn"}]
    dispose()
    assert await _ask(ctx, running.id) == []
    await _stop(ctx, running.id)
    assert running.cancels == []
