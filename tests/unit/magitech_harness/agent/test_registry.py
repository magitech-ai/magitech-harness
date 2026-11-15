"""Unit tests for AgentRegistry lifecycle, entry/announcement, and factory seam."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from anima import Context, FiberState, Service
from anima.core import symbols
from anima.reflect import _BoundService

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.agent.registry import AgentRegistry
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.store import Session


class _StubAgent:
    def __init__(self, agent_id: str, session: Session | None = None) -> None:
        self.id = agent_id
        self.options: dict[str, Any] = {}
        self.session = session or Session.create(agent_id)
        self.status = "idle"
        self.ctx = Context()


async def test_registry_waits_for_session_projections_and_reads_turn_boundaries() -> (
    None
):
    ctx = Context()
    fiber = ctx.use(AgentRegistry)
    await asyncio.sleep(0)
    assert ctx.get("agents") is None
    assert fiber.state is FiberState.PENDING

    await ctx.use(SessionProjectionRegistry)
    await fiber

    session = Session.create("turn-positions")
    assert ctx.agents.turn_boundary_of(session) == {
        "openTurnStartSeq": None,
        "lastStepStartSeq": None,
        "lastStepBoundary": None,
        "lastTurn": 0,
    }
    turn = session.append("turn/start", {"turn": 3})
    step = session.append("step/start", {"turn": 3, "step": 1})
    assert ctx.agents.turn_boundary_of(session) == {
        "openTurnStartSeq": turn.seq,
        "lastStepStartSeq": step.seq,
        "lastStepBoundary": {"kind": "start", "seq": step.seq},
        "lastTurn": 3,
    }
    session.append("turn/end", {"turn": 3, "reason": {"kind": "completed"}})
    assert ctx.agents.turn_boundary_of(session)["openTurnStartSeq"] is None

    row = ctx.sessionProjections.checkpoint(session)["turnBoundary"]
    assert row["val"]["lastTurn"] == 3

    await fiber.dispose()
    session.append("turn/start", {"turn": 4})
    # After disposal, projection unit is released: old session keeps its checkpoint,
    # new session gets none.
    assert ctx.sessionProjections.checkpoint(session)["turnBoundary"] == row
    assert "turnBoundary" not in ctx.sessionProjections.checkpoint(
        Session.create("after-unload")
    )


def test_registry_fails_loud_without_session_projections() -> None:
    ctx = Context()
    svc = AgentRegistry(ctx)
    with pytest.raises(RuntimeError, match="sessionProjections"):
        svc.turn_boundary_of(Session.create("missing"))
    with pytest.raises(RuntimeError, match="sessionProjections"):
        asyncio.run(svc.init())


async def test_registers_entries_emits_lifecycle_events_and_unregisters_on_dispose() -> (
    None
):
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    lifecycle: list[str] = []
    ctx.on("agent/created", lambda p: lifecycle.append(f"created:{p['agent'].id}"))
    ctx.on("agent/disposed", lambda p: lifecycle.append(f"disposed:{p['agent'].id}"))

    agent = _StubAgent("a1")
    dispose = await ctx.agents.register(agent)
    assert ctx.agents.get(agent.id) is agent
    assert ctx.agents.list() == [agent]

    with pytest.raises(ValueError, match="already registered"):
        await ctx.agents.register(_StubAgent("a1"))

    dispose()
    assert ctx.agents.get(agent.id) is None
    assert ctx.agents.list() == []
    assert lifecycle == ["created:a1", "disposed:a1"]


async def test_rejects_mismatched_agent_and_session_ids() -> None:
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    agent = _StubAgent("agent-id", session=Session.create("session-id"))

    with pytest.raises(
        ValueError, match='agent id "agent-id" does not match session id "session-id"'
    ):
        ctx.agents.enter(agent)
    assert ctx.agents.list() == []


async def test_rolls_back_entry_and_pairs_disposed_when_created_listener_throws() -> (
    None
):
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    lifecycle: list[str] = []
    ctx.on("agent/created", lambda p: lifecycle.append(f"created:{p['agent'].id}"))

    def _veto(_payload: dict[str, Any]) -> None:
        raise RuntimeError("creation veto")

    ctx.on("agent/created", _veto)
    ctx.on("agent/disposed", lambda p: lifecycle.append(f"disposed:{p['agent'].id}"))

    with pytest.raises(RuntimeError, match="creation veto"):
        await ctx.agents.register(_StubAgent("vetoed"))
    assert ctx.agents.get("vetoed") is None
    assert lifecycle == ["created:vetoed", "disposed:vetoed"]


async def test_rolls_back_async_created_rejection_and_contains_disposed_failures() -> (
    None
):
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    heard: list[str] = []

    async def _async_veto(_payload: dict[str, Any]) -> None:
        raise RuntimeError("created async")

    def _disposed_sync(_payload: dict[str, Any]) -> None:
        raise RuntimeError("disposed sync")

    async def _disposed_async(_payload: dict[str, Any]) -> None:
        raise RuntimeError("disposed async")

    ctx.on("agent/created", _async_veto)
    ctx.on("agent/created", lambda _p: heard.append("unreachable"))
    ctx.on("agent/disposed", _disposed_sync)
    ctx.on("agent/disposed", _disposed_async)
    ctx.on("agent/disposed", lambda p: heard.append(p["agent"].id))

    with pytest.raises(RuntimeError, match="created async"):
        await ctx.agents.register(_StubAgent("contained"))
    await asyncio.sleep(0)

    assert heard == ["contained"]
    assert ctx.agents.list() == []
    messages = "\n".join(
        " ".join(str(a) for a in msg.args) for msg in ctx.logger.buffer
    )
    assert 'agent "contained": agent/disposed listener threw' in messages
    assert "disposed sync" in messages
    assert 'agent "contained": agent/disposed listener rejected' in messages
    assert "disposed async" in messages


async def test_awaits_each_created_listener_and_rejects_reentrant_announce() -> None:
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    entered = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def _first(_payload: dict[str, Any]) -> None:
        order.append("first:start")
        entered.set()
        await release.wait()
        order.append("first:end")

    ctx.on("agent/created", _first)
    ctx.on("agent/created", lambda _p: order.append("second"))

    agent = _StubAgent("serial")

    async def _do_reg() -> Any:
        res = await ctx.agents.register(agent)
        order.append("ready")
        return res

    task = asyncio.create_task(_do_reg())
    await entered.wait()
    assert order == ["first:start"]
    with pytest.raises(ValueError, match="already announced"):
        await ctx.agents.announce(agent, "startup")
    release.set()
    dispose = await task
    assert order == ["first:start", "first:end", "second", "ready"]
    dispose()


async def test_separates_entry_from_announcement_and_guards_stale_detach() -> None:
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    lifecycle: list[str] = []
    ctx.on("agent/created", lambda p: lifecycle.append(f"created:{p['agent'].id}"))
    ctx.on("agent/disposed", lambda p: lifecycle.append(f"disposed:{p['agent'].id}"))

    first = _StubAgent("split")
    detach_first = ctx.agents.enter(first)
    assert lifecycle == []
    sig = AbortSignal()
    seen_signals: list[AbortSignal | None] = []
    ctx.on("agent/created", lambda p: seen_signals.append(p.get("signal")))
    await ctx.agents.announce(first, "startup", sig)
    assert seen_signals == [sig]

    with pytest.raises(ValueError, match="already announced"):
        await ctx.agents.announce(first, "startup")

    detach_first()
    detach_first()

    replacement = _StubAgent("split")
    detach_replacement = ctx.agents.enter(replacement)
    detach_first()
    assert ctx.agents.get(replacement.id) is replacement
    with pytest.raises(ValueError, match="not live"):
        await ctx.agents.announce(first, "startup")

    # Never announced -> no agent/disposed emitted for replacement
    detach_replacement()
    assert lifecycle == ["created:split", "disposed:split"]


async def test_defers_detach_requested_by_created_listener_until_dispatch_settles() -> (
    None
):
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    order: list[str] = []
    agent = _StubAgent("reentrant")
    detach = ctx.agents.enter(agent)

    def _on_first(_p: dict[str, Any]) -> None:
        order.append(f"first:{ctx.agents.get(agent.id) is agent}")
        detach()
        order.append(f"after-detach:{ctx.agents.get(agent.id) is agent}")

    ctx.on("agent/created", _on_first)
    ctx.on(
        "agent/created",
        lambda _p: order.append(f"second:{ctx.agents.get(agent.id) is agent}"),
    )
    ctx.on("agent/disposed", lambda _p: order.append("disposed"))

    await ctx.agents.announce(agent, "startup")
    assert order == ["first:True", "after-detach:True", "second:True", "disposed"]
    assert ctx.agents.get(agent.id) is None


async def test_factory_seam_delegates_through_calling_context_and_canonicalizes() -> (
    None
):
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)

    with pytest.raises(RuntimeError, match="no agent factory"):
        await ctx.agents.create({"sessionId": "s"})

    calls: dict[str, list[tuple[Context, Any, dict[str, Any]]]] = {
        "create": [],
        "resume": [],
    }

    class _PlainFactory:
        async def create_agent(
            self, owner_ctx: Context, options: dict[str, Any]
        ) -> dict[str, Any]:
            calls["create"].append((owner_ctx, owner_ctx.fiber, options))

            async def _dispose() -> None:
                return None

            return {"agent": _StubAgent(options["sessionId"]), "dispose": _dispose}

        async def resume(
            self, owner_ctx: Context, options: dict[str, Any]
        ) -> dict[str, Any]:
            calls["resume"].append((owner_ctx, owner_ctx.fiber, options))

            async def _dispose() -> None:
                return None

            return {
                "agent": _StubAgent(options["resumeSessionId"]),
                "dispose": _dispose,
            }

    factory = _PlainFactory()
    dispose_factory = ctx.agents.set_factory(factory)

    caller_contexts: list[Context] = []
    caller_fibers: list[Any] = []

    async def _consumer(inner: Context) -> None:
        caller_contexts.append(inner)
        caller_fibers.append(inner.fiber)
        await inner.agents.create({"sessionId": "create-s"})
        await inner.agents.resume({"resumeSessionId": "resume-s"})

    _consumer.inject = ["agents"]  # type: ignore[attr-defined]
    await ctx.use(_consumer)

    assert calls["create"][0][0] is caller_contexts[0]
    assert calls["create"][0][1] is caller_fibers[0]
    assert calls["resume"][0][0] is caller_contexts[0]
    assert calls["resume"][0][1] is caller_fibers[0]

    dispose_factory()
    with pytest.raises(RuntimeError, match="no agent factory"):
        await ctx.agents.create({"sessionId": "after"})


async def test_factory_seam_rejects_second_factory_and_clears_on_owner_dispose() -> (
    None
):
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)

    class _Factory:
        async def create_agent(
            self, _ctx: Context, options: dict[str, Any]
        ) -> dict[str, Any]:
            return {"agent": _StubAgent(options["sessionId"]), "dispose": lambda: None}

        async def resume(
            self, _ctx: Context, options: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "agent": _StubAgent(options["resumeSessionId"]),
                "dispose": lambda: None,
            }

    def _owner(inner: Context) -> None:
        inner.agents.set_factory(_Factory())
        with pytest.raises(RuntimeError, match="already registered"):
            inner.agents.set_factory(_Factory())

    _owner.inject = ["agents"]  # type: ignore[attr-defined]
    owner_fiber = await ctx.use(_owner)
    assert await ctx.agents.create({"sessionId": "before-s"}) is not None
    await owner_fiber.dispose()
    with pytest.raises(RuntimeError, match="no agent factory"):
        await ctx.agents.create({"sessionId": "after-s"})


async def test_factory_seam_canonicalizes_bound_service() -> None:
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)
    seen: list[tuple[str, Any]] = []

    class _ServiceFactory(Service):
        def __init__(self, inner: Context) -> None:
            super().__init__(inner, "traced_factory")

        async def create_agent(
            self, owner_ctx: Context, options: dict[str, Any]
        ) -> dict[str, Any]:
            seen.append(("create", owner_ctx))
            return {"agent": _StubAgent(options["sessionId"]), "dispose": lambda: None}

        async def resume(
            self, owner_ctx: Context, options: dict[str, Any]
        ) -> dict[str, Any]:
            seen.append(("resume", owner_ctx))
            return {
                "agent": _StubAgent(options["resumeSessionId"]),
                "dispose": lambda: None,
            }

    await ctx.use(_ServiceFactory)
    child = ctx.extend()
    bound = child.traced_factory
    assert isinstance(bound, _BoundService)
    ctx.agents.set_factory(bound)  # type: ignore[arg-type]

    consumer = ctx.extend()
    await consumer.agents.create({"sessionId": "c1"})
    await consumer.agents.resume({"resumeSessionId": "r1"})
    assert seen == [("create", consumer), ("resume", consumer)]


async def test_factory_seam_canonicalizes_original_proxy_attribute() -> None:
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(AgentRegistry)

    class _InnerFactory:
        async def create_agent(
            self, _ctx: Context, options: dict[str, Any]
        ) -> dict[str, Any]:
            return {"agent": _StubAgent(options["sessionId"]), "dispose": lambda: None}

        async def resume(
            self, _ctx: Context, options: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "agent": _StubAgent(options["resumeSessionId"]),
                "dispose": lambda: None,
            }

    inner = _InnerFactory()
    proxy: dict[Any, Any] = {}

    class _Proxy:
        pass

    wrapped = _Proxy()
    setattr(wrapped, symbols["original"], inner)
    _ = proxy
    ctx.agents.set_factory(wrapped)  # type: ignore[arg-type]
    created = await ctx.agents.create({"sessionId": "orig-1"})
    assert created["agent"].id == "orig-1"
