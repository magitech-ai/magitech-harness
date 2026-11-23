"""Unit tests for AgentRegistry process-local initiator attribution and draining."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from anima import Context

from magitech_harness.agent.registry import AgentRegistry
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from tests._mount import mount


class _StubAgent:
    def __init__(self, agent_id: str) -> None:
        self.id = agent_id


async def _harness() -> tuple[Context, AgentRegistry, Any]:
    ctx = Context()
    await mount(ctx, SessionProjectionRegistry)
    fiber = ctx.use(AgentRegistry)
    await fiber.load()
    return ctx, ctx.agents, fiber


async def test_absent_and_required_initiator() -> None:
    _ctx, service, fiber = await _harness()
    assert service.current_initiator() is None
    with pytest.raises(RuntimeError, match="no initiating agent is active"):
        service.require_initiator()
    await fiber.dispose()


async def test_preserves_sync_and_async_return_identities_across_await() -> None:
    _ctx, service, fiber = await _harness()
    initiator = _StubAgent("identity")
    sentinel = {"result": True}

    def _sync() -> dict[str, bool]:
        assert service.require_initiator() is initiator
        return sentinel

    assert service.with_initiator(initiator, _sync) is sentinel

    async def _async_op() -> dict[str, bool]:
        assert service.require_initiator() is initiator
        await asyncio.sleep(0)
        assert service.require_initiator() is initiator
        return sentinel

    assert await service.with_initiator(initiator, _async_op) is sentinel

    # Pre-built Future keeps exact identity
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[dict[str, bool]] = loop.create_future()
    ret = service.with_initiator(initiator, lambda: fut)
    assert ret is fut
    fut.set_result(sentinel)
    assert await ret is sentinel

    assert service.current_initiator() is None
    await fiber.dispose()


async def test_isolates_overlapping_async_initiators() -> None:
    _ctx, service, fiber = await _harness()
    a = _StubAgent("a")
    b = _StubAgent("b")
    both_started = asyncio.Event()
    release = asyncio.Event()
    starts = 0

    async def _run(initiator: _StubAgent) -> None:
        nonlocal starts
        assert service.require_initiator() is initiator
        starts += 1
        if starts == 2:
            both_started.set()
        await release.wait()
        assert service.require_initiator() is initiator

    t1 = asyncio.create_task(service.with_initiator(a, lambda: _run(a)))
    t2 = asyncio.create_task(service.with_initiator(b, lambda: _run(b)))
    await both_started.wait()
    assert service.current_initiator() is None
    release.set()
    await asyncio.gather(t1, t2)
    await fiber.dispose()


async def test_restores_nested_and_explicitly_cleared_boundaries() -> None:
    _ctx, service, fiber = await _harness()
    parent = _StubAgent("parent")
    child = _StubAgent("child")

    def _outer() -> None:
        assert service.require_initiator() is parent

        def _inner() -> None:
            assert service.require_initiator() is child

        service.with_initiator(child, _inner)
        assert service.require_initiator() is parent

        def _cleared() -> None:
            assert service.current_initiator() is None
            with pytest.raises(RuntimeError, match="no initiating agent is active"):
                service.require_initiator()

        service.without_initiator(_cleared)
        assert service.require_initiator() is parent

    service.with_initiator(parent, _outer)
    assert service.current_initiator() is None
    await fiber.dispose()


async def test_restores_parent_after_sync_and_async_errors() -> None:
    _ctx, service, fiber = await _harness()
    parent = _StubAgent("parent")
    child = _StubAgent("child")

    def _sync_fail() -> None:
        raise RuntimeError("sync failure")

    def _outer() -> None:
        with pytest.raises(RuntimeError, match="sync failure"):
            service.with_initiator(child, _sync_fail)
        assert service.require_initiator() is parent

    service.with_initiator(parent, _outer)

    async def _async_fail() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("async failure")

    with pytest.raises(RuntimeError, match="async failure"):
        await service.with_initiator(child, _async_fail)
    assert service.current_initiator() is None
    await fiber.dispose()


async def test_stops_new_boundaries_drains_active_and_invalidates_after_dispose() -> (
    None
):
    ctx, service, fiber = await _harness()
    initiator = _StubAgent("draining")
    release = asyncio.Event()
    started = asyncio.Event()

    async def _op() -> None:
        started.set()
        await release.wait()
        assert service.require_initiator() is initiator

    pending = asyncio.create_task(service.with_initiator(initiator, _op))
    await started.wait()

    disposal = asyncio.create_task(fiber.dispose())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    disposal_dup = asyncio.create_task(fiber.dispose())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="agent initiator scope is disposed"):
        service.with_initiator(initiator, lambda: 1)
    with pytest.raises(RuntimeError, match="agent initiator scope is disposed"):
        service.without_initiator(lambda: 1)
    assert not disposal.done()
    assert ctx.get("agents") is None

    release.set()
    await pending
    await disposal
    await disposal_dup

    with pytest.raises(RuntimeError, match="agent initiator scope is disposed"):
        service.current_initiator()
    with pytest.raises(RuntimeError, match="agent initiator scope is disposed"):
        service.require_initiator()


async def test_does_not_self_deadlock_on_sync_or_nested_dispose() -> None:
    _ctx, service, fiber = await _harness()
    parent = _StubAgent("parent-disposer")
    child = _StubAgent("child-disposer")

    async def _dispose_chain() -> None:
        await service.with_initiator(
            parent, lambda: service.with_initiator(child, fiber.dispose)
        )

    await asyncio.wait_for(_dispose_chain(), timeout=1.0)
    with pytest.raises(RuntimeError, match="agent initiator scope is disposed"):
        service.current_initiator()


async def test_excludes_async_teardown_initiator_while_draining_unrelated() -> None:
    _ctx, service, fiber = await _harness()
    initiator = _StubAgent("async-disposer")
    unrelated = _StubAgent("unrelated")
    started = asyncio.Event()
    release = asyncio.Event()

    async def _unrelated_op() -> None:
        started.set()
        await release.wait()
        assert service.require_initiator() is unrelated

    pending = asyncio.create_task(service.with_initiator(unrelated, _unrelated_op))
    await started.wait()

    async def _teardown_op() -> None:
        await asyncio.sleep(0)
        await fiber.dispose()

    returned = asyncio.create_task(service.with_initiator(initiator, _teardown_op))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not returned.done()

    release.set()
    await pending
    await asyncio.wait_for(returned, timeout=1.0)
    with pytest.raises(RuntimeError, match="agent initiator scope is disposed"):
        service.current_initiator()


async def test_custom_awaitable_and_custom_task_completion_drains_cleanly() -> None:
    _ctx, service, fiber = await _harness()
    initiator = _StubAgent("custom-await")

    class _CustomAwaitable:
        def __await__(self) -> Any:
            async def _impl() -> str:
                assert service.require_initiator() is initiator
                return "ok"

            return _impl().__await__()

    assert await service.with_initiator(initiator, _CustomAwaitable) == "ok"

    # Also test without_initiator with async function
    async def _cleared_async() -> Any:
        await asyncio.sleep(0)
        return service.current_initiator()

    assert await service.without_initiator(_cleared_async) is None
    await fiber.dispose()
