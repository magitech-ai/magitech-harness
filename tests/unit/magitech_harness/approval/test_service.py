"""Unit tests for ApprovalService and canonical argument digests."""

import asyncio

import pytest
from anima import Context, EventMode

from magitech_harness.approval.service import (
    ApprovalRequest,
    ApprovalService,
    calculate_argument_digest,
)


def test_calculate_argument_digest_canonical_json() -> None:
    """Verifies that different key orderings produce identical sha256 digests."""
    d1 = {"b": 2, "a": 1, "nested": {"y": "val", "x": [1, 2]}}
    d2 = {"a": 1, "nested": {"x": [1, 2], "y": "val"}, "b": 2}

    h1 = calculate_argument_digest(d1)
    h2 = calculate_argument_digest(d2)

    assert h1.startswith("sha256:")
    assert h1 == h2

    d3 = {"a": 1, "b": 3}
    assert calculate_argument_digest(d3) != h1


@pytest.mark.asyncio
async def test_approval_service_default_verdict_and_setter() -> None:
    """Verifies default verdict behavior and property setter."""
    ctx = Context()
    service = ApprovalService(ctx, {"default_verdict": False})
    assert not service.default_verdict

    verdict = await service.request_approval("bash", {"command": "rm -rf /"})
    assert verdict is False

    service.default_verdict = True
    assert service.default_verdict
    # Clear cache to avoid reusing false verdict
    service.clear_cache()
    verdict2 = await service.request_approval("bash", {"command": "rm -rf /"})
    assert verdict2 is True


@pytest.mark.asyncio
async def test_approval_service_custom_handler() -> None:
    """Verifies custom async approval handler."""
    ctx = Context()
    service = ApprovalService(ctx)
    requests_received: list[ApprovalRequest] = []

    async def _handler(req: ApprovalRequest) -> bool:
        requests_received.append(req)
        return req.arguments.get("command") == "allowed"

    service.set_handler(_handler)

    v1 = await service.request_approval("bash", {"command": "allowed"})
    assert v1 is True

    service.clear_cache()
    v2 = await service.request_approval("bash", {"command": "disallowed"})
    assert v2 is False

    assert len(requests_received) == 2
    assert requests_received[0].digest.startswith("sha256:")


@pytest.mark.asyncio
async def test_approval_service_waterfall_override() -> None:
    """Verifies waterfall hook interception of approval requests."""
    ctx = Context()
    ctx.plugin(ApprovalService)
    await ctx.flush()

    assert ctx.has("approvals")

    async def guard_plugin(req: ApprovalRequest, next_fn):
        if req.tool_name == "safe_tool":
            return True
        return await next_fn(req)

    ctx.on("approval/request", guard_plugin, mode=EventMode.WATERFALL)

    v_safe = await ctx.approvals.request_approval("safe_tool", {"action": "read"})
    assert v_safe is True

    v_unsafe = await ctx.approvals.request_approval("unsafe_tool", {"action": "write"})
    assert v_unsafe is False


@pytest.mark.asyncio
async def test_approval_service_caching() -> None:
    """Verifies verdict caching and clear_cache."""
    ctx = Context()
    service = ApprovalService(ctx)

    digest = calculate_argument_digest({"key": "val"})
    assert service.get_cached(digest) is None

    service.remember(digest, True)
    assert service.get_cached(digest) is True

    # request_approval returns cached True without asking handler
    handler_called = []
    service.set_handler(lambda r: handler_called.append(True) or False)  # type: ignore[return-value]

    v = await service.request_approval("tool", {"key": "val"})
    assert v is True
    assert not handler_called

    service.clear_cache()
    assert service.get_cached(digest) is None


@pytest.mark.asyncio
async def test_approval_service_mutex_serialization() -> None:
    """Verifies that concurrent requests are serialized under the asyncio.Lock."""
    ctx = Context()
    service = ApprovalService(ctx)

    execution_order: list[str] = []

    async def slow_handler(req: ApprovalRequest) -> bool:
        name = req.arguments.get("task", "")
        execution_order.append(f"start:{name}")
        await asyncio.sleep(0.05)
        execution_order.append(f"end:{name}")
        return True

    service.set_handler(slow_handler)

    # Launch two requests concurrently
    t1 = asyncio.create_task(service.request_approval("tool", {"task": "1"}))
    t2 = asyncio.create_task(service.request_approval("tool", {"task": "2"}))

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1 is True
    assert r2 is True

    # Mutex ensures task 1 completes before task 2 starts
    assert execution_order == ["start:1", "end:1", "start:2", "end:2"]


@pytest.mark.asyncio
async def test_approval_service_cache_hit_after_acquiring_lock() -> None:
    """Verifies cache check inside lock handles race conditions where prior task cached verdict."""
    ctx = Context()
    service = ApprovalService(ctx)

    args = {"shared": "payload"}
    digest = calculate_argument_digest(args)

    # 1. Hold lock externally
    await service._lock.acquire()

    # 2. Start request, which passes pre-lock cache check and blocks waiting for lock
    task = asyncio.create_task(service.request_approval("tool", args))
    await asyncio.sleep(0.01)

    # 3. Populate cache while task is blocked on the lock
    service.remember(digest, True)

    # 4. Release lock to let task enter and hit the post-lock cache check
    service._lock.release()

    verdict = await task
    assert verdict is True
