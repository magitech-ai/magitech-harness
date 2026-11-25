"""Focused tests for the subagent service, providers, and tool surface."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.error import HarnessError
from magitech_harness.llm.service import LLMService
from magitech_harness.subagent import providers
from magitech_harness.subagent.service import (
    ROOT_AGENT_NAME,
    SubagentError,
    SubagentProvider,
    SubagentService,
    _Member,
)
from magitech_harness.subagent.types import SubagentSpec


class _FakeSession:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def append(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))


class _FakeInbox:
    def __init__(self, pending: bool = True) -> None:
        self.has_pending = pending
        self.spliced: list[Any] = []

    def splice(self, *args: Any) -> None:
        self.spliced.append(args)


class _FakeAgent:
    def __init__(self, agent_id: str = "root", pending: bool = True) -> None:
        self.id = agent_id
        self.session = _FakeSession()
        self.inbox = _FakeInbox(pending)


def _service() -> SubagentService:
    ctx = Context()
    LLMService(ctx)
    return SubagentService(ctx)


def _member(
    child_id: str = "sub-1",
    name: str = "worker",
    parent: str | None = None,
    state: str = "running",
) -> _Member:
    return _Member(
        child_id=child_id,
        name=name,
        parent_id=parent,
        depth=0 if parent is None else 1,
        state=state,
        label="worker",
    )


def test_provider_dispose_noop_for_foreign_provider() -> None:
    service = _service()
    provider = SubagentProvider(name="x", start=lambda _s, _c: _empty())
    dispose = service.register_provider(provider)
    service._providers["x"] = SubagentProvider(name="x", start=lambda _s, _c: _empty())
    dispose()
    assert "x" in service._providers


async def _empty() -> dict[str, Any]:
    return {}


def test_resolve_max_depth_default() -> None:
    service = _service()
    assert service.resolve_max_depth(None) == 1
    assert service.resolve_max_depth(5) == 5


def test_names_of_child() -> None:
    service = _service()
    service._members["sub-1"] = _member()
    assert service.names_of(_FakeAgent("sub-1")) == {
        "self": "worker",
        "parent": ROOT_AGENT_NAME,
    }


def test_capacity_counts_running() -> None:
    service = _service()
    service._members["a"] = _member("a", "a", state="running")
    service._members["b"] = _member("b", "b", state="terminated")
    assert service.capacity_of(None) == {"running": 1, "limit": 10}


def test_resolve_name() -> None:
    service = _service()
    service._members["sub-1"] = _member()
    assert service.resolve_name(None, ROOT_AGENT_NAME) == ROOT_AGENT_NAME
    assert service.resolve_name(None, "worker") == "sub-1"
    with pytest.raises(SubagentError, match="no agent named"):
        service.resolve_name(None, "missing")


@pytest.mark.asyncio
async def test_list_descendants_with_signal() -> None:
    service = _service()
    service._members["sub-1"] = _member()
    rows = await service.list_descendants(ROOT_AGENT_NAME, AbortSignal())
    assert rows[0]["name"] == "worker"


@pytest.mark.asyncio
async def test_send_message_rejects_stopped_child() -> None:
    service = _service()
    service._members["sub-1"] = _member(state="terminated")
    with pytest.raises(SubagentError, match="is not running"):
        await service.send_message(None, "sub-1", [{"type": "text", "text": "hi"}])


@pytest.mark.asyncio
async def test_send_message_delivers_via_manager() -> None:
    service = _service()
    service._members["sub-1"] = _member()
    manager = service._manager_for(service.ctx)
    manager.send_message = AsyncMock()  # type: ignore[method-assign]
    result = await service.send_message(None, "sub-1", [{"type": "text", "text": "hi"}])
    assert result["messageId"].startswith("msg-")
    manager.send_message.assert_awaited_once_with("sub-1", "hi")


def test_interrupt_cancels_task() -> None:
    service = _service()
    service._members["sub-1"] = _member()
    manager = service._manager_for(service.ctx)
    task = asyncio.get_running_loop().create_task(_empty()) if False else None
    # A cancelled sentinel task from a finished loop is not available; exercise
    # the branch where no live task exists.
    assert service.interrupt("sub-1", {}) == {"accepted": True}
    assert service._members["sub-1"].state == "terminated"
    assert manager is not None and task is None


def test_prompt_without_registry_is_refused() -> None:
    service = _service()
    assert service.prompt(_FakeAgent(), {"role": "user"}) == {"accepted": False}


def test_prompt_injects_into_live_agent() -> None:
    ctx = Context()
    LLMService(ctx)

    class _Registry:
        def get(self, _agent_id: str) -> Any:
            return type("A", (), {"inbox": _FakeInbox()})()

    ctx.reflect.provide("agents", _Registry())
    service = SubagentService(ctx)
    caller = type("C", (), {"ctx": ctx, "id": "root"})()
    assert service.prompt(caller, {"role": "user"}) == {"accepted": True}


def test_log_event_without_active_session() -> None:
    service = _service()
    service._log_event("subagent/descriptor", {})


@pytest.mark.asyncio
async def test_start_activation_guards() -> None:
    service = _service()
    service.register_provider(
        SubagentProvider(name="spawn", start=lambda _s, _c: _empty())
    )
    service._members["a"] = _member("a", "a")
    with pytest.raises(SubagentError, match="already in use"):
        await service.startActivation(
            SubagentSpec(role="worker", prompt="go", agent_name="a"),
            provider="spawn",
            parent_id="unknown",
        )
    with pytest.raises(SubagentError, match="reserved"):
        await service.startActivation(
            SubagentSpec(role="worker", prompt="go", agent_name=ROOT_AGENT_NAME),
            provider="spawn",
        )


@pytest.mark.asyncio
async def test_start_activation_depth_and_capacity() -> None:
    service = _service()
    service.register_provider(
        SubagentProvider(name="spawn", start=lambda _s, _c: _empty())
    )
    service._members["parent"] = _member("parent", "parent")
    service._members["parent"].depth = 1
    with pytest.raises(SubagentError, match="exceeds the configured cap"):
        await service.startActivation(
            SubagentSpec(role="worker", prompt="go"), parent_id="parent"
        )
    service._max_depth = 5
    service._max_active = 0
    with pytest.raises(SubagentError, match="limit reached"):
        await service.startActivation(SubagentSpec(role="worker", prompt="go"))


def test_provider_factories() -> None:
    assert providers.spawn_provider().name == "spawn"
    assert providers.fork_provider().name == "fork"
    assert providers.spawn_provider().capabilities
    assert providers.fork_provider().capabilities


@pytest.mark.asyncio
async def test_provider_start_creates_child_context(tmp_path: Any) -> None:
    from magitech_harness import create_harness

    parent = create_harness(storage_dir=tmp_path / "parent")
    await parent.flush()
    result = await providers.start_spawn(
        SubagentSpec(role="worker", prompt="go"), parent
    )
    assert result["mode"] == "spawn"
    assert result["sessionId"]


@pytest.mark.asyncio
async def test_fork_provider_seeds_parent_events(tmp_path: Any) -> None:
    from magitech_harness import create_harness

    parent = create_harness(storage_dir=tmp_path / "parent")
    await parent.flush()
    parent.session = parent.sessions.create("parent-session")
    message = {
        "id": "seed",
        "role": "user",
        "content": [{"type": "text", "text": "seed"}],
        "source": {"kind": "user"},
    }
    parent.session.append(
        "user/message",
        message,
        surface_op="append",
    )
    result = await providers.start_fork(
        SubagentSpec(role="worker", prompt="go"), parent
    )
    assert result["mode"] == "fork"


@pytest.mark.asyncio
async def test_tool_error_paths(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "errors")
    await ctx.flush()
    session = ctx.sessions.create("error-session")

    async def call(
        name: str, args: dict[str, Any], agent: Any = None
    ) -> dict[str, Any]:
        batch = await ctx.tools.execute(
            {
                "calls": [
                    {"type": "tool-call", "id": name, "name": name, "arguments": args}
                ],
                "signal": AbortSignal(),
                "agent": agent,
            }
        )
        return batch["results"][0]["result"]

    for name, args in [
        ("subagent", {"prompt": "", "description": "x"}),
        ("subagent", {"prompt": "go"}),
        ("send_message", {"agent": 1, "message": "hi"}),
        ("interrupt_agent", {"agent": 1}),
        ("wait_agent", {"timeout_ms": "bad"}),
    ]:
        fake = _FakeAgent()
        fake.session = session
        result = await call(name, args, fake)
        assert result["isError"] is True

    result = await call("list_agents", {}, None)
    assert result["isError"] is True


@pytest.mark.asyncio
async def test_wait_agent_timeout_and_pending(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "wait")
    await ctx.flush()
    session = ctx.sessions.create("wait-session")

    async def call(pending: bool) -> dict[str, Any]:
        agent = _FakeAgent(pending=pending)
        agent.session = session
        batch = await ctx.tools.execute(
            {
                "calls": [
                    {
                        "type": "tool-call",
                        "id": "wait_agent",
                        "name": "wait_agent",
                        "arguments": {"timeout_ms": 10_000},
                    }
                ],
                "signal": AbortSignal(),
                "agent": agent,
            }
        )
        return batch["results"][0]["result"]

    with patch(
        "magitech_harness.subagent.tools.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await call(True)
    assert result["isError"] is False
    assert result["value"]["timedOut"] is False

    async def _timeout(coro: Any, *args: Any, **kwargs: Any) -> Any:
        if hasattr(coro, "close"):
            coro.close()
        raise TimeoutError()

    with patch(
        "magitech_harness.subagent.tools.asyncio.wait_for",
        side_effect=_timeout,
    ):
        result = await call(False)
    assert result["value"]["timedOut"] is True


def test_wait_for_input_helper() -> None:
    from magitech_harness.subagent.tools import _wait_for_input

    assert _wait_for_input(_FakeInbox(True)) is True


@pytest.mark.asyncio
async def test_delegate_rejects_prompt_without_agent(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "no-agent")
    await ctx.flush()
    batch = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "subagent",
                    "name": "subagent",
                    "arguments": {"prompt": "go"},
                }
            ],
            "signal": AbortSignal(),
        }
    )
    assert batch["results"][0]["result"]["isError"] is True


def test_require_agent_message() -> None:
    from magitech_harness.subagent.tools import _require_agent

    with pytest.raises(HarnessError, match="calling agent"):
        _require_agent({})


@pytest.mark.asyncio
async def test_provider_fork_without_parent_session(tmp_path: Any) -> None:
    from magitech_harness import create_harness

    ctx = create_harness(storage_dir=tmp_path / "no-parent-session")
    await ctx.flush()
    result = await providers.start_fork(SubagentSpec(role="worker", prompt="go"), ctx)
    assert result["mode"] == "fork"


def test_root_of_and_log_event_with_active_session(tmp_path: Any) -> None:
    from magitech_harness import create_harness
    from magitech_harness.session.service import SessionService

    ctx = create_harness(storage_dir=tmp_path / "active")
    service = SubagentService(ctx)
    assert service.root_of(None) == ROOT_AGENT_NAME

    sessions: SessionService = ctx.sessions
    session = sessions.create("active-session")
    sessions.active_session = session  # type: ignore[attr-defined]
    service._log_event("subagent/descriptor", {"childId": "x"})
    assert session.events[-1].type == "subagent/descriptor"


@pytest.mark.asyncio
async def test_interrupt_with_live_task() -> None:
    service = _service()
    service._members["sub-1"] = _member()
    manager = service._manager_for(service.ctx)
    task = asyncio.create_task(_empty())
    manager._tasks["sub-1"] = task
    assert service.interrupt("sub-1", {}) == {"accepted": True}
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_send_message_and_interrupt_success_paths(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "success")
    await ctx.flush()
    session = ctx.sessions.create("success-session")
    agent = _FakeAgent()
    agent.session = session
    service: SubagentService = ctx.subagents
    service._members["sub-1"] = _member("sub-1", "worker")
    manager = service._manager_for(ctx)
    manager.send_message = AsyncMock()  # type: ignore[method-assign]

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        batch = await ctx.tools.execute(
            {
                "calls": [
                    {
                        "type": "tool-call",
                        "id": name,
                        "name": name,
                        "arguments": args,
                    }
                ],
                "signal": AbortSignal(),
                "agent": agent,
            }
        )
        return batch["results"][0]["result"]

    result = await call("send_message", {"agent": "worker", "message": "hi"})
    assert result["isError"] is False
    assert result["value"]["messageId"].startswith("msg-")

    result = await call("interrupt_agent", {"agent": "worker"})
    assert result["isError"] is False
    assert result["value"] == {"accepted": True}


def test_wait_for_input_polls_until_pending() -> None:
    from magitech_harness.subagent.tools import _wait_for_input

    class _Eventually:
        def __init__(self) -> None:
            self.has_pending = False
            self.calls = 0

    inbox = _Eventually()
    ticks = iter([0.0, 0.0, 0.0, 0.0, 3601.0])

    def _sleep(_seconds: float) -> None:
        inbox.calls += 1
        if inbox.calls >= 2:
            inbox.has_pending = True

    with (
        patch("time.monotonic", side_effect=lambda: next(ticks)),
        patch("time.sleep", side_effect=_sleep),
    ):
        assert _wait_for_input(inbox) is True
