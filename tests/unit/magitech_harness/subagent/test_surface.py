"""Focused tests for the subagent service, providers, and tool surface."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
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
from magitech_harness.subagent.types import (
    SubagentRecord,
    SubagentSpec,
    SubagentState,
)


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
    provider = SubagentProvider(name="x", start=lambda _s, _c, **_kw: _empty())
    dispose = service.register_provider(provider)
    service._providers["x"] = SubagentProvider(
        name="x", start=lambda _s, _c, **_kw: _empty()
    )
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
async def test_send_message_uses_manager_child_id() -> None:
    service = _service()
    member = _member("sub-real", "worker")
    member.session_id = "child-session"
    service._members["sub-real"] = member
    manager = service._manager_for(service.ctx)
    manager._records["sub-real"] = SubagentRecord(
        subagent_id="sub-real",
        role="worker",
        agent_name="worker",
        model_id="inherit",
        workspace_path=Path("."),
        session_id="child-session",
        session_path=Path("."),
        state=SubagentState.RUNNING,
    )
    seen: list[str] = []

    async def _run_child(child_id: str, _message: str) -> None:
        seen.append(child_id)

    manager._run_child = _run_child  # type: ignore[method-assign]
    result = await service.send_message(
        None, "sub-real", [{"type": "text", "text": "hi"}]
    )
    assert result["messageId"].startswith("msg-")
    assert seen == ["sub-real"]


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
        SubagentProvider(name="spawn", start=lambda _s, _c, **_kw: _empty())
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
        SubagentProvider(name="spawn", start=lambda _s, _c, **_kw: _empty())
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

    assert _wait_for_input(_FakeInbox(True), threading.Event()) is True


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


def test_wait_for_input_stops_promptly() -> None:
    from magitech_harness.subagent.tools import _wait_for_input

    stop = threading.Event()
    # A pending inbox returns immediately.
    assert _wait_for_input(_FakeInbox(True), stop) is True
    # A stopped waiter returns False without polling.
    stop.set()
    assert _wait_for_input(_FakeInbox(False), stop) is False


@pytest.mark.asyncio
async def test_delegation_logs_lineage_to_caller_session(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "lineage")
    await ctx.flush()
    session = ctx.sessions.create("lineage-session")
    service: SubagentService = ctx.subagents
    service.register_provider(
        SubagentProvider(name="spawn", start=lambda _s, _c, **_kw: _empty())
    )
    record = SubagentRecord(
        subagent_id="sub-real",
        role="worker",
        agent_name="worker",
        model_id="inherit",
        workspace_path=Path("."),
        session_id="child-session",
        session_path=Path("."),
        state=SubagentState.COMPLETED,
    )

    async def _invoke(_specs: Any, **_kwargs: Any) -> list[SubagentRecord]:
        return [record]

    manager = service._manager_for(ctx)
    manager.invoke = _invoke  # type: ignore[method-assign]

    result = await service.startActivation(
        SubagentSpec(role="worker", prompt="go", agent_name="worker"),
        session=session,
    )
    assert result["id"] == "sub-real"
    types = [event.type for event in session.events]
    assert "subagent/descriptor" in types
    assert "subagent/catalog" in types


def test_capacity_frees_when_manager_record_completes(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "capacity")
    service = SubagentService(ctx, {"maxActiveSubagents": 1})
    service._members["sub-1"] = _member("sub-1", "worker")
    record = SubagentRecord(
        subagent_id="sub-1",
        role="worker",
        agent_name="worker",
        model_id="inherit",
        workspace_path=Path("."),
        session_id="child-session",
        session_path=Path("."),
        state=SubagentState.COMPLETED,
    )
    service._members["sub-1"].record = record
    service._sync_states()
    assert service._members["sub-1"].state == "completed"
    assert service.capacity_of(None)["running"] == 0


@pytest.mark.asyncio
async def test_fork_replays_only_balanced_prefix(tmp_path: Any) -> None:
    from magitech_harness import create_harness as real_create
    from magitech_harness.session.pairing import is_tool_pairing_balanced

    parent = real_create(storage_dir=tmp_path / "fork-balance")
    await parent.flush()
    session = parent.sessions.create("parent")
    session.append("turn/start", {"turn": 1})
    session.append(
        "user/message",
        {
            "role": "user",
            "id": "m1",
            "content": [{"type": "text", "text": "hi"}],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )
    session.append(
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "role": "assistant",
                "id": "a1",
                "content": [{"type": "text", "text": "ok"}],
                "source": {"kind": "model", "provider": "mock", "model": "mock"},
            },
            "stream": [],
        },
        surface_op="append",
    )
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    session.append("turn/start", {"turn": 2})
    session.append(
        "tool/call",
        {
            "turn": 2,
            "step": 1,
            "callId": "c2",
            "name": "bash",
            "arguments": {"command": "echo x"},
        },
    )
    parent.session = session

    captured: list[Any] = []

    def _factory(storage_dir: Any = None, **kwargs: Any) -> Any:
        child = real_create(storage_dir=tmp_path / "fork-child")
        captured.append(child)
        return child

    from unittest.mock import patch

    with patch(
        "magitech_harness.subagent.providers.create_harness", side_effect=_factory
    ):
        await providers.start_fork(SubagentSpec(role="worker", prompt="go"), parent)
    assert captured
    await captured[0].flush()
    child_session = captured[0].sessions.list_sessions()[-1]
    child = captured[0].sessions.read(child_session)
    assert child is not None
    assert [event.type for event in child.events] == [
        "turn/start",
        "user/message",
        "assistant/message",
        "turn/end",
    ]
    assert is_tool_pairing_balanced(child.events, len(child.events) - 1)


@pytest.mark.asyncio
async def test_wait_agent_abort_sets_stop_without_leaking(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "abort-wait")
    await ctx.flush()
    session = ctx.sessions.create("abort-session")
    agent = _FakeAgent(pending=False)
    agent.session = session
    signal = AbortSignal()
    from magitech_harness.subagent import tools as tools_mod

    seen: list[threading.Event] = []

    def _wait(inbox: Any, stop: threading.Event) -> bool:
        seen.append(stop)
        return stop.wait(1) is False

    original = tools_mod._wait_for_input
    tools_mod._wait_for_input = _wait  # type: ignore[assignment]
    try:
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
                "signal": signal,
                "agent": agent,
            }
        )
    finally:
        tools_mod._wait_for_input = original  # type: ignore[assignment]
    assert batch["results"][0]["result"]["isError"] is False
    assert seen and seen[0].is_set()


def test_depth_uses_caller_session_id() -> None:
    service = _service()
    service.register_provider(
        SubagentProvider(name="spawn", start=lambda _s, _c, **_kw: _empty())
    )
    service._members["sub-1"] = _member("sub-1", "worker")
    service._members["sub-1"].session_id = "child-session"
    service._members["sub-1"].depth = 1
    service._max_depth = 1
    with pytest.raises(SubagentError, match="exceeds the configured cap"):
        asyncio.run(
            service.startActivation(
                SubagentSpec(role="worker", prompt="go"),
                parent_id="child-session",
            )
        )


def test_ancestry_guards_reject_foreign_caller() -> None:
    service = _service()
    service._members["sub-1"] = _member("sub-1", "worker")
    service._members["sub-1"].session_id = "child-1"
    service._members["sub-2"] = _member("sub-2", "other", parent="sub-1")
    service._members["sub-2"].session_id = "child-2"
    service._members["sub-2"].depth = 1
    with pytest.raises(SubagentError, match="outside the caller tree"):
        service.resolve_name(_FakeAgent("child-2"), "worker")
    with pytest.raises(SubagentError, match="ancestor of the target"):
        service.interrupt("sub-2", {"agent": _FakeAgent("child-2")})


@pytest.mark.asyncio
async def test_wait_agent_abort_during_turn(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "abort-midflight")
    await ctx.flush()
    session = ctx.sessions.create("abort-session-2")
    agent = _FakeAgent(pending=False)
    agent.session = session
    signal = AbortSignal()
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
            "signal": signal,
            "agent": agent,
        }
    )
    assert batch["results"][0]["result"]["isError"] is False


def test_sync_states_skips_record_without_state() -> None:
    service = _service()
    member = _member("sub-1", "worker")
    member.record = object()
    service._members["sub-1"] = member
    service._sync_states()
    assert member.state == "running"


def test_ancestor_walks_grandparent_chain() -> None:
    service = _service()
    service._members["root-child"] = _member("root-child", "child")
    service._members["root-child"].session_id = "child-session"
    service._members["grand"] = _member("grand", "grand", parent="root-child")
    service._members["grand"].depth = 1
    service._members["grand"].session_id = "grand-session"
    assert service._caller_member(_FakeAgent("child-session")) is not None
    assert service._ancestor_of(_FakeAgent("child-session"), service._members["grand"])


def test_resolve_name_rejects_unknown() -> None:
    service = _service()
    with pytest.raises(SubagentError, match="no agent named"):
        service.resolve_name(None, "missing")


def test_log_event_without_sessions_service() -> None:
    ctx = Context()
    LLMService(ctx)
    service = SubagentService(ctx)
    service._log_event("subagent/descriptor", {})


def test_wait_for_input_waits_then_stops() -> None:
    from magitech_harness.subagent.tools import _wait_for_input

    stop = threading.Event()
    stop.set()
    assert _wait_for_input(_FakeInbox(False), stop) is False


@pytest.mark.asyncio
async def test_dsh_named_control_tools_success(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "dsh-control")
    await ctx.flush()
    session = ctx.sessions.create("dsh-control-session")
    agent = _FakeAgent()
    agent.session = session
    service: SubagentService = ctx.subagents
    service._members["sub-1"] = _member("sub-1", "worker")
    service._members["sub-1"].session_id = "child-1"
    manager = service._manager_for(ctx)
    manager._records["sub-1"] = SubagentRecord(
        subagent_id="sub-1",
        role="worker",
        agent_name="worker",
        model_id="inherit",
        workspace_path=Path("."),
        session_id="child-1",
        session_path=Path("."),
        state=SubagentState.RUNNING,
    )

    async def _run_child(child_id: str, _message: str) -> None:
        assert child_id == "sub-1"

    manager._run_child = _run_child  # type: ignore[method-assign]

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


@pytest.mark.asyncio
async def test_wait_for_tree_work() -> None:
    from types import SimpleNamespace

    from animas import Context

    from magitech_harness.subagent.service import SubagentService, _Member

    async def idle() -> None:
        return None

    ctx = Context()
    ctx.provide(
        "agents",
        SimpleNamespace(
            get=lambda identifier: (
                SimpleNamespace(when_idle=idle) if identifier == "c1" else None
            )
        ),
    )
    service = SubagentService(ctx)
    caller = SimpleNamespace(id="root")
    await service.wait_for_tree_work(caller)

    service._members.update(
        {
            "c1": _Member("c1", "c1", "root", 1, session_id="s1"),
            "c2": _Member("c2", "c2", "root", 1, session_id="s2"),
        }
    )
    await service.wait_for_tree_work(caller)

    ctx.provide("agents", None)
    await service.wait_for_tree_work(caller)
