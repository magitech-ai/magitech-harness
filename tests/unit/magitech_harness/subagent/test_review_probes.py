"""Reviewer-driven probes for the relocated subagent surface."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortController
from magitech_harness.subagent.service import (
    SubagentError,
    SubagentProvider,
    SubagentService,
)
from magitech_harness.subagent.types import (
    SubagentRecord,
    SubagentSpec,
    SubagentState,
)


class _Agent:
    def __init__(self, agent_id: str = "root", session: Any = None) -> None:
        self.id = agent_id
        self.session = session
        self.inbox = type("Inbox", (), {"has_pending": True})()


async def _empty() -> dict[str, Any]:
    return {}


def _record(
    child_id: str, state: SubagentState = SubagentState.COMPLETED
) -> SubagentRecord:
    return SubagentRecord(
        subagent_id=child_id,
        role="worker",
        agent_name="worker",
        model_id="inherit",
        workspace_path=Path("."),
        session_id="child-session",
        session_path=Path("."),
        state=state,
    )


@pytest.mark.asyncio
async def test_lineage_skipped_without_session(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "no-session")
    await ctx.flush()
    service: SubagentService = ctx.subagents
    service.register_provider(
        SubagentProvider(name="spawn", start=lambda _s, _c, **_kw: _empty())
    )
    manager = service._manager_for(ctx)

    async def _invoke(_specs: Any, **_kwargs: Any) -> list[SubagentRecord]:
        return [_record("sub-nolineage")]

    manager.invoke = _invoke  # type: ignore[method-assign]
    result = await service.startActivation(SubagentSpec(role="worker", prompt="go"))
    assert result["id"] == "sub-nolineage"


@pytest.mark.asyncio
async def test_send_message_outside_tree_refused() -> None:
    ctx = create_harness(storage_dir="/tmp/opencode/sub-outside")
    await ctx.flush()
    service: SubagentService = ctx.subagents
    service._members["sub-1"] = type(
        "M",
        (),
        {
            "child_id": "sub-1",
            "name": "worker",
            "parent_id": None,
            "depth": 0,
            "state": "running",
            "label": "worker",
            "session_id": "child-1",
            "record": _record("sub-1", SubagentState.RUNNING),
        },
    )()
    outsider = type(
        "M",
        (),
        {
            "child_id": "sub-outsider",
            "name": "outsider",
            "parent_id": None,
            "depth": 0,
            "state": "running",
            "label": "outsider",
            "session_id": "other-child",
            "record": None,
        },
    )()
    service._members["sub-outsider"] = outsider
    with pytest.raises(SubagentError, match="messaging requires an ancestor"):
        await service.send_message(
            _Agent("other-child"), "sub-1", [{"type": "text", "text": "hi"}]
        )


@pytest.mark.asyncio
async def test_log_event_without_active_session(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "inactive")
    service = SubagentService(ctx)
    service._log_event("subagent/descriptor", {})


@pytest.mark.asyncio
async def test_wait_agent_abort_registers_and_disposes(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "abort-register")
    await ctx.flush()
    agent = _Agent(session=ctx.sessions.create("abort-session"))
    controller = AbortController()
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
            "signal": controller.signal,
            "agent": agent,
        }
    )
    assert batch["results"][0]["result"]["isError"] is False


def test_wait_for_input_returns_false_when_stopped() -> None:
    from magitech_harness.subagent.tools import _wait_for_input

    stop = threading.Event()
    stop.set()
    assert _wait_for_input(_Agent(session=None), stop) is False


@pytest.mark.asyncio
async def test_background_completion_frees_capacity_without_manual_sync(
    tmp_path: Any,
) -> None:
    ctx = create_harness(storage_dir=tmp_path / "background-capacity")
    await ctx.flush()
    service: SubagentService = ctx.subagents
    service._max_active = 1
    service.register_provider(
        SubagentProvider(name="spawn", start=lambda _s, _c, **_kw: _empty())
    )
    manager = service._manager_for(ctx)
    record = _record("sub-bg", SubagentState.PENDING)

    async def _invoke(_specs: Any, **_kwargs: Any) -> list[SubagentRecord]:
        # A background child returns before its turn completes.
        return [record]

    manager.invoke = _invoke  # type: ignore[method-assign]
    first = await service.startActivation(
        SubagentSpec(role="worker", prompt="go", agent_name="w1")
    )
    assert first["id"] == "sub-bg"
    assert service.capacity_of(None)["running"] == 1

    # The manager's record settles later; the service must project that state
    # on the next read rather than requiring a manual _sync_states() call.
    record.state = SubagentState.COMPLETED
    assert service.capacity_of(None)["running"] == 0

    async def _invoke_two(_specs: Any, **_kwargs: Any) -> list[SubagentRecord]:
        return [_record("sub-bg-2", SubagentState.COMPLETED)]

    manager.invoke = _invoke_two  # type: ignore[method-assign]
    second = await service.startActivation(
        SubagentSpec(role="worker", prompt="go", agent_name="w2")
    )
    assert second["id"] == "sub-bg-2"
