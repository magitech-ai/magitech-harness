"""Tests for the same-session goal-round driver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness.goal.round_driver import (
    ROUND_LIMIT_CODE,
    GoalRoundDriver,
    apply,
    render_goal_round_prompt,
)
from magitech_harness.goal.types import GoalSnapshot, GoalView


def _goal(
    rounds: int = 0,
    maximum: int = 3,
    phase: str = "active",
    activation: str = "armed",
) -> GoalView:
    return GoalView(
        goal=GoalSnapshot(
            id="goal-1",
            revision=1,
            objective="ship it",  # type: ignore[arg-type]
            phase=phase,  # type: ignore[arg-type]
            max_goal_rounds=maximum,
        ),
        rounds_started=rounds,
        created_at=0,
        updated_at=0,
        activation=activation,  # type: ignore[arg-type]
    )


def _agent(status: str = "idle") -> Any:
    followups: list[Any] = []
    return SimpleNamespace(
        id="agent-1",
        status=status,
        session=SimpleNamespace(session_id="agent-1"),
        followup=followups.append,
        _followups=followups,
    )


def test_render_prompt() -> None:
    blocks = render_goal_round_prompt(_goal(), 2)
    text = blocks[0]["text"]
    assert text.startswith("<goal_round>")
    assert 'Objective: "ship it"' in text
    assert "Round: 2/3" in text
    assert text.endswith("</goal_round>")


def test_state_and_readiness() -> None:
    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent()
    state = driver._state(agent)
    assert driver._state(agent) is state
    assert driver._current_goal(agent) is None
    assert driver._drivable(None) is False
    assert driver._drivable(_goal()) is True
    assert driver._drivable(_goal(phase="complete")) is False
    assert driver._drivable(_goal(activation="disarmed")) is False
    assert driver._ready(agent, state) is True
    state.stopping = True
    assert driver._ready(agent, state) is False
    state.stopping = False
    state.competing_queued = True
    assert driver._ready(agent, state) is False

    ctx.provide("agents", SimpleNamespace(get=lambda identifier: None))
    assert driver._ready(agent, state) is False
    assert driver._current_goal(agent) is None
    ctx.provide("agents", SimpleNamespace(get=lambda identifier: agent))
    ctx.provide("goals", SimpleNamespace(get=lambda a: _goal()))
    assert driver._current_goal(agent) is not None

    bad = SimpleNamespace(id="a", status="running", session=agent.session)
    assert driver._ready(bad, driver._state(bad)) is False


@pytest.mark.asyncio
async def test_drive_paths() -> None:
    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent()
    state = driver._state(agent)
    ctx.provide("agents", SimpleNamespace(get=lambda identifier: agent))

    # Not ready.
    agent.status = "running"
    await driver.drive(agent, state)
    assert not agent._followups
    agent.status = "idle"

    # Checkpoint path with no sessions service.
    state.needs_checkpoint = True
    await driver.drive(agent, state)
    assert state.needs_checkpoint is False

    # Checkpoint path with a sessions service whose flush fails recheck.
    flushed: list[Any] = []
    ctx.provide("sessions", SimpleNamespace(flush=lambda s: flushed.append(s)))
    state.needs_checkpoint = True
    await driver.drive(agent, state)
    assert len(flushed) == 1

    # An outstanding attempt schedules the next round.
    from magitech_harness.goal.round_driver import _Round

    state.attempt = _Round("goal-1", 1, 1, "m1")
    await driver.drive(agent, state)
    assert state.attempt is None
    assert state.needs_checkpoint is True

    # No drivable goal.
    state.needs_checkpoint = False
    await driver.drive(agent, state)
    assert not agent._followups

    ctx.provide("goals", SimpleNamespace(get=lambda a: _goal(), disarm=lambda a: None))
    # Subagent wait path.
    ctx.provide("subagents", SimpleNamespace(wait_for_tree_work=lambda a: None))
    await driver.drive(agent, state)
    assert len(agent._followups) == 1
    source = agent._followups[0]["source"]
    assert source["kind"] == "goal" and source["round"] == 1

    # Round limit.
    ctx.provide(
        "goals", SimpleNamespace(get=lambda a: _goal(rounds=3), block=lambda *a: None)
    )
    await driver.drive(agent, state)
    assert len(agent._followups) == 1

    # Follow-up failure disarms.
    def fail(_message: Any) -> None:
        raise RuntimeError("queue failed")

    agent.followup = fail
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(),
            disarm=lambda a: disarmed.append(a),
        ),
    )
    disarmed: list[Any] = []
    await driver.drive(agent, state)
    assert disarmed == [agent]


@pytest.mark.asyncio
async def test_request_drive_and_disarm() -> None:
    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent()
    ctx.provide("agents", SimpleNamespace(get=lambda identifier: agent))
    state = driver._state(agent)
    state.stopping = True
    driver.request_drive(agent)
    assert state.run is None
    state.stopping = False
    ctx.provide("goals", SimpleNamespace(get=lambda a: _goal(), disarm=lambda a: None))
    driver.request_drive(agent)
    task = state.run
    assert task is not None
    await task
    driver.request_drive(agent)
    assert state.run is not None
    await state.run

    # No goals service means disarm is a no-op.
    driver._disarm(agent)
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(),
            disarm=lambda a: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
    )
    driver._disarm(agent)


@pytest.mark.asyncio
async def test_listeners_and_dispose() -> None:
    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent()
    ctx.provide(
        "agents",
        SimpleNamespace(get=lambda identifier: agent, currentInitiator=lambda: None),
    )
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(),
            disarm=lambda a: None,
            block=lambda *a: None,
        ),
    )
    dispose = await driver.init()
    await ctx.flush()

    ctx.emit("agent/status", {"agent": agent, "status": "idle"})
    ctx.emit("agent/error", {"agent": agent})
    ctx.emit("agent/disposed", {"agent": agent})
    ctx.emit("agent/created", {"agent": agent})
    ctx.emit("agent/status", {"agent": None, "status": "idle"})
    ctx.emit("agent/error", {"agent": None})
    ctx.emit("agent/disposed", {"agent": None})
    ctx.emit("agent/created", {"agent": None})
    state = driver._state(agent)

    from magitech_harness.goal.round_driver import _Round

    state.attempt = _Round("goal-1", 1, 1, "m1")
    message = {
        "id": "m1",
        "source": {"kind": "goal", "goalId": "goal-1", "revision": 1, "round": 1},
    }
    ctx.emit("agent/inbox/inserted", {"agent": agent, "message": message})
    ctx.emit("agent/inbox/claimed", {"agent": agent, "message": message})
    ctx.emit("agent/inbox/discarded", {"agent": agent, "message": message})
    ctx.emit("agent/inbox/inserted", {"agent": agent, "message": {"id": "other"}})
    ctx.emit("agent/inbox/inserted", {"agent": None, "message": message})
    ctx.emit("agent/inbox/inserted", {"agent": agent, "message": None})

    session = SimpleNamespace(session_id="agent-1")
    ctx.emit(
        "session/event",
        session,
        SimpleNamespace(type="goal/change", payload={"operation": "create"}),
    )
    ctx.emit(
        "session/event",
        session,
        SimpleNamespace(type="user/message", payload={"id": "m1"}),
    )
    ctx.emit(
        "session/event",
        session,
        SimpleNamespace(type="turn/end", payload={"reason": {"kind": "max-tokens"}}),
    )
    ctx.emit(
        "session/event",
        session,
        SimpleNamespace(type="turn/end", payload={"reason": {"kind": "aborted"}}),
    )
    ctx.emit(
        "session/event",
        session,
        SimpleNamespace(type="turn/end", payload={"reason": {"kind": "stop"}}),
    )
    ctx.emit(
        "session/event",
        SimpleNamespace(session_id="unknown"),
        SimpleNamespace(type="goal/change", payload={}),
    )
    dispose()
    assert state.stopping is True


def test_standalone_apply() -> None:
    ctx = Context()
    assert isinstance(apply(ctx), GoalRoundDriver)
    assert ROUND_LIMIT_CODE == "round-limit"


@pytest.mark.asyncio
async def test_driver_remaining_branches() -> None:
    from unittest.mock import patch

    from magitech_harness.goal.round_driver import _Round

    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent()
    ctx.provide("agents", SimpleNamespace(get=lambda identifier: agent))
    state = driver._state(agent)

    # _disarm with no goals service, then with a raising disarm.
    driver._disarm(agent)
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(),
            disarm=lambda a: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
    )
    driver._disarm(agent)

    # Round-limit path with no goals service.
    ctx.provide(
        "goals", SimpleNamespace(get=lambda a: _goal(rounds=3), block=lambda *a: None)
    )
    await driver.drive(agent, state)

    # Awaiting subagent tree work; then a recheck that fails.
    async def wait(_agent: Any) -> None:
        return None

    ctx.provide("subagents", SimpleNamespace(wait_for_tree_work=wait))
    ctx.provide("goals", SimpleNamespace(get=lambda a: _goal()))
    await driver.drive(agent, state)
    state.needs_checkpoint = True
    await driver.drive(agent, state)

    # create_task failure disarms.
    def raise_task(_coro: Any) -> None:
        raise RuntimeError("no loop")

    with patch("asyncio.create_task", side_effect=raise_task):
        driver.request_drive(agent)

    # Listeners: idle clears a queued reservation; competing input clears one.
    dispose = await driver.init()
    await ctx.flush()
    state.attempt = _Round("goal-1", 1, 1, "m1", phase="queued")
    ctx.emit("agent/status", {"agent": agent, "status": "idle"})
    await ctx.flush()
    state.attempt = _Round("goal-1", 1, 2, "m2", phase="queued")
    ctx.emit(
        "agent/inbox/inserted",
        {"agent": agent, "message": {"id": "other", "source": {}}},
    )
    dispose()


@pytest.mark.asyncio
async def test_wait_recheck_and_drive_failure() -> None:

    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent()
    ctx.provide("agents", SimpleNamespace(get=lambda identifier: agent))
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(), disarm=lambda a: None, block=lambda *a: None
        ),
    )
    state = driver._state(agent)

    async def wait_and_stop(_agent: Any) -> None:
        agent.status = "running"

    ctx.provide("subagents", SimpleNamespace(wait_for_tree_work=wait_and_stop))
    await driver.drive(agent, state)
    assert state.attempt is None

    agent.status = "idle"
    original = driver.drive

    async def failing(a: Any, s: Any) -> None:
        raise RuntimeError("drive failed")

    driver.drive = failing  # type: ignore[method-assign]
    state.run = None
    state.requested = False
    driver.request_drive(agent)
    if state.run is not None:
        await state.run
    driver.drive = original  # type: ignore[method-assign]
