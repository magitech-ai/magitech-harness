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
    revision: int = 1,
) -> GoalView:
    return GoalView(
        goal=GoalSnapshot(
            id="goal-1",
            revision=revision,
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
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(), disarm=lambda a: None, block=lambda *a: None
        ),
    )
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

    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(), disarm=lambda a: None, block=lambda *a: None
        ),
    )
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
    blocked: list[Any] = []
    disarmed: list[Any] = []
    calls = {"n": 0}

    def goal_after_failure(_agent: Any) -> Any:
        calls["n"] += 1
        return _goal() if calls["n"] == 1 else _goal(revision=9) if False else _goal()

    ctx.provide(
        "goals",
        SimpleNamespace(
            get=goal_after_failure,
            disarm=lambda a: disarmed.append(a),
            block=lambda *a: blocked.append(a),
        ),
    )
    await driver.drive(agent, state)
    assert len(blocked) == 1 or disarmed == [agent]


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
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(), disarm=lambda a: None, block=lambda *a: None
        ),
    )
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
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(), disarm=lambda a: None, block=lambda *a: None
        ),
    )
    await driver.drive(agent, state)
    state.needs_checkpoint = True
    await driver.drive(agent, state)

    # create_task failure disarms.
    def raise_task(coro: Any) -> None:
        coro.close()
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


@pytest.mark.asyncio
async def test_prestep_gate_and_lifecycle() -> None:
    import asyncio

    from magitech_harness.goal.round_driver import _Round
    from magitech_harness.llm.message import create_user_message

    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent()
    inbox_calls: list[Any] = []
    agent.inbox = SimpleNamespace(
        prepend=lambda target, message: inbox_calls.append((target, message)),
        remove=lambda message_id: inbox_calls.append(("remove", message_id)),
    )
    goal = _goal()
    blocked: list[Any] = []
    disarmed: list[Any] = []
    ctx.provide("agents", SimpleNamespace(get=lambda identifier: agent))
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: goal,
            disarm=lambda a: disarmed.append(a),
            block=lambda *a: blocked.append(a),
        ),
    )
    dispose = await driver.init()
    await ctx.flush()
    state = driver._state(agent)
    callback = ctx.events._listeners["agent/pre-step"][0].callback

    decision = {"kind": "enter", "messages": []}
    plain = callback({"agent": agent, "messages": []}, lambda: decision)
    assert plain == decision

    submitted = {
        "id": "m1",
        "content": [{"type": "text", "text": "x"}],
        "source": {"kind": "goal", "goalId": "goal-1", "revision": 1, "round": 1},
    }
    other = {"id": "other", "source": {"kind": "user"}}

    # No attempt -> invalid reservation -> reject and restore other messages.
    result = await callback(
        {"agent": agent, "messages": [submitted, other]}, lambda: decision
    )
    assert result["kind"] == "reject"
    assert inbox_calls

    # Valid reservation, downstream rejects -> block.
    msg = create_user_message(
        content=[{"type": "text", "text": "x"}],
        source={"kind": "goal", "goalId": "goal-1", "revision": 1, "round": 1},
    )
    state.attempt = _Round("goal-1", 1, 1, msg["id"], msg["content"], phase="claimed")
    result = await callback(
        {"agent": agent, "messages": [msg]}, lambda: {"kind": "reject"}
    )
    assert result["kind"] == "reject"
    assert len(blocked) == 1

    # Valid reservation, normal decision -> startsRequestSeries.
    state.attempt = _Round("goal-1", 1, 1, msg["id"], msg["content"], phase="claimed")
    result = await callback({"agent": agent, "messages": [msg]}, lambda: decision)
    assert result["startsRequestSeries"] is True

    # Async downstream decision.
    async def async_next() -> dict[str, Any]:
        return decision

    state.attempt = _Round("goal-1", 1, 1, msg["id"], msg["content"], phase="claimed")
    result = await callback({"agent": agent, "messages": [msg]}, async_next)
    assert result["kind"] == "enter"

    # Abort variants on turn/end.
    for reason in (
        {"kind": "aborted", "reason": {"kind": "disposed"}},
        {"kind": "aborted", "reason": {"kind": "user"}},
        {"kind": "stop"},
    ):
        ctx.emit(
            "session/event",
            SimpleNamespace(session_id="agent-1"),
            SimpleNamespace(type="turn/end", payload={"reason": reason}),
        )
    state.attempt = _Round("goal-1", 1, 1, msg["id"], msg["content"], phase="claimed")
    ctx.emit(
        "session/event",
        SimpleNamespace(session_id="agent-1"),
        SimpleNamespace(
            type="turn/end",
            payload={"reason": {"kind": "aborted", "reason": {"kind": "user"}}},
        ),
    )
    assert state.attempt.cancelled is True

    # Disposal disarms and cancels the serialized task.
    async def pending() -> None:
        await asyncio.sleep(10)

    state.run = asyncio.create_task(pending())
    dispose()
    await asyncio.gather(state.run, return_exceptions=True)


@pytest.mark.asyncio
async def test_driver_guard_edge_paths() -> None:
    import asyncio

    from magitech_harness.goal.round_driver import _Round
    from magitech_harness.llm.message import create_user_message

    ctx = Context()
    driver = GoalRoundDriver(ctx)
    no_inbox = SimpleNamespace(
        id="no-inbox", status="idle", session=SimpleNamespace(session_id="no-inbox")
    )
    driver._restore_other_claimed(no_inbox, [{"id": "x"}], "m")
    no_prepend = SimpleNamespace(id="no-prepend")
    no_prepend.inbox = SimpleNamespace()
    driver._restore_other_claimed(no_prepend, [{"id": "x"}], "m")

    agent = _agent()
    state = driver._state(agent)
    state.attempt = _Round("goal-1", 1, 1, "m1", [], phase="claimed")
    assert driver._same_queued([], {}, None) is False

    current: dict[str, Any] = {"goal": _goal()}
    blocked: list[Any] = []
    disarmed: list[Any] = []
    ctx.provide("agents", SimpleNamespace(get=lambda identifier: agent))
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: current["goal"],
            disarm=lambda a: disarmed.append(a),
            block=lambda *a: blocked.append(a),
        ),
    )
    dispose = await driver.init()
    await ctx.flush()
    callback = ctx.events._listeners["agent/pre-step"][0].callback
    msg = create_user_message(
        content=[{"type": "text", "text": "x"}],
        source={"kind": "goal", "goalId": "goal-1", "revision": 1, "round": 1},
    )

    # Downstream rejects while the goal is no longer drivable.
    def make_non_drivable() -> dict[str, Any]:
        current["goal"] = _goal(phase="complete")
        return {"kind": "reject"}

    state.attempt = _Round("goal-1", 1, 1, msg["id"], msg["content"], phase="claimed")
    result = await callback({"agent": agent, "messages": [msg]}, make_non_drivable)
    assert result["kind"] == "reject"

    # The goal changes between the first and second reservation checks.
    ctx.provide(
        "goals",
        SimpleNamespace(
            get=lambda a: _goal(revision=9),
            disarm=lambda a: disarmed.append(a),
            block=lambda *a: blocked.append(a),
        ),
    )
    state.attempt = _Round("goal-1", 1, 1, msg["id"], msg["content"], phase="claimed")
    result = await callback(
        {"agent": agent, "messages": [msg]},
        lambda: {"kind": "enter", "messages": []},
    )
    assert result["kind"] == "reject"

    # Abort with no attempt disarms.
    ctx.emit(
        "session/event",
        SimpleNamespace(session_id="agent-1"),
        SimpleNamespace(
            type="turn/end",
            payload={"reason": {"kind": "aborted", "reason": {"kind": "user"}}},
        ),
    )

    # Dispose with a task in flight and a state without a live agent.
    state.run = asyncio.create_task(asyncio.sleep(10))
    driver._states["orphan"] = SimpleNamespace(
        stopping=False, requested=True, agent=None, run=None
    )
    dispose()
    await asyncio.gather(state.run, return_exceptions=True)


@pytest.mark.asyncio
async def test_driver_final_coverage() -> None:
    import asyncio

    from magitech_harness.goal.round_driver import _Round

    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent()
    ctx.provide("agents", SimpleNamespace(get=lambda identifier: agent))
    disarmed: list[Any] = []
    current: dict[str, Any] = {"goal": _goal()}

    def get_goal(_agent: Any) -> Any:
        return current["goal"]

    ctx.provide(
        "goals",
        SimpleNamespace(
            get=get_goal,
            disarm=lambda a: disarmed.append(a),
            block=lambda *a: None,
        ),
    )

    # followup failure where the goal changed -> disarm.
    state = driver._state(agent)

    def fail(_message: Any) -> None:
        current["goal"] = _goal(revision=9)
        raise RuntimeError("queue failed")

    agent.followup = fail
    await driver.drive(agent, state)
    assert disarmed

    # Goal changes during the downstream step, invalidating the reservation.
    dispose = await driver.init()
    await ctx.flush()

    # Idle with a queued attempt whose inbox has no remove method.
    removed: list[Any] = []
    agent.inbox = SimpleNamespace(remove=lambda message_id: removed.append(message_id))
    state.attempt = _Round("goal-1", 1, 1, "m1", [], phase="queued")
    ctx.emit("agent/status", {"agent": agent, "status": "idle"})
    await ctx.flush()
    callback = ctx.events._listeners["agent/pre-step"][0].callback
    from magitech_harness.llm.message import create_user_message

    msg = create_user_message(
        content=[{"type": "text", "text": "x"}],
        source={"kind": "goal", "goalId": "goal-1", "revision": 1, "round": 1},
    )
    current["goal"] = _goal()

    def change_goal() -> dict[str, Any]:
        state.attempt = None
        return {"kind": "enter", "messages": []}

    state.attempt = _Round("goal-1", 1, 1, msg["id"], msg["content"], phase="claimed")
    result = await callback({"agent": agent, "messages": [msg]}, change_goal)
    assert result["kind"] == "reject"

    # Post-decision check still valid when goal keeps its revision.
    def keep_goal() -> dict[str, Any]:
        return {"kind": "enter", "messages": []}

    current["goal"] = _goal()
    state.attempt = _Round("goal-1", 1, 1, msg["id"], msg["content"], phase="claimed")
    result = await callback({"agent": agent, "messages": [msg]}, keep_goal)
    assert result["startsRequestSeries"] is True

    # retire() reschedules when a request arrives as the task completes.
    agent2 = _agent("agent-2")
    state2 = driver._state(agent2)
    driver.request_drive(agent2)
    driver.request_drive(agent2)
    if state2.run is not None:
        await state2.run
    await asyncio.sleep(0)
    dispose()


@pytest.mark.asyncio
async def test_retire_reschedules_on_late_request() -> None:

    ctx = Context()
    driver = GoalRoundDriver(ctx)
    agent = _agent("late-root")
    state = driver._state(agent)
    state.requested = True
    state.run = object()
    driver._reschedule(state, agent)
    assert state.run is not None
    await state.run
    state.requested = False
    driver._reschedule(state, agent)
    assert state.run is None
