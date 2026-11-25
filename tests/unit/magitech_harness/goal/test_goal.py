"""Tests for the goal domain, its projection, and the three goal tools.

Covers the DSH ``@deepseek-ai/dsh-goal`` / ``@deepseek-ai/dsh-tool-goal``
contract: create/edit/pause/resume/complete/block/clear with compare-and-set
revisions, the admitted-round counter, durable replay validation, and the
authority rules for the model-facing tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.goal.service import GoalService
from magitech_harness.goal.types import (
    GoalBlockReason,
    GoalError,
    GoalFoldState,
    GoalRef,
    GoalSnapshot,
    GoalView,
    apply_goal_change,
    apply_goal_event,
    change_ref,
    decode_goal_change,
    fold_goal,
)
from magitech_harness.llm.error import HarnessError
from magitech_harness.session.events import SessionEvent, ToolCallData


async def _boot(tmp_path: Path) -> Any:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    return ctx


async def _live_agent(ctx: Any) -> Any:
    handle = await ctx.agents.create({})
    return handle["agent"]


def _change(
    operation: str,
    *,
    goal: dict[str, Any] | None = None,
    rounds: int = 0,
    created: int = 1000,
    updated: int = 1000,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": "goal/change",
        "version": 1,
        "operation": operation,
        "roundsStarted": rounds,
        "createdAt": created,
        "updatedAt": updated,
    }
    if goal is not None:
        value["goal"] = goal
    return value


def _snapshot(
    *,
    goal_id: str = "g1",
    revision: int = 1,
    phase: str = "active",
    objective: str = "ship it",
    max_rounds: int = 256,
    blocked: dict[str, str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": goal_id,
        "revision": revision,
        "objective": objective,
        "phase": phase,
        "maxGoalRounds": max_rounds,
    }
    if blocked is not None:
        value["blockedReason"] = blocked
    return value


def _event(event_type: str, payload: dict[str, Any], seq: int = 0) -> SessionEvent:
    return SessionEvent(
        id=f"s:{seq}",
        session_id="s",
        type=event_type,
        timestamp=0.0,
        payload=payload,
        seq=seq,
    )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def test_decode_returns_none_for_other_kinds() -> None:
    assert decode_goal_change({"kind": "other"}) is None
    assert decode_goal_change("not a record") is None


def test_decode_rejects_unknown_version() -> None:
    with pytest.raises(GoalError, match="unsupported goal change version"):
        decode_goal_change(
            {"kind": "goal/change", "version": 99, "operation": "create"}
        )


def test_decode_rejects_extra_fields() -> None:
    with pytest.raises(GoalError, match="must have exactly"):
        decode_goal_change({**_change("create", goal=_snapshot()), "extra": 1})


def test_decode_rejects_invalid_operation() -> None:
    with pytest.raises(GoalError, match="operation is invalid"):
        decode_goal_change(
            {
                "kind": "goal/change",
                "version": 1,
                "operation": "nope",
                "goal": _snapshot(),
                "roundsStarted": 0,
                "createdAt": 1,
                "updatedAt": 1,
            }
        )


def test_decode_rejects_updated_before_created() -> None:
    with pytest.raises(GoalError, match="cannot precede createdAt"):
        decode_goal_change(_change("create", goal=_snapshot(), created=10, updated=5))


def test_decode_clear_tombstone() -> None:
    change = decode_goal_change(
        {
            "kind": "goal/change",
            "version": 1,
            "operation": "clear",
            "cleared": {"id": "g1", "revision": 2},
            "clearedAt": 2000,
        }
    )
    assert change is not None
    assert change_ref(change) == GoalRef(id="g1", revision=2)


def test_decode_clear_rejects_extra_fields() -> None:
    with pytest.raises(GoalError, match="goal clear change must have exactly"):
        decode_goal_change(
            {
                "kind": "goal/change",
                "version": 1,
                "operation": "clear",
                "cleared": {"id": "g1", "revision": 2},
                "clearedAt": 2000,
                "goal": _snapshot(),
            }
        )


def test_decode_snapshot_field_validation() -> None:
    cases = [
        ("id", {"id": ""}, "goal.id must be a non-empty string"),
        ("objective", {"objective": " x "}, "must be non-empty and normalized"),
        ("phase", {"phase": "bogus"}, "goal.phase is invalid"),
        ("revision", {"revision": 0}, "goal.revision must be a positive"),
    ]
    for _label, override, message in cases:
        goal = {**_snapshot(), **override}
        with pytest.raises(GoalError, match=message):
            decode_goal_change(_change("create", goal=goal))


def test_decode_blocked_requires_reason() -> None:
    goal = {
        **_snapshot(phase="blocked"),
        "blockedReason": {"code": "x", "message": "m"},
    }
    change = decode_goal_change(_change("block", goal=goal))
    assert change is not None
    assert change.goal.blocked_reason == GoalBlockReason(code="x", message="m")


def test_decode_blocked_reason_shape() -> None:
    goal = {
        **_snapshot(phase="blocked"),
        "blockedReason": {"code": "x", "message": "m", "other": 1},
    }
    with pytest.raises(GoalError, match="exactly code and message"):
        decode_goal_change(_change("block", goal=goal))


def test_decode_blocked_reason_code_must_be_kebab() -> None:
    goal = {
        **_snapshot(phase="blocked"),
        "blockedReason": {"code": "NotKebab", "message": "m"},
    }
    with pytest.raises(GoalError, match="lower-kebab-case"):
        decode_goal_change(_change("block", goal=goal))


def test_decode_blocked_reason_message_normalized() -> None:
    goal = {
        **_snapshot(phase="blocked"),
        "blockedReason": {"code": "x", "message": " m "},
    }
    with pytest.raises(GoalError, match="non-empty and normalized"):
        decode_goal_change(_change("block", goal=goal))


def test_decode_snapshot_unexpected_phase_fields() -> None:
    goal = {**_snapshot(), "blockedReason": {"code": "x", "message": "m"}}
    with pytest.raises(GoalError, match="must have exactly"):
        decode_goal_change(_change("create", goal=goal))


# ---------------------------------------------------------------------------
# Fold transitions
# ---------------------------------------------------------------------------


def test_fold_create_then_edit() -> None:
    events = [
        _event("goal/change", _change("create", goal=_snapshot()), seq=0),
        _event(
            "goal/change",
            _change(
                "edit",
                goal=_snapshot(revision=2, objective="ship it now"),
                updated=1100,
            ),
            seq=1,
        ),
    ]
    state = fold_goal(events)
    assert state.goal is not None
    assert state.goal.revision == 2
    assert state.goal.objective == "ship it now"


def test_fold_rejects_create_when_active_goal_exists() -> None:
    state = GoalFoldState()
    apply_goal_change(
        state,
        decode_goal_change(_change("create", goal=_snapshot())),
    )
    with pytest.raises(GoalError, match="fresh active revision-one goal"):
        apply_goal_change(
            state,
            decode_goal_change(_change("create", goal=_snapshot(goal_id="g2"))),
        )


def test_fold_allows_create_after_complete() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_change(
        state,
        decode_goal_change(
            _change(
                "complete", goal=_snapshot(revision=2, phase="complete"), updated=1100
            )
        ),
    )
    apply_goal_change(
        state,
        decode_goal_change(_change("create", goal=_snapshot(goal_id="g2"))),
    )
    assert state.goal is not None
    assert state.goal.id == "g2"


def test_fold_rejects_reused_goal_id() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_change(
        state,
        decode_goal_change(
            _change(
                "complete", goal=_snapshot(revision=2, phase="complete"), updated=1100
            )
        ),
    )
    with pytest.raises(GoalError, match="fresh active revision-one goal"):
        apply_goal_change(
            state, decode_goal_change(_change("create", goal=_snapshot()))
        )


def test_fold_rejects_clear_without_goal() -> None:
    state = GoalFoldState()
    with pytest.raises(GoalError, match="requires a current goal"):
        apply_goal_change(
            state,
            decode_goal_change(
                {
                    "kind": "goal/change",
                    "version": 1,
                    "operation": "clear",
                    "cleared": {"id": "g1", "revision": 2},
                    "clearedAt": 5,
                }
            ),
        )


def test_fold_clear_resets_counters() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_change(
        state,
        decode_goal_change(
            {
                "kind": "goal/change",
                "version": 1,
                "operation": "clear",
                "cleared": {"id": "g1", "revision": 2},
                "clearedAt": 2000,
            }
        ),
    )
    assert state.goal is None
    assert state.rounds_started == 0
    assert state.created_at is None


def test_fold_clear_timestamp_cannot_precede_update() -> None:
    state = GoalFoldState()
    apply_goal_change(
        state, decode_goal_change(_change("create", goal=_snapshot(), updated=2000))
    )
    with pytest.raises(GoalError, match="cannot precede the current goal update"):
        apply_goal_change(
            state,
            decode_goal_change(
                {
                    "kind": "goal/change",
                    "version": 1,
                    "operation": "clear",
                    "cleared": {"id": "g1", "revision": 2},
                    "clearedAt": 1000,
                }
            ),
        )


def test_fold_revision_must_advance_by_one() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    with pytest.raises(GoalError, match="advance the current goal by one revision"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change("edit", goal=_snapshot(revision=3), updated=1100)
            ),
        )


def test_fold_edit_cannot_change_phase_or_reason() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    with pytest.raises(GoalError, match="cannot change phase or blocked reason"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change(
                    "edit", goal=_snapshot(revision=2, phase="paused"), updated=1100
                )
            ),
        )


def test_fold_pause_requires_active() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_change(
        state,
        decode_goal_change(
            _change("pause", goal=_snapshot(revision=2, phase="paused"), updated=1100)
        ),
    )
    with pytest.raises(GoalError, match="invalid phase transition"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change(
                    "pause", goal=_snapshot(revision=3, phase="paused"), updated=1200
                )
            ),
        )


def test_fold_pause_cannot_change_objective() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    with pytest.raises(GoalError, match="cannot change objective or maxGoalRounds"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change(
                    "pause",
                    goal=_snapshot(revision=2, phase="paused", objective="other"),
                    updated=1100,
                )
            ),
        )


def test_fold_resume_requires_capacity() -> None:
    state = GoalFoldState()
    apply_goal_change(
        state,
        decode_goal_change(_change("create", goal=_snapshot(max_rounds=1))),
    )
    apply_goal_change(
        state,
        decode_goal_change(
            _change(
                "pause",
                goal=_snapshot(phase="paused", revision=2, max_rounds=1),
                updated=1100,
            )
        ),
    )
    state.rounds_started = 1
    with pytest.raises(GoalError, match="invalid phase transition or exhausted"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change(
                    "resume",
                    goal=_snapshot(revision=3, max_rounds=1),
                    rounds=1,
                    updated=1200,
                )
            ),
        )


def test_fold_resume_from_paused() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_change(
        state,
        decode_goal_change(
            _change("pause", goal=_snapshot(phase="paused", revision=2), updated=1100)
        ),
    )
    apply_goal_change(
        state,
        decode_goal_change(_change("resume", goal=_snapshot(revision=3), updated=1200)),
    )
    assert state.goal is not None
    assert state.goal.phase == "active"


def test_fold_complete_twice_rejected() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_change(
        state,
        decode_goal_change(
            _change(
                "complete", goal=_snapshot(revision=2, phase="complete"), updated=1100
            )
        ),
    )
    with pytest.raises(GoalError, match="invalid phase transition"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change(
                    "complete",
                    goal=_snapshot(revision=3, phase="complete"),
                    updated=1200,
                )
            ),
        )


def test_fold_block_requires_active() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_change(
        state,
        decode_goal_change(
            _change("pause", goal=_snapshot(phase="paused", revision=2), updated=1100)
        ),
    )
    with pytest.raises(GoalError, match="invalid phase transition"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change(
                    "block",
                    goal=_snapshot(
                        phase="blocked",
                        revision=3,
                        blocked={"code": "x", "message": "m"},
                    ),
                    updated=1200,
                )
            ),
        )


def test_fold_snapshot_must_preserve_counters() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    with pytest.raises(GoalError, match="does not preserve the current counters"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change(
                    "edit",
                    goal=_snapshot(revision=2),
                    rounds=1,
                    updated=1100,
                )
            ),
        )


def test_fold_event_rejects_bad_kind() -> None:
    state = GoalFoldState()
    with pytest.raises(GoalError, match="has an invalid kind"):
        apply_goal_event(
            state,
            _event("goal/change", {"kind": "not-goal"}, seq=3),
        )


# ---------------------------------------------------------------------------
# Goal round counting
# ---------------------------------------------------------------------------


def _round_source(
    goal_id: str = "g1", revision: int = 1, round_number: int = 1
) -> dict[str, Any]:
    return {
        "kind": "goal",
        "goalId": goal_id,
        "revision": revision,
        "round": round_number,
    }


def test_fold_advances_round_counter() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_event(
        state,
        _event("user/message", {"source": _round_source(), "content": []}, seq=1),
    )
    assert state.rounds_started == 1


def test_fold_rejects_out_of_order_round() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    with pytest.raises(GoalError, match="not the next admitted round"):
        apply_goal_event(
            state,
            _event("user/message", {"source": _round_source(round_number=2)}, seq=1),
        )


def test_fold_rejects_nonin_goal_message() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_event(state, _event("user/message", {"source": {"kind": "user"}}, seq=1))
    assert state.rounds_started == 0


def test_fold_rejects_invalid_round_source() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    with pytest.raises(GoalError, match="goal message source is invalid"):
        apply_goal_event(
            state,
            _event("user/message", {"source": {"kind": "goal", "goalId": ""}}, seq=1),
        )


def test_fold_rejects_round_beyond_cap() -> None:
    state = GoalFoldState()
    apply_goal_change(
        state, decode_goal_change(_change("create", goal=_snapshot(max_rounds=1)))
    )
    state.rounds_started = 1
    with pytest.raises(GoalError, match="not the next admitted round"):
        apply_goal_event(
            state,
            _event("user/message", {"source": _round_source(round_number=2)}, seq=1),
        )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_create_get_pause_resume_complete(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    goals: GoalService = ctx.goals

    assert goals.get(agent) is None
    created = goals.create(agent, "ship the port")
    assert created.phase == "active"
    assert created.revision == 1
    assert created.activation == "armed"
    assert goals.get(agent) is not None

    paused = goals.pause(agent, GoalRef(id=created.id, revision=created.revision))
    assert paused.phase == "paused"
    assert paused.activation == "disarmed"

    resumed = goals.resume(agent, GoalRef(id=paused.id, revision=paused.revision))
    assert resumed.phase == "active"
    assert resumed.activation == "armed"

    completed = goals.complete(agent, GoalRef(id=resumed.id, revision=resumed.revision))
    assert completed.phase == "complete"


@pytest.mark.asyncio
async def test_service_rejects_second_active_goal(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    ctx.goals.create(agent, "first")
    with pytest.raises(GoalError, match="already exists"):
        ctx.goals.create(agent, "second")


@pytest.mark.asyncio
async def test_service_create_after_complete_allowed(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    first = ctx.goals.create(agent, "first")
    done = ctx.goals.complete(agent, GoalRef(id=first.id, revision=1))
    second = ctx.goals.create(agent, "second")
    assert second.id != done.id


@pytest.mark.asyncio
async def test_service_edit_and_clear(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    edited = ctx.goals.edit(
        agent, GoalRef(id=created.id, revision=1), objective="better objective"
    )
    assert edited.objective == "better objective"
    assert edited.revision == 2
    tombstone = ctx.goals.clear(agent, GoalRef(id=edited.id, revision=edited.revision))
    assert tombstone.revision == 3
    assert ctx.goals.get(agent) is None


@pytest.mark.asyncio
async def test_service_block_and_resume(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    blocked = ctx.goals.block(
        agent,
        GoalRef(id=created.id, revision=1),
        GoalBlockReason(code="model-reported", message="needs credentials"),
    )
    assert blocked.phase == "blocked"
    assert blocked.blocked_reason == GoalBlockReason(
        code="model-reported", message="needs credentials"
    )
    resumed = ctx.goals.resume(agent, GoalRef(id=blocked.id, revision=2))
    assert resumed.phase == "active"


@pytest.mark.asyncio
async def test_service_disarm(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    ctx.goals.create(agent, "objective")
    disarmed = ctx.goals.disarm(agent)
    assert disarmed is not None
    assert disarmed.activation == "disarmed"


@pytest.mark.asyncio
async def test_service_stale_revision_rejected(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    with pytest.raises(GoalError, match="is not the current revision"):
        ctx.goals.pause(agent, GoalRef(id=created.id, revision=99))


@pytest.mark.asyncio
async def test_service_missing_goal_rejected(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    with pytest.raises(GoalError, match="no current goal"):
        ctx.goals.pause(agent, GoalRef(id="g1", revision=1))


@pytest.mark.asyncio
async def test_service_edit_requires_a_field(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    with pytest.raises(GoalError, match="requires objective and/or maxGoalRounds"):
        ctx.goals.edit(agent, GoalRef(id=created.id, revision=1))


@pytest.mark.asyncio
async def test_service_resume_rejects_armed_active(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    with pytest.raises(GoalError, match="already active and armed"):
        ctx.goals.resume(agent, GoalRef(id=created.id, revision=1))


@pytest.mark.asyncio
async def test_service_resume_rejects_complete(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    done = ctx.goals.complete(agent, GoalRef(id=created.id, revision=1))
    with pytest.raises(GoalError, match="cannot resume"):
        ctx.goals.resume(agent, GoalRef(id=done.id, revision=done.revision))


@pytest.mark.asyncio
async def test_service_requires_live_agent(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    with pytest.raises(GoalError, match="exact live agent"):
        ctx.goals.get(_FakeAgent())


@pytest.mark.asyncio
async def test_service_create_validates_objective(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    with pytest.raises(GoalError, match="non-empty string"):
        ctx.goals.create(agent, "   ")
    with pytest.raises(GoalError, match="normalized"):
        ctx.goals.create(agent, " padded ")


@pytest.mark.asyncio
async def test_service_create_validates_round_cap(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    with pytest.raises(GoalError, match="positive safe integer"):
        ctx.goals.create(agent, "objective", 0)


@pytest.mark.asyncio
async def test_service_block_validates_reason(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    with pytest.raises(GoalError, match="reason message must be non-empty"):
        ctx.goals.block(
            agent,
            GoalRef(id=created.id, revision=1),
            GoalBlockReason(code="x", message=" padded "),
        )


@pytest.mark.asyncio
async def test_projection_clears_after_clear(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    ctx.goals.clear(agent, GoalRef(id=created.id, revision=1))
    snapshot = ctx.sessionProjections.snapshot(agent.session, keys=["goal"])
    assert snapshot["values"]["goal"] is None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class _FakeAgent:
    id = "fake"


@pytest.mark.asyncio
async def test_get_goal_returns_null_without_goal(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    result = await ctx.tools.execute(
        ToolCallData(call_id="g1", name="get_goal", arguments={}),
        session=agent.session,
        signal=AbortSignal(),
    )
    assert result.is_error is True or result.content == {"goal": None}


@pytest.mark.asyncio
async def test_goal_tools_mounted_by_base(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    for name in ("get_goal", "create_goal", "update_goal"):
        assert ctx.tools.get(name) is not None
    prompt = await ctx.systemPrompt.assemble()
    section_text = "\n".join(
        str(section.get("text", "")) for section in prompt["sections"]
    )
    assert "create_goal may infer goal intent" in section_text


def test_apply_rejects_bad_threshold() -> None:
    from magitech_harness.goal.tools import apply as apply_goal_tools

    with pytest.raises(TypeError, match="positive safe integer"):
        apply_goal_tools(None, blocked_after_consecutive_rounds=0)


# ---------------------------------------------------------------------------
# Projection unit
# ---------------------------------------------------------------------------


def test_projection_state_validator_rejects_bad_shapes() -> None:
    from magitech_harness.goal.projection import _validate_state

    good = {"current": None, "seenGoalIds": [], "failure": None}
    assert _validate_state(good) == good
    for bad, message in [
        ("nope", "must be an object"),
        ({"current": None}, "must have exactly"),
        (
            {"current": None, "seenGoalIds": ["a", "a"], "failure": None},
            "must be unique",
        ),
        (
            {"current": None, "seenGoalIds": [], "failure": ""},
            "non-empty string or null",
        ),
    ]:
        with pytest.raises((TypeError, ValueError), match=message):
            _validate_state(bad)


def test_projection_value_validator_rejects_bad_goal() -> None:
    from magitech_harness.goal.projection import _validate_projection

    assert _validate_projection(None) is None
    with pytest.raises(TypeError, match="must be an object or null"):
        _validate_projection("x")
    base = {
        "goal": _snapshot(),
        "roundsStarted": 0,
        "createdAt": 1,
        "updatedAt": 1,
    }
    assert _validate_projection(base) is not None
    for override, message in [
        ({"roundsStarted": -1}, "non-negative integer"),
        ({"createdAt": -1}, "non-negative integer"),
        ({"updatedAt": 0}, "cannot precede its creation"),
        ({"roundsStarted": 999}, "cannot exceed its configured limit"),
    ]:
        with pytest.raises(ValueError, match=message):
            _validate_projection({**base, **override})
    for goal_override, message in [
        ({"id": ""}, "non-empty string"),
        ({"objective": ""}, "non-empty string"),
        ({"phase": "x"}, "phase is invalid"),
        ({"revision": 0}, "positive integer"),
        ({"maxGoalRounds": 0}, "positive integer"),
    ]:
        with pytest.raises(ValueError, match=message):
            _validate_projection({**base, "goal": {**_snapshot(), **goal_override}})
    blocked = {
        **base,
        "goal": {
            **_snapshot(phase="blocked"),
            "blockedReason": {"code": "", "message": "m"},
        },
    }
    with pytest.raises(ValueError, match="blockedReason.code"):
        _validate_projection(blocked)


def test_projection_apply_records_replay_failure() -> None:
    from magitech_harness.goal.projection import GOAL_PROJECTION

    state = GOAL_PROJECTION.init({}, 0)
    # A malformed change records the failure instead of raising out of the fold.
    state = GOAL_PROJECTION.apply(
        state,
        _event(
            "goal/change",
            {"kind": "goal/change", "version": 1, "operation": "create"},
            seq=0,
        ),
    )
    assert state["failure"] is not None
    # A later event leaves a failed state untouched.
    assert GOAL_PROJECTION.apply(state, _event("turn/start", {}, seq=1)) is state


def test_projection_ignores_unrelated_events() -> None:
    from magitech_harness.goal.projection import GOAL_PROJECTION

    state = GOAL_PROJECTION.init({}, 0)
    assert GOAL_PROJECTION.apply(state, _event("turn/start", {}, seq=0)) is state


def test_snapshot_roundtrip() -> None:
    from magitech_harness.goal.projection import (
        projection_to_snapshot,
        snapshot_to_projection,
    )

    snapshot = GoalSnapshot(
        id="g1",
        revision=2,
        objective="o",
        phase="blocked",
        max_goal_rounds=10,
        blocked_reason=GoalBlockReason(code="x", message="m"),
    )
    value = snapshot_to_projection(snapshot)
    assert projection_to_snapshot(value) == snapshot


import json

from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.service import LlmAdapter


def _scripted_adapter(tool: str, holder: dict[str, Any]) -> Any:
    """Adapter that calls one tool on its first step, then answers."""

    class _Adapter(LlmAdapter):
        def __init__(self) -> None:
            self.calls = 0

        async def list_models(self, provider: str) -> list[dict[str, Any]]:
            return [{"id": "m1", "name": "M1", "provider": provider}]

        async def stream(self, options: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                yield {"type": "block-start", "index": 0, "blockType": "tool-call"}
                yield {
                    "type": "tool-call-delta",
                    "index": 0,
                    "id": "call-1",
                    "name": holder["tool"],
                    "argumentsDelta": json.dumps(holder["arguments"]),
                }
                yield {"type": "finish", "reason": {"kind": "tool-calls"}}
                return
            yield {"type": "block-start", "index": 0, "blockType": "text"}
            yield {"type": "text-delta", "index": 0, "text": "done"}
            yield {
                "type": "block-end",
                "index": 0,
                "block": {"type": "text", "text": "done"},
            }
            yield {"type": "finish", "reason": {"kind": "stop"}}

    adapter = _Adapter()
    holder["adapter"] = adapter
    return adapter


async def _run_tool_turn(
    ctx: Any,
    tool: str,
    arguments: dict[str, Any],
    prompt: str,
    *,
    source: str = "user",
    source_extra: dict[str, Any] | None = None,
    holder: dict[str, Any] | None = None,
    agent: Any = None,
) -> Any:
    if holder is None or "adapter" not in holder:
        target = holder if holder is not None else {}
        target.update({"tool": tool, "arguments": arguments, "adapter": None})
        holder = target
        ctx.llm.register_adapter(["scripted"], _scripted_adapter(tool, holder))
    else:
        holder["tool"] = tool
        holder["arguments"] = arguments
        holder["adapter"].calls = 0
    if agent is None:
        handle = await ctx.agents.create(
            {"agentOptions": {"provider": "scripted", "model": "m1"}}
        )
        agent = handle["agent"]
    payload_source = {"kind": source, **(source_extra or {})}
    agent.followup(
        create_user_message(
            content=[{"type": "text", "text": prompt}],
            source=payload_source,
        )
    )
    await agent.when_idle()
    return agent, holder


def _last_tool_result(agent: Any) -> str:
    results = [e for e in agent.session.events if e.type == "tool/result"]
    assert results, "the scripted turn logged no tool result"
    return str(results[-1].payload)


# ---------------------------------------------------------------------------
# Tools inside a live turn (authority paths)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_goal_tool_requires_human_turn(tmp_path: Path) -> None:
    """A non-human turn has no create authority."""
    ctx = await _boot(tmp_path)
    agent, _holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "ship"}, "hi", source="model"
    )
    assert "direct human turn" in _last_tool_result(agent)
    assert ctx.goals.get(agent) is None


@pytest.mark.asyncio
async def test_create_goal_tool_from_human_turn(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent, _holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "ship the port"}, "make a goal"
    )
    view = ctx.goals.get(agent)
    assert view is not None
    assert view.objective == "ship the port"
    assert view.activation == "armed"


@pytest.mark.asyncio
async def test_get_goal_tool_from_human_turn(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent, _holder = await _run_tool_turn(ctx, "get_goal", {}, "read the goal")
    assert "goal" in _last_tool_result(agent)


@pytest.mark.asyncio
async def test_update_goal_tool_edit(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    seeded = ctx.sessions.create("seed-edit")
    del seeded
    holder: dict[str, Any] = {}
    agent, _ = await _run_tool_turn(
        ctx,
        "update_goal",
        {"goal_id": "placeholder", "revision": 1, "action": "edit", "objective": "x"},
        "edit it",
        source="model",
    )
    del holder
    # A non-human turn is rejected before touching the goal.
    assert "direct human turn" in _last_tool_result(agent)


@pytest.mark.asyncio
async def test_update_goal_tool_complete_from_human(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    probe_session = ctx.sessions.create("probe")
    del probe_session
    # Create the goal on a live agent, then complete it from a human turn.
    setup = await ctx.agents.create({})
    setup_agent = setup["agent"]
    created = ctx.goals.create(setup_agent, "objective")
    ctx.goals.disarm(setup_agent)

    agent, holder = await _run_tool_turn(
        ctx,
        "update_goal",
        {
            "goal_id": created.id,
            "revision": created.revision,
            "action": "complete",
        },
        "finish it",
    )
    del agent, holder


@pytest.mark.asyncio
async def test_goal_tool_requires_calling_agent(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    result = await ctx.tools.execute(
        ToolCallData(call_id="g9", name="get_goal", arguments={}),
        session=None,
        signal=AbortSignal(),
    )
    assert result.is_error is True
    assert "calling agent" in str(result.content)


# ---------------------------------------------------------------------------
# Remaining validator and guard branches
# ---------------------------------------------------------------------------


def test_validate_block_reason_rejects_shape() -> None:
    from magitech_harness.goal.projection import _validate_block_reason

    with pytest.raises(ValueError, match="exactly code and message"):
        _validate_block_reason({"code": "x"})
    with pytest.raises(ValueError, match="code must be a non-empty string"):
        _validate_block_reason({"code": 1, "message": "m"})
    with pytest.raises(ValueError, match="message must be a non-empty string"):
        _validate_block_reason({"code": "x", "message": ""})


def test_validate_projection_rejects_key_set() -> None:
    from magitech_harness.goal.projection import _validate_projection

    with pytest.raises(ValueError, match="must have exactly goal"):
        _validate_projection({"goal": _snapshot()})


def test_validate_state_rejects_non_list_seen_and_unretained_current() -> None:
    from magitech_harness.goal.projection import _validate_state

    with pytest.raises(ValueError, match="seenGoalIds must be an array"):
        _validate_state({"current": None, "seenGoalIds": "x", "failure": None})
    current = {
        "goal": _snapshot(goal_id="g1"),
        "roundsStarted": 0,
        "createdAt": 1,
        "updatedAt": 1,
    }
    with pytest.raises(ValueError, match="must be retained among seen goal ids"):
        _validate_state({"current": current, "seenGoalIds": [], "failure": None})


def test_projection_apply_short_circuits_after_failure() -> None:
    from magitech_harness.goal.projection import GOAL_PROJECTION

    state = GOAL_PROJECTION.init({}, 0)
    state = {
        "current": None,
        "seenGoalIds": [],
        "failure": "broken",
    }
    assert (
        GOAL_PROJECTION.apply(
            state, _event("goal/change", _change("create", goal=_snapshot()), seq=0)
        )
        is state
    )


def test_projection_from_fold_requires_timestamps() -> None:
    from magitech_harness.goal.projection import projection_from_fold

    with pytest.raises(ValueError, match="lacks timestamps"):
        projection_from_fold(GoalFoldState(goal=GoalSnapshot("g", 1, "o", "active", 5)))


def test_decode_block_reason_and_snapshot_shapes() -> None:
    with pytest.raises(GoalError, match="must have exactly code and message"):
        decode_goal_change(
            _change(
                "block",
                goal={
                    **_snapshot(phase="blocked"),
                    "blockedReason": "nope",
                },
            )
        )
    with pytest.raises(GoalError, match="goal must be a record"):
        decode_goal_change(_change("create", goal="nope"))
    with pytest.raises(GoalError, match="goal.id must be a non-empty string"):
        decode_goal_change(_change("create", goal={**_snapshot(), "id": ""}))
    with pytest.raises(GoalError, match="must be non-empty and normalized"):
        decode_goal_change(
            _change("create", goal={**_snapshot(), "objective": " padded"})
        )
    with pytest.raises(GoalError, match="tombstone must have exactly"):
        decode_goal_change(
            {
                "kind": "goal/change",
                "version": 1,
                "operation": "clear",
                "cleared": {"id": "g1"},
                "clearedAt": 1,
            }
        )
    with pytest.raises(GoalError, match="tombstone id must be a non-empty string"):
        decode_goal_change(
            {
                "kind": "goal/change",
                "version": 1,
                "operation": "clear",
                "cleared": {"id": "", "revision": 2},
                "clearedAt": 1,
            }
        )


def test_fold_pause_from_non_active() -> None:
    state = GoalFoldState()
    apply_goal_change(state, decode_goal_change(_change("create", goal=_snapshot())))
    apply_goal_change(
        state,
        decode_goal_change(
            _change(
                "complete", goal=_snapshot(revision=2, phase="complete"), updated=1100
            )
        ),
    )
    with pytest.raises(GoalError, match="goal pause has an invalid phase transition"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change(
                    "pause", goal=_snapshot(revision=3, phase="paused"), updated=1200
                )
            ),
        )


def test_fold_rejects_unvalidated_operation() -> None:
    from magitech_harness.goal.types import (
        GoalSnapshotChange,
        _validate_snapshot_transition,
    )

    state = GoalFoldState(
        goal=GoalSnapshot("g", 1, "o", "active", 5),
        created_at=1,
        updated_at=1,
    )
    change = GoalSnapshotChange(
        operation="mystery",
        goal=GoalSnapshot("g", 2, "o", "active", 5),
        rounds_started=0,
        created_at=1,
        updated_at=1,
    )
    with pytest.raises(GoalError, match="cannot be validated as a current-goal"):
        _validate_snapshot_transition(state, change, state.goal)  # type: ignore[arg-type]


def test_fold_non_create_requires_current_goal() -> None:
    state = GoalFoldState()
    with pytest.raises(GoalError, match="requires a current goal"):
        apply_goal_change(
            state,
            decode_goal_change(
                _change("edit", goal=_snapshot(revision=2), updated=1100)
            ),
        )


def test_service_objective_must_be_string() -> None:
    from magitech_harness.goal.service import _resolve_objective

    with pytest.raises(GoalError, match="must be a string"):
        _resolve_objective(42)
    with pytest.raises(GoalError, match="non-empty string"):
        _resolve_objective("   ")
    with pytest.raises(GoalError, match="normalized"):
        _resolve_objective(" padded ")


def test_service_block_reason_type_checked() -> None:
    from magitech_harness.goal.service import _resolve_block_reason

    with pytest.raises(GoalError, match="reason is invalid"):
        _resolve_block_reason("nope")
    with pytest.raises(GoalError, match="code must be a non-empty string"):
        _resolve_block_reason(GoalBlockReason(code="", message="m"))
    with pytest.raises(GoalError, match="message must be non-empty"):
        _resolve_block_reason(GoalBlockReason(code="x", message=" m "))


@pytest.mark.asyncio
async def test_service_init_requires_registry(tmp_path: Path) -> None:
    from animas import Context

    from magitech_harness.goal.service import GoalService

    ctx = Context()
    service = GoalService(ctx, {})
    with pytest.raises((RuntimeError, TypeError)):
        await service.init()


@pytest.mark.asyncio
async def test_service_state_requires_handle() -> None:
    from animas import Context

    from magitech_harness.goal.service import GoalService

    service = GoalService(Context(), {})
    with pytest.raises(RuntimeError, match="requires the sessionProjections"):
        service._state_of(object())


@pytest.mark.asyncio
async def test_service_disarm_without_goal(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    assert ctx.goals.disarm(agent) is None


@pytest.mark.asyncio
async def test_service_requires_agent_registry() -> None:
    from animas import Context

    from magitech_harness.goal.service import GoalService

    service = GoalService(Context(), {})
    with pytest.raises(GoalError, match="requires the agent registry"):
        service._assert_live(_FakeAgent())


@pytest.mark.asyncio
async def test_service_block_from_paused_rejected(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    paused = ctx.goals.pause(agent, GoalRef(created.id, 1))
    with pytest.raises(GoalError, match="cannot block"):
        ctx.goals.block(
            agent,
            GoalRef(paused.id, paused.revision),
            GoalBlockReason(code="x", message="m"),
        )


@pytest.mark.asyncio
async def test_service_resume_exhausted_budget(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective", 1)
    agent.session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "round one"}],
            source={
                "kind": "goal",
                "goalId": created.id,
                "revision": created.revision,
                "round": 1,
            },
        ),
        surface_op="append",
    )
    active = ctx.goals.get(agent)
    assert active is not None and active.rounds_started == 1
    paused = ctx.goals.pause(agent, GoalRef(active.id, active.revision))
    with pytest.raises(GoalError, match="exhausted 1 goal rounds"):
        ctx.goals.resume(agent, GoalRef(paused.id, paused.revision))


@pytest.mark.asyncio
async def test_service_commit_without_projection(tmp_path: Path) -> None:
    from magitech_harness.goal.service import GoalService

    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    service = GoalService(ctx, {})
    # Mount the projection so the agent path is valid, then drop the handle.
    await service.init()
    service._projection = lambda _session: None  # type: ignore[method-assign]
    with pytest.raises(GoalError, match="left no current goal"):
        service.create(agent, "objective")


def test_goal_value_includes_blocked_reason() -> None:
    from magitech_harness.goal.tools import _goal_value

    view = GoalView(
        goal=GoalSnapshot(
            "g",
            2,
            "o",
            "blocked",
            5,
            GoalBlockReason(code="x", message="m"),
        ),
        rounds_started=1,
        created_at=1,
        updated_at=2,
        activation="disarmed",
    )
    value = _goal_value(view)
    assert value["goal"]["blockedReason"] == {"code": "x", "message": "m"}


def test_goal_ref_rejects_bad_values() -> None:
    from magitech_harness.goal.tools import _goal_ref

    with pytest.raises(HarnessError, match="goal_id must be non-empty"):
        _goal_ref(" ", 1)
    with pytest.raises(HarnessError, match="goal_id must be non-empty"):
        _goal_ref("g", 0)


def test_goal_execution_guards(tmp_path: Path) -> None:
    from animas import Context

    from magitech_harness.goal.tools import _goal_tool_execution

    ctx = create_harness(storage_dir=tmp_path)
    with pytest.raises(HarnessError, match="require a calling agent"):
        _goal_tool_execution(ctx, {})
    bare = Context()
    with pytest.raises(HarnessError, match="require the agent registry"):
        _goal_tool_execution(bare, {"agent": _FakeAgent()})


@pytest.mark.asyncio
async def test_goal_execution_requires_live_agent_and_open_turn(tmp_path: Path) -> None:
    from magitech_harness.goal.tools import _goal_tool_execution

    ctx = await _boot(tmp_path)
    with pytest.raises(HarnessError, match="exact live calling agent"):
        _goal_tool_execution(ctx, {"agent": _FakeAgent()})
    agent = await _live_agent(ctx)
    with pytest.raises(HarnessError, match="require an open model turn"):
        _goal_tool_execution(ctx, {"agent": agent})


def test_direct_human_guards() -> None:
    from animas import Context

    from magitech_harness.goal.tools import _Execution, _has_direct_human_input

    bare = Context()
    execution = _Execution(agent=_FakeAgent(), events=[], open_turn_start_seq=None)  # type: ignore[arg-type]
    assert _has_direct_human_input(bare, execution) is False
    fake_with_session = _FakeAgent()
    fake_with_session.session = type("S", (), {"header": {"origin": "subagent"}})()
    execution = _Execution(agent=fake_with_session, events=[], open_turn_start_seq=None)  # type: ignore[arg-type]
    assert _has_direct_human_input(bare, execution) is False


def test_wrapup_text_variants() -> None:
    from magitech_harness.goal.tools import _wrapup_text

    assert "goal is complete" in _wrapup_text("o")
    assert "goal is blocked" in _wrapup_text("o", "stuck")


def test_completion_authority_rejects() -> None:
    from magitech_harness.goal.tools import _completion_authority, _Execution

    class _NoGoals:
        def get(self, agent: Any) -> None:
            return None

    execution = _Execution(agent=_FakeAgent(), events=[], open_turn_start_seq=0)
    with pytest.raises(HarnessError, match="complete and blocked require"):
        _completion_authority(
            type("C", (), {"get": lambda self, k, d=None, strict=False: None})(),
            execution,
            _NoGoals(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_create_goal_without_round_cap(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent, _holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "no cap"}, "make a goal"
    )
    view = ctx.goals.get(agent)
    assert view is not None
    assert view.max_goal_rounds == 256


@pytest.mark.asyncio
async def test_update_goal_edit_from_human(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent, holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "objective"}, "make a goal"
    )
    created = ctx.goals.get(agent)
    assert created is not None
    await _run_tool_turn(
        ctx,
        "update_goal",
        {
            "goal_id": created.id,
            "revision": created.revision,
            "action": "edit",
            "objective": "edited",
        },
        "edit it",
        holder=holder,
        agent=agent,
    )
    view = ctx.goals.get(agent)
    assert view is not None
    assert view.objective == "edited"


@pytest.mark.asyncio
async def test_update_goal_pause_from_human(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent, holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "objective"}, "make a goal"
    )
    created = ctx.goals.get(agent)
    assert created is not None
    await _run_tool_turn(
        ctx,
        "update_goal",
        {"goal_id": created.id, "revision": created.revision, "action": "pause"},
        "pause it",
        holder=holder,
        agent=agent,
    )
    paused = ctx.goals.get(agent)
    assert paused is not None
    assert paused.phase == "paused"


@pytest.mark.asyncio
async def test_update_goal_bad_revision(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent, _holder = await _run_tool_turn(
        ctx,
        "update_goal",
        {"goal_id": "g", "revision": -1, "action": "complete"},
        "go",
    )
    assert "positive safe" in _last_tool_result(agent)


@pytest.mark.asyncio
async def test_complete_from_goal_round_authority(tmp_path: Path) -> None:
    """The exact admitted goal round may complete without a human turn."""
    ctx = await _boot(tmp_path)
    holder: dict[str, Any] = {}
    agent, holder = await _run_tool_turn(
        ctx, "get_goal", {}, "bootstrap", holder=holder
    )
    created = ctx.goals.create(agent, "objective")
    await _run_tool_turn(
        ctx,
        "update_goal",
        {
            "goal_id": created.id,
            "revision": created.revision,
            "action": "complete",
        },
        "round",
        source="goal",
        source_extra={
            "goalId": created.id,
            "revision": created.revision,
            "round": 1,
        },
        holder=holder,
        agent=agent,
    )
    view = ctx.goals.get(agent)
    assert view is not None
    assert view.phase == "complete"


@pytest.mark.asyncio
async def test_blocked_from_goal_round_below_threshold(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    holder: dict[str, Any] = {}
    agent, holder = await _run_tool_turn(
        ctx, "get_goal", {}, "bootstrap", holder=holder
    )
    created = ctx.goals.create(agent, "objective")
    await _run_tool_turn(
        ctx,
        "update_goal",
        {
            "goal_id": created.id,
            "revision": created.revision,
            "action": "blocked",
            "blocked_reason": "stuck",
        },
        "round",
        source="goal",
        source_extra={
            "goalId": created.id,
            "revision": created.revision,
            "round": 1,
        },
        holder=holder,
        agent=agent,
    )
    assert "blocked requires at least 3" in _last_tool_result(agent)


@pytest.mark.asyncio
async def test_matching_goal_round_helper(tmp_path: Path) -> None:
    from magitech_harness.goal.tools import (
        _Execution,
        _is_matching_goal_round,
        _some_open_turn_event,
    )

    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    goal = ctx.goals.create(agent, "objective")
    agent.session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "round"}],
            source={
                "kind": "goal",
                "goalId": goal.id,
                "revision": goal.revision,
                "round": 1,
            },
        ),
        surface_op="append",
    )
    events = agent.session.snapshot_events()
    execution = _Execution(
        agent=agent,
        events=events,
        open_turn_start_seq=0,
    )
    current = ctx.goals.get(agent)
    assert current is not None
    assert _is_matching_goal_round(execution, current) is True
    assert _some_open_turn_event(execution, lambda _e: False) is False


def test_validate_goal_rejects_non_dict() -> None:
    from magitech_harness.goal.projection import _validate_goal

    with pytest.raises(TypeError, match="goal must be an object"):
        _validate_goal("nope")


def test_decode_block_reason_direct() -> None:
    from magitech_harness.goal.types import _decode_block_reason

    with pytest.raises(GoalError, match="exactly code and message"):
        _decode_block_reason(["x"])


def test_is_kebab_rejects_empty_and_non_ascii_parts() -> None:
    from magitech_harness.goal.types import _is_kebab

    assert _is_kebab("") is False
    assert _is_kebab("a--b") is False
    assert _is_kebab("café-x") is False
    assert _is_kebab("ok-code") is True


def test_fold_transition_requires_updated_at() -> None:
    from magitech_harness.goal.types import (
        GoalSnapshotChange,
        _validate_snapshot_transition,
    )

    state = GoalFoldState(
        goal=GoalSnapshot("g", 1, "o", "active", 5),
        created_at=1,
        updated_at=None,
    )
    change = GoalSnapshotChange(
        operation="pause",
        goal=GoalSnapshot("g", 2, "o", "paused", 5),
        rounds_started=0,
        created_at=1,
        updated_at=2,
    )
    with pytest.raises(GoalError, match="lacks updatedAt"):
        _validate_snapshot_transition(state, change, state.goal)  # type: ignore[arg-type]


def test_fold_clear_requires_updated_at() -> None:
    from magitech_harness.goal.types import GoalClearChange

    state = GoalFoldState(
        goal=GoalSnapshot("g", 1, "o", "active", 5),
        created_at=1,
        updated_at=None,
    )
    with pytest.raises(GoalError, match="lacks updatedAt"):
        apply_goal_change(state, GoalClearChange(cleared=GoalRef("g", 2), cleared_at=5))


def test_fold_ignores_unrelated_event_types() -> None:
    state = GoalFoldState()
    apply_goal_event(state, _event("turn/start", {}, seq=0))
    assert state.goal is None


@pytest.mark.asyncio
async def test_service_pause_twice_rejected(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    created = ctx.goals.create(agent, "objective")
    paused = ctx.goals.pause(agent, GoalRef(created.id, 1))
    with pytest.raises(GoalError, match="cannot pause"):
        ctx.goals.pause(agent, GoalRef(paused.id, paused.revision))


def test_direct_human_rejects_subagent_origin() -> None:
    from animas import Context

    from magitech_harness.goal.tools import _Execution, _has_direct_human_input

    class _FakeAgents:
        def __init__(self, agent: Any) -> None:
            self._agent = agent

        def get(self, agent_id: str) -> Any:
            return self._agent if agent_id == self._agent.id else None

    class _Agent:
        id = "sub"

        def __init__(self) -> None:
            self.session = type("S", (), {"header": {"origin": "subagent"}})()

        @property
        def ctx(self) -> Any:
            return None

    agent = _Agent()
    ctx = Context()
    ctx.reflect.provide("agents", _FakeAgents(agent))
    execution = _Execution(agent=agent, events=[], open_turn_start_seq=0)  # type: ignore[arg-type]
    assert _has_direct_human_input(ctx, execution) is False


@pytest.mark.asyncio
async def test_model_cannot_resume_paused_via_tool(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    holder: dict[str, Any] = {}
    agent, holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "objective"}, "goal", holder=holder
    )
    created = ctx.goals.get(agent)
    assert created is not None
    paused = ctx.goals.pause(agent, GoalRef(created.id, created.revision))
    await _run_tool_turn(
        ctx,
        "update_goal",
        {"goal_id": paused.id, "revision": paused.revision, "action": "resume"},
        "resume",
        holder=holder,
        agent=agent,
    )
    assert "cannot resume a paused goal" in _last_tool_result(agent)


@pytest.mark.asyncio
async def test_blocked_without_reason_rejected(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    holder: dict[str, Any] = {}
    agent, holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "objective"}, "goal", holder=holder
    )
    created = ctx.goals.get(agent)
    assert created is not None
    await _run_tool_turn(
        ctx,
        "update_goal",
        {
            "goal_id": created.id,
            "revision": created.revision,
            "action": "blocked",
        },
        "block",
        holder=holder,
        agent=agent,
    )
    assert "blocked_reason is required" in _last_tool_result(agent)


@pytest.mark.asyncio
async def test_blocked_with_reason_from_human(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    holder: dict[str, Any] = {}
    agent, holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "objective"}, "goal", holder=holder
    )
    created = ctx.goals.get(agent)
    assert created is not None
    await _run_tool_turn(
        ctx,
        "update_goal",
        {
            "goal_id": created.id,
            "revision": created.revision,
            "action": "blocked",
            "blocked_reason": "needs credentials",
        },
        "block",
        holder=holder,
        agent=agent,
    )
    view = ctx.goals.get(agent)
    assert view is not None
    assert view.phase == "blocked"
    assert view.blocked_reason is not None
    assert view.blocked_reason.message == "needs credentials"


def test_validate_block_reason_accepts_valid() -> None:
    from magitech_harness.goal.projection import _validate_block_reason

    assert _validate_block_reason({"code": "x", "message": "m"}) is None


def test_non_negative_int_rejects_negative() -> None:
    from magitech_harness.goal.types import _non_negative_int

    with pytest.raises(GoalError, match="non-negative safe integer"):
        _non_negative_int(-1, "x")
    with pytest.raises(GoalError, match="non-negative safe integer"):
        _non_negative_int(True, "x")


@pytest.mark.asyncio
async def test_goal_disposer_without_listeners(tmp_path: Path) -> None:
    from animas import Context

    from magitech_harness.goal.service import GoalService
    from magitech_harness.session.projection_registry import SessionProjectionRegistry

    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    await ctx.flush()
    service = GoalService(ctx, {})
    dispose = await service.init()
    service._listen = None
    service._handle = None
    dispose()


def test_goal_apply_without_system_prompt() -> None:
    from animas import Context

    from magitech_harness.goal.tools import apply as apply_goal_tools
    from magitech_harness.tools.service import ToolsService

    class _FakeGoals:
        def get(self, agent: Any) -> None:
            return None

    ctx = Context()
    ToolsService(ctx)
    ctx.reflect.provide("goals", _FakeGoals())
    apply_goal_tools(ctx, blocked_after_consecutive_rounds=3)
    assert ctx.tools.get("create_goal") is not None


@pytest.mark.asyncio
async def test_complete_from_human_tool(tmp_path: Path) -> None:
    ctx = await _boot(tmp_path)
    holder: dict[str, Any] = {}
    agent, holder = await _run_tool_turn(
        ctx, "create_goal", {"objective": "objective"}, "goal", holder=holder
    )
    created = ctx.goals.get(agent)
    assert created is not None
    await _run_tool_turn(
        ctx,
        "update_goal",
        {
            "goal_id": created.id,
            "revision": created.revision,
            "action": "complete",
        },
        "complete",
        holder=holder,
        agent=agent,
    )
    view = ctx.goals.get(agent)
    assert view is not None
    assert view.phase == "complete"


@pytest.mark.asyncio
async def test_blocked_from_goal_round_at_threshold(tmp_path: Path) -> None:
    """After three admitted rounds the model may self-report blocked."""
    # This probe drives rounds by hand, so the automatic round driver is
    # disabled to keep its own reservations out of the sequence.
    ctx = create_harness(
        storage_dir=tmp_path,
        patches=[[{"id": "goal-round-driver", "disabled": True}]],
    )
    await ctx.flush()
    holder: dict[str, Any] = {}
    agent, holder = await _run_tool_turn(
        ctx, "get_goal", {}, "bootstrap", holder=holder
    )
    created = ctx.goals.create(agent, "objective")
    # Admit three goal rounds with a plain get_goal turn between them.
    for round_number in (1, 2, 3):
        await _run_tool_turn(
            ctx,
            "get_goal",
            {},
            f"round {round_number}",
            source="goal",
            source_extra={
                "goalId": created.id,
                "revision": created.revision,
                "round": round_number,
            },
            holder=holder,
            agent=agent,
        )
    current = ctx.goals.get(agent)
    assert current is not None and current.rounds_started == 3
    await _run_tool_turn(
        ctx,
        "update_goal",
        {
            "goal_id": created.id,
            "revision": created.revision,
            "action": "blocked",
            "blocked_reason": "same failure three rounds running",
        },
        "round 4",
        source="goal",
        source_extra={
            "goalId": created.id,
            "revision": created.revision,
            "round": 4,
        },
        holder=holder,
        agent=agent,
    )
    view = ctx.goals.get(agent)
    assert view is not None
    assert view.phase == "blocked"


# ---------------------------------------------------------------------------
# /goal command
# ---------------------------------------------------------------------------


def test_parse_goal_command_grammar() -> None:
    from magitech_harness.goal.command import parse_goal_command

    assert parse_goal_command("").kind == "show"
    assert parse_goal_command("  ").kind == "show"
    assert parse_goal_command("clear").kind == "clear"
    assert parse_goal_command("CLEAR").kind == "clear"
    assert parse_goal_command("pause").kind == "pause"
    assert parse_goal_command("resume").kind == "resume"
    assert parse_goal_command("edit").kind == "invalid-edit"
    edited = parse_goal_command("edit  new objective ")
    assert (edited.kind, edited.objective) == ("edit", "new objective")
    created = parse_goal_command("ship the port")
    assert (created.kind, created.objective) == ("create", "ship the port")
    assert parse_goal_command("editx").kind == "create"


def test_render_goal_blocked_and_complete_hints() -> None:
    from magitech_harness.goal.command import render_goal

    blocked = GoalView(
        goal=GoalSnapshot(
            "g", 2, "o", "blocked", 5, GoalBlockReason(code="x", message="m")
        ),
        rounds_started=1,
        created_at=1,
        updated_at=2,
        activation="disarmed",
    )
    text = render_goal("Blocked", blocked)
    assert "Blocker: x: m" in text
    assert "/goal resume" in text

    complete = GoalView(
        goal=GoalSnapshot("g", 3, "o", "complete", 5),
        rounds_started=2,
        created_at=1,
        updated_at=3,
        activation="disarmed",
    )
    assert "/goal <objective>" in render_goal("Complete", complete)

    active_disarmed = GoalView(
        goal=GoalSnapshot("g", 4, "o", "active", 5),
        rounds_started=2,
        created_at=1,
        updated_at=4,
        activation="disarmed",
    )
    assert "/goal resume" in render_goal("Active", active_disarmed)


def test_render_goal_rejects_blocked_without_reason() -> None:
    from magitech_harness.goal.command import render_goal

    view = GoalView(
        goal=GoalSnapshot("g", 2, "o", "blocked", 5),
        rounds_started=1,
        created_at=1,
        updated_at=2,
        activation="disarmed",
    )
    with pytest.raises(TypeError, match="missing its reason"):
        render_goal("Blocked", view)


@pytest.mark.asyncio
async def test_goal_command_create_show_edit_pause_resume_clear(tmp_path: Path) -> None:
    from magitech_harness.goal.command import run_goal_command

    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    session = agent.session

    assert "No goal is set" in run_goal_command(ctx, session, "")
    assert "Created goal" in run_goal_command(ctx, session, "ship the port")
    assert "Current goal" in run_goal_command(ctx, session, "")
    assert "Edited goal" in run_goal_command(ctx, session, "edit better objective")
    assert "Paused goal" in run_goal_command(ctx, session, "pause")
    assert "Resumed goal" in run_goal_command(ctx, session, "resume")
    assert "Cleared" in run_goal_command(ctx, session, "clear")
    assert "No goal is set" in run_goal_command(ctx, session, "")


@pytest.mark.asyncio
async def test_goal_command_missing_goal_paths(tmp_path: Path) -> None:
    from magitech_harness.goal.command import run_goal_command

    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    for action in ("pause", "resume", "clear"):
        assert "No goal is currently set" in run_goal_command(
            ctx, agent.session, action
        )
    assert "/goal edit requires an objective" in run_goal_command(
        ctx, agent.session, "edit"
    )


@pytest.mark.asyncio
async def test_goal_command_reports_domain_errors(tmp_path: Path) -> None:
    from magitech_harness.goal.command import run_goal_command

    ctx = await _boot(tmp_path)
    agent = await _live_agent(ctx)
    ctx.goals.create(agent, "objective")
    text = run_goal_command(ctx, agent.session, "another")
    assert "/goal:" in text
    assert "already exists" in text


def test_goal_command_requires_live_agent(tmp_path: Path) -> None:
    from magitech_harness.goal.command import run_goal_command

    ctx = create_harness(storage_dir=tmp_path)
    session = ctx.sessions.create("no-agent")
    assert "requires a live agent" in run_goal_command(ctx, session, "")


def test_goal_command_without_agents_service(tmp_path: Path) -> None:
    from animas import Context

    from magitech_harness.goal.command import _live_agent

    assert _live_agent(Context(), object()) is None
