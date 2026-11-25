"""The ``goal`` session projection: durable current goal plus round counter.

Ports ``goalProjectionDefinition`` from DSH
``packages/goal/goal/src/index.ts`` (fold in ``fold.ts``). The unit is
host-and-client visible (``summary: false``) and its state is
checkpoint-safe: ``current`` plus the retained goal ids and first replay
failure.
"""

from __future__ import annotations

from typing import Any

from magitech_harness.goal.types import (
    GoalBlockReason,
    GoalError,
    GoalFoldState,
    GoalProjection,
    GoalSnapshot,
    apply_goal_event,
)
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    WireView,
)


def _validate_block_reason(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"code", "message"}:
        raise ValueError("blockedReason must have exactly code and message")
    if not isinstance(value["code"], str) or not value["code"]:
        raise ValueError("blockedReason.code must be a non-empty string")
    if not isinstance(value["message"], str) or not value["message"]:
        raise ValueError("blockedReason.message must be a non-empty string")


def _validate_goal(value: Any) -> None:
    if not isinstance(value, dict):
        raise TypeError("goal must be an object")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError("goal.id must be a non-empty string")
    if not isinstance(value.get("objective"), str) or not value["objective"]:
        raise ValueError("goal.objective must be a non-empty string")
    if value.get("phase") not in ("active", "paused", "blocked", "complete"):
        raise ValueError("goal.phase is invalid")
    for key in ("revision", "maxGoalRounds"):
        if (
            isinstance(value.get(key), bool)
            or not isinstance(value.get(key), int)
            or value[key] < 1
        ):
            raise ValueError(f"goal.{key} must be a positive integer")
    if value["phase"] == "blocked":
        _validate_block_reason(value.get("blockedReason"))


def _validate_projection(value: Any) -> dict[str, Any] | None:
    """Validate one projection value (``None`` or a goal snapshot tuple)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("goal projection must be an object or null")
    if set(value) != {"goal", "roundsStarted", "createdAt", "updatedAt"}:
        raise ValueError(
            "goal projection must have exactly goal, roundsStarted, createdAt, "
            "updatedAt"
        )
    _validate_goal(value["goal"])
    rounds = value["roundsStarted"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise ValueError("roundsStarted must be a non-negative integer")
    for key in ("createdAt", "updatedAt"):
        stamp = value[key]
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if value["updatedAt"] < value["createdAt"]:
        raise ValueError("goal update cannot precede its creation")
    if rounds > value["goal"]["maxGoalRounds"]:
        raise ValueError("goal rounds cannot exceed its configured limit")
    return {
        "goal": dict(value["goal"]),
        "roundsStarted": rounds,
        "createdAt": value["createdAt"],
        "updatedAt": value["updatedAt"],
    }


def _validate_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("goal projection state must be an object")
    if set(value) != {"current", "seenGoalIds", "failure"}:
        raise ValueError(
            "goal projection state must have exactly current, seenGoalIds, failure"
        )
    current = _validate_projection(value["current"])
    seen = value["seenGoalIds"]
    if not isinstance(seen, list) or not all(
        isinstance(item, str) and item for item in seen
    ):
        raise ValueError("seenGoalIds must be an array of non-empty strings")
    if len(set(seen)) != len(seen):
        raise ValueError("seen goal ids must be unique")
    failure = value["failure"]
    if failure is not None and (not isinstance(failure, str) or not failure):
        raise ValueError("failure must be a non-empty string or null")
    if current is not None and current["goal"]["id"] not in seen:
        raise ValueError("current goal id must be retained among seen goal ids")
    return {"current": current, "seenGoalIds": list(seen), "failure": failure}


def snapshot_to_projection(goal: GoalSnapshot) -> dict[str, Any]:
    """Project one snapshot into the durable wire shape."""
    value: dict[str, Any] = {
        "id": goal.id,
        "revision": goal.revision,
        "objective": goal.objective,
        "phase": goal.phase,
        "maxGoalRounds": goal.max_goal_rounds,
    }
    if goal.blocked_reason is not None:
        value["blockedReason"] = {
            "code": goal.blocked_reason.code,
            "message": goal.blocked_reason.message,
        }
    return value


def projection_to_snapshot(value: dict[str, Any]) -> GoalSnapshot:
    """Read one validated projection value back into a snapshot."""
    blocked = value.get("blockedReason")
    return GoalSnapshot(
        id=value["id"],
        revision=value["revision"],
        objective=value["objective"],
        phase=value["phase"],
        max_goal_rounds=value["maxGoalRounds"],
        blocked_reason=(
            GoalBlockReason(code=blocked["code"], message=blocked["message"])
            if blocked is not None
            else None
        ),
    )


def _init_goal(header: dict[str, Any], inherited_event_count: int) -> dict[str, Any]:
    del header, inherited_event_count
    return {"current": None, "seenGoalIds": [], "failure": None}


def projection_from_fold(folded: GoalFoldState) -> dict[str, Any] | None:
    """Convert one folded goal state into the durable projection value.

    A fold that carries a goal must also carry its creation and update
    stamps; the decoder guarantees that pairing, so a mismatch is a defect.
    """
    if folded.goal is None:
        return None
    if folded.created_at is None or folded.updated_at is None:
        raise ValueError("current goal fold lacks timestamps")
    return {
        "goal": snapshot_to_projection(folded.goal),
        "roundsStarted": folded.rounds_started,
        "createdAt": folded.created_at,
        "updatedAt": folded.updated_at,
    }


def _apply_goal(state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
    if event.type != "goal/change" and event.type != "user/message":
        return state
    if state["failure"] is not None:
        return state
    folded = GoalFoldState(
        goal=(
            projection_to_snapshot(state["current"]["goal"])
            if state["current"] is not None
            else None
        ),
        rounds_started=(
            state["current"]["roundsStarted"] if state["current"] is not None else 0
        ),
        created_at=(
            state["current"]["createdAt"] if state["current"] is not None else None
        ),
        updated_at=(
            state["current"]["updatedAt"] if state["current"] is not None else None
        ),
        seen_goal_ids=set(state["seenGoalIds"]),
    )
    try:
        apply_goal_event(folded, event)
    except GoalError as error:
        return {
            "current": state["current"],
            "seenGoalIds": state["seenGoalIds"],
            "failure": str(error),
        }
    current = projection_from_fold(folded)
    return {
        "current": current,
        "seenGoalIds": sorted(folded.seen_goal_ids),
        "failure": None,
    }


def _goal_view(state: dict[str, Any]) -> Any:
    return state["current"]


GOAL_PROJECTION: ProjectionDefinition[dict[str, Any]] = ProjectionDefinition(
    key="goal",
    state_schema=_validate_state,
    init=_init_goal,
    apply=_apply_goal,
    wire=WireView(
        view_schema=_validate_projection,
        view=_goal_view,
        summary=False,
    ),
    state_version=1,
)


def current_projection(state: dict[str, Any]) -> GoalProjection | None:
    """Read the durable projection out of one validated projection state."""
    current = state["current"]
    if current is None:
        return None
    return GoalProjection(
        goal=projection_to_snapshot(current["goal"]),
        rounds_started=current["roundsStarted"],
        created_at=current["createdAt"],
        updated_at=current["updatedAt"],
    )


__all__ = [
    "GOAL_PROJECTION",
    "current_projection",
    "projection_from_fold",
    "projection_to_snapshot",
    "snapshot_to_projection",
]
