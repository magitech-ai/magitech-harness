"""Pure fold and client view for the ``agentTeam`` projection.

Ports ``packages/team/agent-team/src/projection.ts`` from the pinned dsh source.
The unit is durable-only (the wire view is host-and-client visible) and its
state is checkpoint-safe.
"""

from __future__ import annotations

from typing import Any

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    WireView,
)
from magitech_harness.team.task_graph import assert_task_graph, project_task_view
from magitech_harness.team.types import TeamError, task_number, task_view

VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "deleted"})


def _validate_task(task: Any) -> dict[str, Any]:
    if not isinstance(task, dict):
        raise TypeError("team task must be an object")
    if not isinstance(task.get("id"), str) or not task["id"]:
        raise ValueError("team task.id must be a non-empty string")
    number = task_number(task["id"])
    if number is not None and number < 1:
        raise ValueError("team task numeric suffix must be positive")
    if (
        isinstance(task.get("revision"), bool)
        or not isinstance(task.get("revision"), int)
        or task["revision"] < 1
    ):
        raise ValueError("team task.revision must be a positive integer")
    if not isinstance(task.get("subject"), str) or not isinstance(
        task.get("description"), str
    ):
        raise TypeError("team task subject and description must be strings")
    if task.get("status") not in VALID_STATUSES:
        raise ValueError("team task.status is invalid")
    for key in ("blockedBy", "writeScopes"):
        value = task.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"team task.{key} must be an array of strings")
    owner_id = task.get("ownerId")
    if owner_id is not None and (not isinstance(owner_id, str) or not owner_id):
        raise ValueError("team task.ownerId must be a non-empty string")
    owner_name = task.get("ownerName")
    if owner_name is not None and (not isinstance(owner_name, str) or not owner_name):
        raise ValueError("team task.ownerName must be a non-empty string")
    if owner_name is not None and owner_id is None:
        raise ValueError("team task.ownerName requires ownerId")
    return {
        "id": task["id"],
        "revision": task["revision"],
        "subject": task["subject"],
        "description": task["description"],
        "status": task["status"],
        "blockedBy": list(task["blockedBy"]),
        "writeScopes": list(task["writeScopes"]),
        **({} if owner_id is None else {"ownerId": owner_id}),
        **({} if owner_name is None else {"ownerName": owner_name}),
    }


def validate_state(value: Any) -> dict[str, Any]:
    """Validate one persisted Team projection value."""
    if not isinstance(value, dict):
        raise TypeError("agentTeam projection must be an object")
    expected = {
        "inheritedEventCount",
        "enabled",
        "id",
        "tasks",
        "nextTaskNumber",
        "failure",
    }
    if not set(value) <= expected:
        raise ValueError("agentTeam projection has unknown fields")
    if (
        "inheritedEventCount" not in value
        or "id" not in value
        or "tasks" not in value
        or "nextTaskNumber" not in value
    ):
        raise ValueError("agentTeam projection is missing required fields")
    inherited = value["inheritedEventCount"]
    if isinstance(inherited, bool) or not isinstance(inherited, int) or inherited < 0:
        raise ValueError("agentTeam.inheritedEventCount must be a non-negative integer")
    if not isinstance(value["id"], str) or not value["id"]:
        raise ValueError("agentTeam.id must be a non-empty string")
    if not isinstance(value["tasks"], list):
        raise TypeError("agentTeam.tasks must be an array")
    tasks = [_validate_task(task) for task in value["tasks"]]
    number = value["nextTaskNumber"]
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("agentTeam.nextTaskNumber must be a positive integer")
    enabled = value.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("agentTeam.enabled must be a boolean or absent")
    failure = value.get("failure")
    if failure is not None and not isinstance(failure, str):
        raise ValueError("agentTeam.failure must be a string or absent")
    return {
        "inheritedEventCount": inherited,
        **({} if enabled is None else {"enabled": enabled}),
        "id": value["id"],
        "tasks": tasks,
        "nextTaskNumber": number,
        **({} if failure is None else {"failure": failure}),
    }


def init_state(header: dict[str, Any], inherited_event_count: int) -> dict[str, Any]:
    """Construct empty state for one Team identity."""
    return {
        "inheritedEventCount": inherited_event_count,
        "id": header["id"],
        "tasks": [],
        "nextTaskNumber": 1,
    }


def apply_event(state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
    """Pure transition: previous state + one committed event -> next state."""
    try:
        if event.type == "team/mode":
            if event.seq < state["inheritedEventCount"]:
                return state
            enabled = event.payload.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError("team/mode.enabled must be a boolean")
            return {**state, "enabled": enabled}
        if state.get("failure") is not None or event.type != "team/task":
            return state
        data = event.payload
        if data.get("version") != 2:
            raise ValueError("unsupported Agent Teams event version")
        if data.get("teamId") != state["id"]:
            return state
        task = _validate_task(data.get("task"))
        return apply_task(state, task)
    except (ValueError, TypeError, TeamError) as error:
        return {
            **state,
            "failure": state.get("failure") or str(error),
        }


class TeamValueError(ValueError):
    """Raised when a task transition breaks a Team invariant."""


def apply_task(state: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Apply one validated task snapshot to Team state."""
    index = next(
        (
            position
            for position, candidate in enumerate(state["tasks"])
            if candidate["id"] == task["id"]
        ),
        -1,
    )
    prior = state["tasks"][index] if index >= 0 else None
    if prior is None and task["revision"] != 1:
        raise TeamValueError(f'team task "{task["id"]}" must begin at revision 1')
    if prior is not None and task["revision"] != prior["revision"] + 1:
        raise TeamValueError(f'team task "{task["id"]}" revision is not contiguous')
    assert_task_graph(state["tasks"], task)
    number = task_number(task["id"])
    next_number = state["nextTaskNumber"]
    if number is not None:
        next_number = max(next_number, number if number == 2**53 - 1 else number + 1)
    tasks = list(state["tasks"])
    if index < 0:
        tasks.append(task)
    else:
        tasks[index] = task
    return {
        **state,
        "enabled": state.get("enabled", True),
        "tasks": tasks,
        "nextTaskNumber": next_number,
    }


def projection_view(value: dict[str, Any]) -> dict[str, Any]:
    """Compute the durable client view of one Team state."""
    tasks = [
        task_view(project_task_view(value["tasks"], task))
        for task in value["tasks"]
        if task["status"] != "deleted"
    ]
    return {
        "enabled": value.get("enabled") is True,
        "tasks": tasks,
        **({} if value.get("failure") is None else {"failure": value["failure"]}),
    }


def _validate_wire(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - {"enabled", "tasks", "failure"}:
        raise TypeError("agentTeam wire value is invalid")
    if not isinstance(value.get("enabled"), bool) or not isinstance(
        value.get("tasks"), list
    ):
        raise TypeError("agentTeam wire value is invalid")
    return value


TEAM_PROJECTION: ProjectionDefinition[dict[str, Any]] = ProjectionDefinition(
    key="agentTeam",
    state_schema=validate_state,
    init=init_state,
    apply=apply_event,
    wire=WireView(
        view_schema=_validate_wire,
        view=projection_view,
        summary=False,
    ),
    state_version=7,
)


__all__ = [
    "TEAM_PROJECTION",
    "apply_event",
    "apply_task",
    "projection_view",
    "validate_state",
]
