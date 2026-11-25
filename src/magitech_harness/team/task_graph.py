"""Complete dependency validation for current Team task snapshots.

Ports ``packages/team/agent-team/src/task-graph.ts`` and ``task-view.ts``.
"""

from __future__ import annotations

from typing import Any

from magitech_harness.team.types import TeamError


def assert_task_graph(current: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    """Validate the active task graph after replacing one candidate snapshot."""
    tasks = {task["id"]: task for task in current}
    tasks[candidate["id"]] = candidate

    for task in tasks.values():
        if task["status"] == "deleted":
            continue
        seen: set[str] = set()
        for blocker_id in task["blockedBy"]:
            if blocker_id == task["id"]:
                raise TeamError(
                    f'team task "{task["id"]}" cannot block itself',
                    "TEAM_TASK_DEPENDENCY_CYCLE",
                )
            if blocker_id in seen:
                raise TeamError(
                    f'team task "{task["id"]}" repeats blocker "{blocker_id}"',
                    "TEAM_INVALID_ARGUMENT",
                )
            blocker = tasks.get(blocker_id)
            if blocker is None or blocker["status"] == "deleted":
                raise TeamError(
                    f'blocker task "{blocker_id}" for "{task["id"]}" is missing or deleted',
                    "TEAM_TASK_NOT_FOUND",
                )
            seen.add(blocker_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise TeamError(
                f'task dependency cycle includes "{identifier}"',
                "TEAM_TASK_DEPENDENCY_CYCLE",
            )
        if identifier in visited:
            return
        task = tasks.get(identifier)
        if task is None or task["status"] == "deleted":
            return
        visiting.add(identifier)
        for blocker_id in task["blockedBy"]:
            visit(blocker_id)
        visiting.discard(identifier)
        visited.add(identifier)

    for task in tasks.values():
        visit(task["id"])


def _scopes_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")


def task_ready(tasks: list[dict[str, Any]], task: dict[str, Any]) -> bool:
    """Whether every current blocker of one task completed."""
    statuses = {candidate["id"]: candidate["status"] for candidate in tasks}
    return all(statuses.get(blocker) == "completed" for blocker in task["blockedBy"])


def project_task_view(
    tasks: list[dict[str, Any]], task: dict[str, Any]
) -> dict[str, Any]:
    """Derive one task view with readiness and advisory write overlaps."""
    warnings: set[str] = set()
    if task["status"] in ("pending", "in_progress"):
        for other in tasks:
            if other["id"] == task["id"] or other["status"] != "in_progress":
                continue
            if any(
                _scopes_overlap(left, right)
                for left in task["writeScopes"]
                for right in other["writeScopes"]
            ):
                warnings.add(f"write scopes overlap with {other['id']}")
    return {
        **task,
        "ready": task["status"] == "pending" and task_ready(tasks, task),
        "writeScopeWarnings": sorted(warnings),
    }


__all__ = ["assert_task_graph", "project_task_view", "task_ready"]
