"""Public Agent Teams identities, durable records and error taxonomy.

Ports ``packages/team/agent-team/src/types.ts`` and ``error.ts`` from the pinned
dsh source. A Team is the implicit task board owned by one root Session; tasks
are append-only snapshots on that Session's log.
"""

from __future__ import annotations

from typing import Any, Literal

from magitech_harness.llm.error import HarnessError

TeamTaskStatus = Literal["pending", "in_progress", "completed", "deleted"]
TeamAction = Literal[
    "claim",
    "release",
    "edit",
    "set_dependencies",
    "complete",
    "reopen",
    "reassign",
    "delete",
]

TASK_ID_PREFIX = "task-"


class TeamError(HarnessError):
    """Stable failure taxonomy for the Team domain."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


def required_text(value: Any, field: str, max_length: int) -> str:
    """Normalize one required human-authored string."""
    if not isinstance(value, str):
        raise TeamError(f"{field} must be a non-empty string", "TEAM_INVALID_ARGUMENT")
    text = value.strip()
    if not text:
        raise TeamError(f"{field} must be non-empty", "TEAM_INVALID_ARGUMENT")
    if len(text) > max_length:
        raise TeamError(
            f"{field} exceeds {max_length} characters", "TEAM_INVALID_ARGUMENT"
        )
    return text


def write_scope(value: Any) -> str:
    """Normalize one workspace-relative path prefix without treating it as a lock."""
    if not isinstance(value, str):
        raise TeamError(
            f"invalid workspace-relative write scope {value!r}",
            "TEAM_INVALID_WRITE_SCOPE",
        )
    normalized = value.replace("\\", "/")
    normalized = normalized.removeprefix("./")
    normalized = normalized.rstrip("/")
    segments = normalized.split("/")
    import re

    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[a-z]:", normalized, re.IGNORECASE)
        or any(segment in ("", ".", "..") for segment in segments)
    ):
        raise TeamError(
            f"invalid workspace-relative write scope {value!r}",
            "TEAM_INVALID_WRITE_SCOPE",
        )
    return normalized


def task_id(number: int) -> str:
    """Build a numeric task id."""
    return f"{TASK_ID_PREFIX}{number}"


def task_number(value: str) -> int | None:
    """Parse the numeric suffix of a task id, or ``None`` when non-numeric."""
    if not value.startswith(TASK_ID_PREFIX):
        return None
    suffix = value[len(TASK_ID_PREFIX) :]
    if not suffix.isdigit():
        return None
    number = int(suffix)
    return number if number <= 2**53 - 1 else None


def task_view(task: dict[str, Any]) -> dict[str, Any]:
    """Project one durable task snapshot to the model-facing view shape."""
    return {
        "id": task["id"],
        "revision": task["revision"],
        "subject": task["subject"],
        "description": task["description"],
        "status": task["status"],
        "blockedBy": list(task["blockedBy"]),
        "writeScopes": list(task["writeScopes"]),
        **({} if task.get("ownerId") is None else {"ownerId": task["ownerId"]}),
        **({} if task.get("ownerName") is None else {"ownerName": task["ownerName"]}),
        "ready": bool(task.get("ready", False)),
        "writeScopeWarnings": list(task.get("writeScopeWarnings", [])),
    }


__all__ = [
    "TASK_ID_PREFIX",
    "TeamAction",
    "TeamError",
    "TeamTaskStatus",
    "required_text",
    "task_id",
    "task_number",
    "task_view",
    "write_scope",
]
