"""Todo list tool: whole-list replacement persisted as ``todo/write`` session events.

A Tier 1 profile-bundle plugin. Registers the ``todos`` session-projection unit
(key ``todos``, state version 2) and the ``todo_write`` model-facing tool. Each
call appends a ``todo/write`` snapshot to the calling agent's session; replay is
last-write-wins. A non-agent caller has no owning list and is rejected.

The fold: ``todo/write`` replaces the whole list; ``turn/start`` clears to null;
every other event keeps the state reference. This matches the DSH reference
(@deepseek-ai/dsh-tool-todo) at packages/todo/tool-todo/src/index.ts.
"""

from __future__ import annotations

from typing import Any

from animas import Context

from magitech_harness.llm.error import HarnessError
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    WireView,
)
from magitech_harness.tools.schema import define_tool

STATUSES = frozenset({"pending", "in_progress", "completed"})


class TodoArgsError(HarnessError):
    """Raised when todo_write arguments violate constraints."""

    def __init__(self, violations: list[str]) -> None:
        super().__init__(
            f"invalid arguments: {'; '.join(violations)}",
            "INVALID_ARGS",
        )
        self.violations = list(violations)


def _validate_todos(
    raw: list[dict[str, Any]], allow_parallel: bool
) -> list[dict[str, str]]:
    """Validate and canonicalize the todo list.

    Each item needs non-empty, unique ``content`` and a valid ``status``.
    At most one ``in_progress`` unless the deployment allows parallel work.
    """
    if not isinstance(raw, list):
        raise TodoArgsError(["todos must be an array"])
    todos: list[dict[str, str]] = []
    seen: set[str] = set()
    active = 0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TodoArgsError([f"todos[{index}] must be an object"])
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise TodoArgsError([f"todos[{index}].content must be a non-empty string"])
        content = content.strip()
        if content in seen:
            raise TodoArgsError([f"todos[{index}].content duplicates {content!r}"])
        seen.add(content)
        status = item.get("status")
        if status not in STATUSES:
            raise TodoArgsError(
                [f"todos[{index}].status must be one of {sorted(STATUSES)}"]
            )
        if status == "in_progress":
            active += 1
        todos.append({"content": content, "status": status})
    if not allow_parallel and active > 1:
        raise TodoArgsError([f"at most one task may be in_progress (got {active})"])
    return todos


def _counts(todos: list[dict[str, str]]) -> dict[str, int]:
    return {
        "pending": sum(1 for t in todos if t["status"] == "pending"),
        "inProgress": sum(1 for t in todos if t["status"] == "in_progress"),
        "completed": sum(1 for t in todos if t["status"] == "completed"),
    }


# State schema: a list of {content, status} or null before the first write.
def _validate_todo_state(value: Any) -> list[dict[str, str]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("todos state must be an array or null")
    return _validate_todos(value, allow_parallel=False)


def _init_todos(
    header: dict[str, Any], inherited_event_count: int
) -> list[dict[str, str]] | None:
    del header, inherited_event_count
    return None


def _apply_todos(
    state: list[dict[str, str]] | None, event: SessionEvent
) -> list[dict[str, str]] | None:
    if event.type == "todo/write":
        data = event.payload
        if isinstance(data, dict) and isinstance(data.get("todos"), list):
            return _validate_todos(data["todos"], allow_parallel=False)
        return state
    if event.type == "turn/start":
        return None
    return state


def _todos_view(
    state: list[dict[str, str]] | None,
) -> list[dict[str, str]] | None:
    return state


TODOS_PROJECTION: ProjectionDefinition[list[dict[str, str]] | None] = (
    ProjectionDefinition(
        key="todos",
        state_schema=_validate_todo_state,
        init=_init_todos,
        apply=_apply_todos,
        wire=WireView(
            view_schema=_validate_todo_state,
            view=_todos_view,
            summary=False,
        ),
        state_version=2,
    )
)


def apply(
    ctx: Context,
    *,
    allow_parallel_in_progress: bool = True,
    **kwargs: Any,
) -> None:
    """Mount the todo_write tool and the todos projection unit."""
    ctx.sessionProjections.register(TODOS_PROJECTION)

    def _execute_todo_write(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        session = exec_view.get("session")
        if session is None:
            raise TodoArgsError(["todo_write requires a session"])
        todos = _validate_todos(args.get("todos", []), allow_parallel_in_progress)
        session.append("todo/write", {"todos": todos})
        return {
            "todos": todos,
            "counts": _counts(todos),
        }

    ctx.tools.register(
        define_tool(
            name="todo_write",
            description=(
                "Record and update a task list to plan multi-step work and show "
                "progress; skip it for trivial single-step tasks. Add one todo per "
                "concrete step before you start. "
                + (
                    "While work remains, keep the todos being worked on `in_progress`, "
                    "several only when work runs in parallel. "
                    if allow_parallel_in_progress
                    else "While work remains, keep exactly one todo `in_progress`. "
                )
                + "Mark each todo `completed` as soon as it is done."
            ),
            parameters={
                "todos": {
                    "type": "array",
                    "required": True,
                    "description": "The COMPLETE task list, replacing any previous list.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "content": {
                                "type": "string",
                                "required": True,
                                "description": "What the task is — a short imperative line.",
                            },
                            "status": {
                                "type": "string",
                                "required": True,
                                "enum": sorted(STATUSES),
                                "description": "pending (not started) | in_progress (now) | completed (done).",
                            },
                        },
                    },
                },
            },
            output={
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "todos": {
                            "type": "array",
                            "required": True,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "content": {"type": "string", "required": True},
                                    "status": {
                                        "type": "string",
                                        "required": True,
                                        "enum": sorted(STATUSES),
                                    },
                                },
                            },
                        },
                        "counts": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": True,
                            "properties": {
                                "pending": {"type": "integer", "required": True},
                                "inProgress": {"type": "integer", "required": True},
                                "completed": {"type": "integer", "required": True},
                            },
                        },
                    },
                },
                "render": (
                    lambda _args, value: [
                        {
                            "type": "text",
                            "text": (
                                f"Updated todo list: {value['counts']['pending']} pending, "
                                f"{value['counts']['inProgress']} in progress, "
                                f"{value['counts']['completed']} completed."
                            ),
                        }
                    ]
                ),
            },
            execute=_execute_todo_write,
        )
    )


apply.inject = ["tools", "sessionProjections"]  # type: ignore[attr-defined]

__all__ = ["STATUSES", "TodoArgsError", "apply"]
