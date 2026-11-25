"""Conditionally available task-board tools and guidance for Agent Teams.

Ports ``packages/team/tool-agent-team/src/index.ts``. Tools are admitted only
when the caller's Team board is enabled, matching dsh's progressive disclosure.
"""

from __future__ import annotations

import json
from typing import Any

from animas import Context

from magitech_harness.llm.error import HarnessError
from magitech_harness.team.service import TeamService
from magitech_harness.tools.schema import ToolArgsError, define_tool

TEAM_POLICY = """Agent Teams is available in this session.

The Team Lead and all teammates share the same working directory and filesystem. Edits are immediately visible to every member. Split write work into disjoint scopes, record expected write scopes on shared tasks, and use task dependencies when work must be ordered. Write-scope overlap is advisory, not a lock.

Use the names returned by subagent, subagent_fork, or list_agents as owner when assigning or filtering shared tasks. The root can reassign tasks to itself or any continuable descendant; other agents claim ready tasks themselves. Shared-task workflow is list, get, claim with the current revision, perform the work, then complete. Task readiness never starts an owner. The root agent must wait for required teammates before giving the final answer.

Task changes do not end wait_agent calls, add inbox messages, or start idle agents. Use send_message when another agent needs to act on a task."""

TASK_VIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "required": True},
        "revision": {"type": "integer", "required": True},
        "subject": {"type": "string", "required": True},
        "description": {"type": "string", "required": True},
        "status": {
            "type": "string",
            "required": True,
            "enum": ["pending", "in_progress", "completed", "deleted"],
        },
        "owner": {"type": "string"},
        "blockedBy": {"type": "array", "required": True, "items": {"type": "string"}},
        "writeScopes": {"type": "array", "required": True, "items": {"type": "string"}},
        "ready": {"type": "boolean", "required": True},
        "writeScopeWarnings": {
            "type": "array",
            "required": True,
            "items": {"type": "string"},
        },
    },
}

TASK_LIST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tasks": {"type": "array", "required": True, "items": TASK_VIEW_SCHEMA},
        "nextCursor": {"type": "integer"},
    },
}


def _render(_args: Any, value: Any) -> list[dict[str, str]]:
    return [{"type": "text", "text": json.dumps(value)}]


def _output(schema: dict[str, Any]) -> dict[str, Any]:
    return {"schema": schema, "render": _render}


def _require_agent(exec_view: Any) -> Any:
    agent = exec_view.get("agent") if isinstance(exec_view, dict) else None
    if agent is None:
        raise HarnessError("team tools require a calling agent", "AGENT_REQUIRED")
    return agent


def _named(task: dict[str, Any]) -> dict[str, Any]:
    """Replace internal owner fields with dsh's single model-facing ``owner``."""
    owner_name = task.get("ownerName") or task.get("ownerId")
    return {
        **{
            key: value
            for key, value in task.items()
            if key not in ("ownerId", "ownerName")
        },
        **({} if owner_name is None else {"owner": owner_name}),
    }


def apply(ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Mount the four Team task tools and the Team policy section."""
    del config, kwargs
    service: TeamService = ctx.agentTeams

    def enabled(act_ctx: dict[str, Any]) -> bool:
        agent = act_ctx.get("agent")
        if agent is None:
            return False
        try:
            return service.is_enabled(agent)
        except HarnessError as error:
            if getattr(error, "code", None) != "UNAUTHORIZED":
                raise
            return False

    when = enabled
    system_prompt = ctx.get("systemPrompt", None, strict=False)
    if system_prompt is not None:
        system_prompt.section(
            {
                "name": "team:policy",
                "retirement": "The shared task board is no longer available. Its instructions no longer apply.",
                "order": system_prompt.get_section_order("TEAM_POLICY"),
                "text": lambda context: TEAM_POLICY if enabled(context) else "",
            }
        )

    ctx.tools.register(
        define_tool(
            name="team_task_create",
            description="Create one unowned pending task on the shared Team task board.",
            parameters={
                "subject": {
                    "type": "string",
                    "required": True,
                    "description": "Concise task title.",
                },
                "description": {
                    "type": "string",
                    "required": True,
                    "description": "Complete task details and acceptance criteria.",
                },
                "blocked_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task ids that must complete first.",
                },
                "write_scopes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Advisory workspace-relative file or directory prefixes.",
                },
            },
            output=_output(TASK_VIEW_SCHEMA),
            execute=lambda args, exec_view: _named(
                service.create_task(
                    _require_agent(exec_view),
                    {
                        "subject": args.get("subject"),
                        "description": args.get("description"),
                        "blockedBy": args.get("blocked_by"),
                        "writeScopes": args.get("write_scopes"),
                    },
                )
            ),
        ),
        {"when": when},
    )

    ctx.tools.register(
        define_tool(
            name="team_task_list",
            description="List shared tasks, including readiness, owner, revision, blockers, and write-scope warnings.",
            parameters={
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],
                    "description": "Optional exact status filter.",
                },
                "owner": {
                    "type": "string",
                    "description": "Optional owner agent name filter; cannot be combined with unowned.",
                },
                "unowned": {
                    "type": "boolean",
                    "description": "Filter tasks by whether they have no owner.",
                },
                "ready": {
                    "type": "boolean",
                    "description": "Optional readiness filter.",
                },
                "cursor": {
                    "type": "integer",
                    "description": "Zero-based result offset. Defaults to 0.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of rows, 1 through 100. Defaults to 50.",
                },
            },
            output=_output(TASK_LIST_SCHEMA),
            isConcurrencySafe=lambda _args: True,
            execute=lambda args, exec_view: _list(
                service, ctx, _require_agent(exec_view), args
            ),
        ),
        {"when": when},
    )

    ctx.tools.register(
        define_tool(
            name="team_task_get",
            description="Read the complete latest value of one shared task before changing or executing it.",
            parameters={
                "task_id": {
                    "type": "string",
                    "required": True,
                    "description": "Shared task id.",
                }
            },
            output=_output(TASK_VIEW_SCHEMA),
            isConcurrencySafe=lambda _args: True,
            execute=lambda args, exec_view: _named(
                service.get_task(_require_agent(exec_view), args.get("task_id"))
            ),
        ),
        {"when": when},
    )

    ctx.tools.register(
        define_tool(
            name="team_task_update",
            description="Compare-and-set a shared task action using the latest revision from team_task_get or team_task_list.",
            parameters={
                "task_id": {
                    "type": "string",
                    "required": True,
                    "description": "Shared task id.",
                },
                "expected_revision": {
                    "type": "integer",
                    "required": True,
                    "description": "Current task revision used as the CAS precondition.",
                },
                "action": {
                    "type": "string",
                    "required": True,
                    "enum": [
                        "claim",
                        "release",
                        "edit",
                        "set_dependencies",
                        "complete",
                        "reopen",
                        "reassign",
                        "delete",
                    ],
                    "description": "Task transition to apply.",
                },
                "subject": {
                    "type": "string",
                    "description": "Replacement title for edit.",
                },
                "description": {
                    "type": "string",
                    "description": "Replacement details for edit.",
                },
                "blocked_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Complete blocker list.",
                },
                "write_scopes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replacement write scopes.",
                },
                "owner": {
                    "type": "string",
                    "description": "Agent name for root-only reassign.",
                },
            },
            output=_output(TASK_VIEW_SCHEMA),
            execute=lambda args, exec_view: _named(
                _update(service, _require_agent(exec_view), args)
            ),
        ),
        {"when": when},
    )


def _list(
    service: TeamService, ctx: Context, agent: Any, args: dict[str, Any]
) -> dict[str, Any]:
    cursor = args.get("cursor", 0)
    limit = args.get("limit", 50)
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ToolArgsError(["cursor must be a non-negative safe integer"])
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ToolArgsError(["limit must be an integer from 1 through 100"])
    if args.get("owner") is not None and args.get("unowned") is not None:
        raise ToolArgsError(["owner and unowned filters cannot be combined"])
    owner = args.get("owner")
    if isinstance(owner, str):
        subagents = ctx.get("subagents", None, strict=False)
        if subagents is not None:
            owner = subagents.resolve_name(agent, owner)
    tasks = service.list_tasks(agent)
    filtered = [
        task
        for task in tasks
        if (args.get("status") is None or task["status"] == args["status"])
        and (owner is None or task.get("ownerId") == owner)
        and (
            args.get("unowned") is None
            or (task.get("ownerId") is None) == args["unowned"]
        )
        and (args.get("ready") is None or task["ready"] == args["ready"])
    ]
    page = filtered[cursor : cursor + limit]
    return {
        "tasks": [_named(task) for task in page],
        **({"nextCursor": cursor + limit} if cursor + limit < len(filtered) else {}),
    }


def _update(service: TeamService, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
    return service.update_task(
        agent,
        {
            "taskId": args.get("task_id"),
            "expectedRevision": args.get("expected_revision"),
            "action": args.get("action"),
            "subject": args.get("subject"),
            "description": args.get("description"),
            "blockedBy": args.get("blocked_by"),
            "writeScopes": args.get("write_scopes"),
            "owner": args.get("owner"),
        },
    )


apply.inject = ["agentTeams", "tools", "systemPrompt"]  # type: ignore[attr-defined]

__all__ = ["TEAM_POLICY", "_list", "_named", "_update", "apply"]
