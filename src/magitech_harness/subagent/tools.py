"""Tool surface for the subagent domain.

Ports dsh's tool rows at ``packages/subagent/tool-subagent-control/src`` and
``packages/subagent/tool-subagent/src/index.ts``:

- ``subagent`` / ``subagent_fork`` — delegate a task to a managed child.
- ``send_message`` — steer an active child or continue its conversation.
- ``list_agents`` — discover the caller's tree.
- ``interrupt_agent`` — ask a child to stop (ancestor authority only).
- ``wait_agent`` — block for pending input at the next step.
"""

from __future__ import annotations

import asyncio
from typing import Any

from animas import Context

from magitech_harness.subagent.service import ROOT_AGENT_NAME, SubagentService
from magitech_harness.subagent.types import SubagentSpec
from magitech_harness.tools.schema import define_tool


def _require_agent(exec_view: Any) -> Any:
    agent = exec_view.get("agent")
    if agent is None:
        from magitech_harness.llm.error import HarnessError

        raise HarnessError("subagent tools require a calling agent", "AGENT_REQUIRED")
    return agent


def apply(
    ctx: Context,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Mount the delegation and control tools from one dsh-equivalent row."""
    cfg = {**(config or {}), **kwargs}
    service: SubagentService = ctx.subagents
    provider = str(cfg.get("provider", "spawn"))
    tool_name = str(cfg.get("tool_name", "subagent"))
    control = bool(cfg.get("control", True))
    delegate = bool(cfg.get("delegate", True))

    async def _delegate(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        agent = _require_agent(exec_view)
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            from magitech_harness.llm.error import HarnessError

            raise HarnessError("prompt must be a non-empty string", "INVALID_ARGS")
        description = args.get("description")
        spec = SubagentSpec(
            role=str(description or "worker"),
            prompt=prompt,
            agent_name=args.get("name") if isinstance(args.get("name"), str) else None,
            model=str(args.get("model", "inherit")),
            workspace="branch" if args.get("workspace") == "branch" else "inherit",
            allowed_tools=(
                frozenset(args["allowed_tools"])
                if isinstance(args.get("allowed_tools"), list)
                else None
            ),
            wait=bool(args.get("wait", True)),
        )
        activation = await service.startActivation(
            spec,
            provider=provider,
            parent_id=getattr(agent, "id", None),
            signal=exec_view.get("signal"),
        )
        return {"kind": activation["kind"], "name": activation["name"]}

    async def _send_message(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        agent = _require_agent(exec_view)
        target = args["agent"]
        message = args["message"]
        target_id = service.resolve_name(agent, target, exec_view.get("signal"))
        result = await service.send_message(
            agent, target_id, [{"type": "text", "text": message}]
        )
        return result

    async def _list_agents(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        del args
        _require_agent(exec_view)
        rows = await service.list_descendants(ROOT_AGENT_NAME, exec_view.get("signal"))
        running = sum(1 for row in rows if row["status"] == "running")
        return {
            "running": running,
            "limit": service.max_active_subagents,
            "agents": [
                {
                    "name": ROOT_AGENT_NAME,
                    "status": "running",
                    "description": "root agent",
                },
                *[
                    {
                        "name": row["name"],
                        "status": row["status"],
                        "description": row["description"],
                    }
                    for row in rows
                ],
            ],
        }

    async def _interrupt_agent(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        agent = _require_agent(exec_view)
        target = args["agent"]
        target_id = service.resolve_name(agent, target, exec_view.get("signal"))
        return service.interrupt(target_id, {"kind": "ancestor", "agent": agent})

    async def _wait_agent(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        agent = _require_agent(exec_view)
        timeout_ms = args.get("timeout_ms", 30_000)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 10_000 <= timeout_ms <= 3_600_000
        ):
            from magitech_harness.llm.error import HarnessError

            raise HarnessError(
                "timeout_ms must be an integer from 10000 through 3600000",
                "WAIT_INVALID_TIMEOUT",
            )
        inbox = getattr(agent, "inbox", None)
        if inbox is None:  # pragma: no cover - live agents own an inbox
            return {"timedOut": True}
        signal = exec_view.get("signal")
        result: dict[str, Any] = {"timedOut": True}
        try:
            await asyncio.wait_for(
                asyncio.to_thread(_wait_for_input, inbox),
                timeout=timeout_ms / 1000,
            )
            result["timedOut"] = False
        except TimeoutError:
            pass
        del signal
        return result

    if delegate:
        ctx.tools.register(
            define_tool(
                name=tool_name,
                description=(
                    "Delegate a self-contained task to a subagent that works in its "
                    "own context and returns its result, not its intermediate steps. "
                    "The call starts the child and returns its name immediately."
                ),
                parameters={
                    "description": {
                        "type": "string",
                        "required": True,
                        "description": "Short 3-5 word description for display.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional short unique child name.",
                    },
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "description": "Complete self-contained task for the subagent.",
                    },
                    "model": {
                        "type": "string",
                        "description": "'inherit' or a model id.",
                    },
                    "workspace": {
                        "type": "string",
                        "enum": ["inherit", "branch"],
                        "description": "Workspace isolation mode.",
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional allow-list of tool names.",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Wait for the child turn.",
                    },
                },
                output={
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kind": {"type": "string", "required": True},
                            "name": {"type": "string", "required": True},
                        },
                    },
                    "render": lambda _a, v: [
                        {"type": "text", "text": f"subagent {v['name']} started"}
                    ],
                },
                execute=_delegate,
            )
        )

    if control:
        ctx.tools.register(
            define_tool(
                name="send_message",
                description=(
                    "Send a message to an agent. A working agent receives it at its "
                    "next step; an idle agent starts a new turn with it."
                ),
                parameters={
                    "agent": {"type": "string", "required": True},
                    "message": {"type": "string", "required": True},
                },
                output={
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "messageId": {"type": "string", "required": True}
                        },
                    },
                    "render": lambda a, _v: [
                        {
                            "type": "text",
                            "text": f"message delivered to agent {a['agent']}",
                        }
                    ],
                },
                execute=_send_message,
            )
        )

        ctx.tools.register(
            define_tool(
                name="list_agents",
                description=(
                    "List your entire subagent tree: the root and all reachable "
                    "continuable descendants, including yourself."
                ),
                parameters={},
                output={
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "running": {"type": "integer", "required": True},
                            "limit": {"type": "integer", "required": True},
                            "agents": {
                                "type": "array",
                                "required": True,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string", "required": True},
                                        "status": {"type": "string", "required": True},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    "render": lambda _a, v: [
                        {
                            "type": "text",
                            "text": "\n".join(
                                [
                                    (
                                        f"Subagent capacity: running={v['running']}, "
                                        f"limit={v['limit']}"
                                    ),
                                    *[
                                        f"{row['name']} [{row['status']}]"
                                        + (
                                            f" — {row['description']}"
                                            if row.get("description")
                                            else ""
                                        )
                                        for row in v["agents"]
                                    ],
                                ]
                            ),
                        }
                    ],
                },
                execute=_list_agents,
            )
        )

        ctx.tools.register(
            define_tool(
                name="interrupt_agent",
                description=(
                    "Ask a subagent to stop its current work. This call returns "
                    "without waiting for it to stop."
                ),
                parameters={
                    "agent": {
                        "type": "string",
                        "required": True,
                        "description": "Name of a child in your subagent tree.",
                    }
                },
                output={
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "accepted": {"type": "boolean", "required": True}
                        },
                    },
                    "render": lambda a, _v: [
                        {
                            "type": "text",
                            "text": f"interrupt requested for agent {a['agent']}",
                        }
                    ],
                },
                execute=_interrupt_agent,
            )
        )

        ctx.tools.register(
            define_tool(
                name="wait_agent",
                description=(
                    "Wait for input at your next step, including agent messages, "
                    "completion notifications, and user steering."
                ),
                parameters={
                    "timeout_ms": {
                        "type": "integer",
                        "description": "10000 through 3600000; default 30000.",
                    }
                },
                output={
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "timedOut": {"type": "boolean", "required": True}
                        },
                    },
                    "render": lambda _a, v: [
                        {
                            "type": "text",
                            "text": f'{{"timedOut": {str(v["timedOut"]).lower()}}}',
                        }
                    ],
                },
                execute=_wait_agent,
            )
        )


def _wait_for_input(inbox: Any) -> bool:
    """Block until the next-step queue is non-empty (called in a thread)."""
    import time

    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if inbox.has_pending:
            return True
        time.sleep(0.02)
    return False  # pragma: no cover - 1h safety net; the caller's timeout governs


def apply_control(
    ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any
) -> None:
    """Mount only the shared subagent control tools."""
    apply(ctx, {**(config or {}), **kwargs, "control": True, "delegate": False})


apply.inject = ["tools", "subagents"]  # type: ignore[attr-defined]
apply_control.inject = ["tools", "subagents"]  # type: ignore[attr-defined]

__all__ = ["apply", "apply_control"]
