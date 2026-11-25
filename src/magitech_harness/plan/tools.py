"""Model-facing plan-mode tools: ``plan`` and ``exit_plan_mode``.

Ports ``packages/plan/plan-mode/src/index.ts``'s tool surface. The exit tool is
admitted only after the session has entered plan mode (dsh's admission outlives
the active state, so leaving plan mode only changes the guidance section).
"""

from __future__ import annotations

import re
from typing import Any

from animas import Context

from magitech_harness.context.user_questions import UserQuestionError
from magitech_harness.llm.message import create_user_message
from magitech_harness.plan.service import (
    APPROVE_LABEL,
    EXIT_DESCRIPTION,
    EXIT_PLAN_MODE,
    KEEP_PLANNING_LABEL,
    REVIEW_ID,
    PlanModeController,
)
from magitech_harness.plan.types import PlanError
from magitech_harness.tools.schema import ToolArgsError, define_tool

PLAN_DESCRIPTION = "Enter or leave plan mode for this session."
PLAN_SECTION = (
    "Plan mode is active: explore and design first, then present a complete "
    "markdown plan for the user to approve before taking action."
)


def _first_heading(plan: str) -> str | None:
    for line in plan.split("\n"):
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            return match.group(1)
    return None


def apply(ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Mount the plan tools and their prompt guidance."""
    del config, kwargs
    service: PlanModeController = ctx.planMode

    def exit_when(act_ctx: dict[str, Any]) -> bool:
        agent = act_ctx.get("agent")
        if agent is None:
            return False
        return service.selected_active(agent) or service.entered(agent)

    ctx.tools.register(
        define_tool(
            name="plan",
            description=PLAN_DESCRIPTION,
            parameters={
                "action": {
                    "type": "string",
                    "required": True,
                    "enum": ["on", "off"],
                    "description": "Enter (on) or leave (off) plan mode.",
                },
                "message": {
                    "type": "string",
                    "description": "Optional user-facing note explaining the change.",
                },
            },
            output={
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"status": {"type": "string", "required": True}},
                },
                "render": lambda _args, value: [
                    {"type": "text", "text": f"Plan mode {value['status']}."}
                ],
            },
            execute=lambda args, exec_view: _plan(service, args, exec_view),
        )
    )

    ctx.tools.register(
        define_tool(
            name=EXIT_PLAN_MODE,
            description=EXIT_DESCRIPTION,
            parameters={
                "plan": {
                    "type": "string",
                    "required": True,
                    "description": "The complete plan, as markdown, starting with a # heading that names it.",
                }
            },
            output={
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"approved": {"type": "boolean", "required": True}},
                },
                "render": lambda _args, value: [
                    {
                        "type": "text",
                        "text": (
                            "Plan approved — plan mode exited; carry out the plan starting with your next step."
                            if value["approved"]
                            else "The user dismissed the plan review to reply in their own words; plan mode remains active."
                        ),
                    }
                ],
            },
            presentCall=lambda args: {
                "card": "generic",
                "title": _first_heading(args.get("plan", "")) or "Plan",
                "kind": "other",
                "content": [{"type": "text", "text": args.get("plan", "")}],
            },
            execute=lambda args, exec_view: _exit(ctx, service, args, exec_view),
        ),
        {"when": exit_when},
    )

    system_prompt = ctx.get("systemPrompt", None, strict=False)
    if system_prompt is not None:
        system_prompt.section(
            {
                "name": "tool:plan",
                "order": system_prompt.get_section_order("TOOL_PLAN"),
                "text": lambda context: (
                    PLAN_SECTION
                    if (
                        context.get("agent") is not None
                        and service.selected_active(context["agent"])
                    )
                    else ""
                ),
            }
        )


def _plan(
    service: PlanModeController, args: dict[str, Any], exec_view: Any
) -> dict[str, str]:
    agent = exec_view.get("agent") if isinstance(exec_view, dict) else None
    if agent is None:
        raise PlanError("plan requires a calling agent", "AGENT_REQUIRED")
    action = args.get("action")
    if action not in ("on", "off"):
        raise ToolArgsError(["action must be 'on' or 'off'"])
    status = service.set(agent, action == "on")
    message = args.get("message")
    if isinstance(message, str) and message.strip():
        steer = getattr(agent, "steer", None)
        if callable(steer):
            steer(
                create_user_message(
                    content=[{"type": "text", "text": message.strip()}],
                    source={"kind": "user"},
                )
            )
    return {"status": status}


async def _exit(
    ctx: Context,
    service: PlanModeController,
    args: dict[str, Any],
    exec_view: Any,
) -> dict[str, bool]:
    agent = exec_view.get("agent") if isinstance(exec_view, dict) else None
    if agent is None:
        raise PlanError(f"{EXIT_PLAN_MODE} requires a calling agent", "AGENT_REQUIRED")
    if not service.logged_active(agent):
        raise PlanError(
            f"{EXIT_PLAN_MODE} is only available in plan mode", "PLAN_NOT_ACTIVE"
        )
    plan = args.get("plan")
    if not isinstance(plan, str) or not re.match(r"^#\s+\S", plan.strip()):
        raise ToolArgsError(
            [
                f"{EXIT_PLAN_MODE} requires a non-empty markdown plan starting with a # heading"
            ]
        )
    interaction = ctx.get("userQuestions", None, strict=False)
    if interaction is None:
        raise PlanError(
            "no user-questions channel is available to review the plan; ask the user to switch the session mode instead",
            "PLAN_REVIEW_UNAVAILABLE",
        )
    try:
        answers = await interaction.ask(
            [
                {
                    "id": REVIEW_ID,
                    "header": "Plan review",
                    "question": "Approve this plan and leave plan mode?",
                    "detail": plan,
                    "options": [
                        {
                            "label": APPROVE_LABEL,
                            "description": "Leave plan mode; the plan is carried out from the next step.",
                        },
                        {
                            "label": KEEP_PLANNING_LABEL,
                            "description": "Stay in plan mode; feedback goes back to the model.",
                        },
                    ],
                    "intent": {
                        "kind": "plan-review",
                        "approve": APPROVE_LABEL,
                        "callId": exec_view.get("callId"),
                    },
                }
            ],
            agent=agent,
            signal=exec_view.get("signal"),
        )
    except UserQuestionError as error:
        if getattr(error, "code", None) in ("ASK_CANCELLED", "ASK_ABORTED"):
            conclude = (
                exec_view.get("concludeTurn") if isinstance(exec_view, dict) else None
            )
            if callable(conclude):
                conclude()
            return {"approved": False}
        raise
    if service._disposed:
        raise PlanError(
            "the plan-mode service was reloaded while the plan was under review; present the plan again",
            "PLAN_REVIEW_INTERRUPTED",
        )
    item = next((answer for answer in answers if answer.id == REVIEW_ID), None)
    selected = list(item.selected) if item is not None else []
    approved = (
        len(selected) == 1
        and selected[0] == APPROVE_LABEL
        and item is not None
        and item.custom is None
    )
    if not approved:
        feedback = item.custom if item is not None else None
        raise PlanError(
            (
                "The user chose to keep planning; revise the plan and present it again."
                if not feedback
                else f"The user chose to keep planning; their feedback: {feedback}"
            ),
            "PLAN_REVISION_REQUESTED",
        )
    service.set(agent, False, narrate=False)
    return {"approved": True}


apply.inject = ["planMode", "tools", "systemPrompt"]  # type: ignore[attr-defined]

__all__ = ["PLAN_DESCRIPTION", "PLAN_SECTION", "_first_heading", "apply"]
