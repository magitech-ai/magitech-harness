"""Model-facing goal tools: ``get_goal``, ``create_goal``, ``update_goal``.

Ports DSH ``@deepseek-ai/dsh-tool-goal`` at
``packages/goal/tool-goal/src/{index,authority,wrapup}.ts``.

Authority is enforced in the operation that makes the decision:

* ``create_goal``, ``edit``, ``pause``, and ``resume`` require a direct human
  message in the open turn of a top-level agent.
* ``complete`` and ``blocked`` additionally accept the exact admitted goal
  round (a ``user/message`` whose source is this goal's next round).
* ``blocked`` is refused before the configured minimum round count.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from animas import Context

from magitech_harness.goal.service import GoalService
from magitech_harness.goal.types import GoalBlockReason, GoalRef, GoalView
from magitech_harness.llm.error import HarnessError
from magitech_harness.llm.message import create_user_message
from magitech_harness.tools.schema import define_tool

DEFAULT_BLOCKED_AFTER_ROUNDS = 3
UPDATE_ACTIONS = ("edit", "pause", "resume", "complete", "blocked")

_CREATE_DESCRIPTION = (
    "Create a persisted goal that keeps this session working across automatic "
    "continuation rounds. Use it when the direct human request is a "
    'long-running objective, even if the user did not say "goal"; not for '
    "single-turn work."
)

_GET_DESCRIPTION = (
    "Read the current session goal, including the id and revision that "
    "update_goal requires."
)


@dataclass(frozen=True)
class _Execution:
    """The authenticated agent plus the immutable open-turn cut."""

    agent: Any
    events: list[Any]
    open_turn_start_seq: int


def _guidance(blocked_after: int) -> str:
    return (
        "create_goal may infer goal intent from a direct human request in any "
        "language. After session resume or fork, an active goal is disarmed: "
        "when a human asks to continue or resume in any wording or language, "
        "use update_goal action resume to rearm it. Mark complete only when the "
        "objective is actually achieved. Mark blocked only after the same "
        f"blocking condition persists for at least {blocked_after} consecutive "
        "goal rounds, and report that concrete condition in blocked_reason; "
        "difficulty, uncertainty, or useful remaining work is not blocked."
    )


def _reject(message: str, code: str = "GOAL_TOOL_AUTHORITY_REQUIRED") -> NoReturn:
    raise HarnessError(message, code)


def _goal_value(goal: GoalView | None) -> dict[str, Any]:
    if goal is None:
        return {"goal": None}
    value: dict[str, Any] = {
        "goal": {
            "id": goal.id,
            "revision": goal.revision,
            "objective": goal.objective,
            "phase": goal.phase,
            "roundsStarted": goal.rounds_started,
            "maxGoalRounds": goal.max_goal_rounds,
        },
        "activation": goal.activation,
    }
    if goal.blocked_reason is not None:
        value["goal"]["blockedReason"] = {
            "code": goal.blocked_reason.code,
            "message": goal.blocked_reason.message,
        }
    return value


def _goal_ref(goal_id: Any, revision: Any) -> GoalRef:
    if (
        not isinstance(goal_id, str)
        or not goal_id
        or goal_id != goal_id.strip()
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise HarnessError(
            "goal_id must be non-empty and revision must be a positive safe integer",
            "GOAL_TOOL_INVALID_UPDATE",
        )
    return GoalRef(id=goal_id, revision=revision)


def _goal_tool_execution(ctx: Context, exec_view: Mapping[str, Any]) -> _Execution:
    agent = exec_view.get("agent")
    if agent is None:
        return _reject("goal tools require a calling agent", "GOAL_TOOL_AGENT_REQUIRED")
    agents = ctx.get("agents", None, strict=False)
    if agents is None:
        return _reject(
            "goal tools require the agent registry", "GOAL_TOOL_DRIVER_REQUIRED"
        )
    if agents.get(agent.id) is not agent:
        return _reject(
            "goal tools require the exact live calling agent inside its active driver",
            "GOAL_TOOL_DRIVER_REQUIRED",
        )
    boundary = agents.turn_boundary_of(agent.session)
    open_turn_start = boundary["openTurnStartSeq"]
    if open_turn_start is None:
        return _reject(
            "goal tools require an open model turn", "GOAL_TOOL_DRIVER_REQUIRED"
        )
    return _Execution(
        agent=agent,
        events=agent.session.snapshot_events(),
        open_turn_start_seq=open_turn_start,
    )


def _some_open_turn_event(execution: _Execution, predicate: Any) -> bool:
    for seq in range(execution.open_turn_start_seq + 1, len(execution.events)):
        event = execution.events[seq]
        if event is not None and predicate(event):
            return True
    return False


def _has_direct_human_input(ctx: Context, execution: _Execution) -> bool:
    agents = ctx.get("agents", None, strict=False)
    if agents is None or agents.get(execution.agent.id) is not execution.agent:
        return False
    if execution.agent.session.header.get("origin") == "subagent":
        return False
    return _some_open_turn_event(
        execution,
        lambda event: (
            event.type == "user/message"
            and isinstance(event.payload.get("source"), Mapping)
            and event.payload["source"].get("kind") == "user"
        ),
    )


def _is_matching_goal_round(execution: _Execution, goal: GoalView) -> bool:
    def matches(event: Any) -> bool:
        source = event.payload.get("source")
        return (
            event.type == "user/message"
            and isinstance(source, Mapping)
            and source.get("kind") == "goal"
            and source.get("goalId") == goal.id
            and source.get("revision") == goal.revision
            and source.get("round") == goal.rounds_started
        )

    return _some_open_turn_event(execution, matches)


def _require_direct_human(ctx: Context, execution: _Execution) -> None:
    if _has_direct_human_input(ctx, execution):
        return
    _reject("this goal operation requires a direct human turn on a top-level agent")


def _completion_authority(
    ctx: Context, execution: _Execution, goals: GoalService
) -> str:
    if _has_direct_human_input(ctx, execution):
        return "direct-human"
    goal = goals.get(execution.agent)
    if goal is not None and _is_matching_goal_round(execution, goal):
        return "goal-round"
    return _reject(
        "complete and blocked require a direct human turn or the current goal round"
    )


def _wrapup_text(objective: str, blocked_reason: str | None = None) -> str:
    if blocked_reason is None:
        return (
            f"The goal is complete: {objective}\n"
            "Report the outcome to the user. Do not start another goal round."
        )
    return (
        f"The goal is blocked: {objective}\n"
        f"Blocking condition: {blocked_reason}\n"
        "Report the blocking condition to the user and stop automatic progress."
    )


_GOAL_OUTPUT_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"goal": {"type": "null", "required": True}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "goal": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": True,
                    "properties": {
                        "id": {"type": "string", "required": True},
                        "revision": {"type": "integer", "required": True},
                        "objective": {"type": "string", "required": True},
                        "phase": {
                            "type": "string",
                            "required": True,
                            "enum": ["active", "paused", "blocked", "complete"],
                        },
                        "roundsStarted": {"type": "integer", "required": True},
                        "maxGoalRounds": {"type": "integer", "required": True},
                        "blockedReason": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "code": {"type": "string", "required": True},
                                "message": {"type": "string", "required": True},
                            },
                        },
                    },
                },
                "activation": {
                    "type": "string",
                    "required": True,
                    "enum": ["armed", "disarmed"],
                },
            },
        },
    ]
}


def _render_goal(_args: Any, value: dict[str, Any]) -> list[dict[str, Any]]:
    import json

    return [{"type": "text", "text": json.dumps(value)}]


def apply(
    ctx: Context,
    *,
    blocked_after_consecutive_rounds: int = DEFAULT_BLOCKED_AFTER_ROUNDS,
    **kwargs: Any,
) -> None:
    """Mount the three goal tools and their shared policy section."""
    if (
        isinstance(blocked_after_consecutive_rounds, bool)
        or not isinstance(blocked_after_consecutive_rounds, int)
        or blocked_after_consecutive_rounds < 1
    ):
        raise TypeError("blockedAfterConsecutiveRounds must be a positive safe integer")
    goals: GoalService = ctx.goals
    system_prompt = ctx.get("systemPrompt", None, strict=False)
    if system_prompt is not None:
        system_prompt.section(
            {
                "name": "tool:goal",
                "order": system_prompt.get_section_order("TOOL_GOAL"),
                "text": _guidance(blocked_after_consecutive_rounds),
                "retirement": (
                    "The goal-management tools are no longer available. Their "
                    "guidance no longer applies."
                ),
            }
        )

    def _get_goal(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        execution = _goal_tool_execution(ctx, exec_view)
        return _goal_value(goals.get(execution.agent))

    def _create_goal(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        execution = _goal_tool_execution(ctx, exec_view)
        _require_direct_human(ctx, execution)
        max_rounds = args.get("max_goal_rounds")
        goal = goals.create(
            execution.agent,
            args["objective"],
            max_rounds if isinstance(max_rounds, int) and max_rounds > 0 else None,
        )
        return _goal_value(goal)

    def _update_goal(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        execution = _goal_tool_execution(ctx, exec_view)
        ref = _goal_ref(args.get("goal_id"), args.get("revision"))
        action = args.get("action")
        if action == "edit":
            _require_direct_human(ctx, execution)
            objective = args.get("objective")
            max_rounds = args.get("max_goal_rounds")
            goal = goals.edit(
                execution.agent,
                ref,
                objective=objective
                if isinstance(objective, str) and objective
                else None,
                max_goal_rounds=(
                    max_rounds
                    if isinstance(max_rounds, int)
                    and not isinstance(max_rounds, bool)
                    and max_rounds > 0
                    else None
                ),
            )
            return _goal_value(goal)
        if action in ("pause", "resume"):
            _require_direct_human(ctx, execution)
            current = goals.get(execution.agent)
            if (
                action == "resume"
                and current is not None
                and current.id == ref.id
                and current.revision == ref.revision
                and current.phase == "paused"
            ):
                raise HarnessError(
                    "the model cannot resume a paused goal; the user must resume it",
                    "GOAL_TOOL_RESUME_PAUSED",
                )
            goal = (
                goals.pause(execution.agent, ref)
                if action == "pause"
                else goals.resume(execution.agent, ref)
            )
            return _goal_value(goal)
        authority = _completion_authority(ctx, execution, goals)
        blocked_reason = args.get("blocked_reason")
        if action == "blocked" and (
            not isinstance(blocked_reason, str) or not blocked_reason.strip()
        ):
            raise HarnessError(
                "blocked_reason is required with action blocked",
                "GOAL_TOOL_INVALID_UPDATE",
            )
        if action == "blocked" and authority == "goal-round":
            current = goals.get(execution.agent)
            if (
                current is not None
                and current.rounds_started < blocked_after_consecutive_rounds
            ):
                raise HarnessError(
                    "blocked requires at least "
                    f"{blocked_after_consecutive_rounds} consecutive goal rounds; "
                    f"current round is {current.rounds_started}",
                    "GOAL_TOOL_BLOCK_THRESHOLD",
                )
        if action == "complete":
            goal = goals.complete(execution.agent, ref)
        else:
            goal = goals.block(
                execution.agent,
                ref,
                GoalBlockReason(
                    code="model-reported",
                    message=str(blocked_reason),
                ),
            )
        if authority == "goal-round":
            # The pipeline always supplies deferContext for a tool run; the
            # wrap-up context lands in the next admitted request.
            exec_view["deferContext"](
                create_user_message(
                    content=[
                        {
                            "type": "text",
                            "text": _wrapup_text(
                                goal.objective,
                                None if action == "complete" else blocked_reason,
                            ),
                        }
                    ],
                    source={
                        "kind": "tool-goal",
                        "form": "notice",
                        "summary": f"{action}: {goal.objective}"[:200],
                    },
                )
            )
        return _goal_value(goal)

    ctx.tools.register(
        define_tool(
            name="get_goal",
            description=_GET_DESCRIPTION,
            parameters={},
            output={"schema": _GOAL_OUTPUT_SCHEMA, "render": _render_goal},
            execute=_get_goal,
        )
    )
    ctx.tools.register(
        define_tool(
            name="create_goal",
            description=_CREATE_DESCRIPTION,
            parameters={
                "objective": {
                    "type": "string",
                    "required": True,
                    "description": (
                        "The concrete completion objective inferred from the "
                        "direct human request."
                    ),
                },
                "max_goal_rounds": {
                    "type": "number",
                    "description": (
                        "Optional positive safe-integer limit on automatic "
                        "continuation rounds."
                    ),
                },
            },
            output={"schema": _GOAL_OUTPUT_SCHEMA, "render": _render_goal},
            execute=_create_goal,
        )
    )
    ctx.tools.register(
        define_tool(
            name="update_goal",
            description="Update the current goal.",
            parameters={
                "goal_id": {
                    "type": "string",
                    "required": True,
                    "description": "Exact id returned by get_goal.",
                },
                "revision": {
                    "type": "number",
                    "required": True,
                    "description": "Exact positive revision returned by get_goal.",
                },
                "action": {
                    "type": "string",
                    "required": True,
                    "enum": list(UPDATE_ACTIONS),
                    "description": (
                        "edit, pause, and resume require a direct top-level human "
                        "request. complete and blocked are also allowed during an "
                        "automatic continuation of this goal; blocked is rejected "
                        "before the configured minimum round count."
                    ),
                },
                "objective": {
                    "type": "string",
                    "description": "Replacement objective; valid only with action edit.",
                },
                "max_goal_rounds": {
                    "type": "number",
                    "description": "Replacement cap; valid only with action edit.",
                },
                "blocked_reason": {
                    "type": "string",
                    "description": (
                        "Required only with action blocked: the concrete condition "
                        "that persisted across rounds and blocks progress."
                    ),
                },
            },
            output={"schema": _GOAL_OUTPUT_SCHEMA, "render": _render_goal},
            execute=_update_goal,
        )
    )


apply.inject = ["tools", "goals", "agents"]  # type: ignore[attr-defined]

__all__ = [
    "DEFAULT_BLOCKED_AFTER_ROUNDS",
    "UPDATE_ACTIONS",
    "apply",
]
