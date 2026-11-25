"""Human-facing ``/goal`` command over the persisted same-session goal domain.

Ports DSH ``@deepseek-ai/dsh-command-goal`` at
``packages/goal/command-goal/src/index.ts``: an omitted argument shows the
current goal; any other text creates one; ``clear``/``pause``/``resume``/
``edit <objective>`` control it. Compare-and-set revisions stay internal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from animas import Context

from magitech_harness.goal.service import GoalService
from magitech_harness.goal.types import GoalError, GoalRef, GoalView

USAGE = "Usage: /goal [<objective>|clear|edit <objective>|pause|resume]"


@dataclass(frozen=True)
class _Command:
    kind: str
    objective: str = ""


def parse_goal_command(raw_input: str) -> _Command:
    """Parse only the grammar ``/goal`` owns; other input is an objective."""
    text = raw_input.strip()
    if not text:
        return _Command("show")
    control = text.lower()
    if control == "clear":
        return _Command("clear")
    if control == "pause":
        return _Command("pause")
    if control == "resume":
        return _Command("resume")
    if control == "edit":
        return _Command("invalid-edit")
    if len(text) >= 4 and text[:4].lower() == "edit" and text[4:5].isspace():
        return _Command("edit", text[4:].strip())
    return _Command("create", text)


def _phase_label(phase: str) -> str:
    return phase


def _command_hint(goal: GoalView) -> str:
    if goal.phase == "active":
        return (
            "/goal edit <objective>, /goal pause, /goal clear"
            if goal.activation == "armed"
            else "/goal edit <objective>, /goal resume, /goal clear"
        )
    if goal.phase in ("paused", "blocked"):
        return "/goal edit <objective>, /goal resume, /goal clear"
    return "/goal <objective>, /goal clear"


def render_goal(title: str, goal: GoalView) -> str:
    """Render one goal without exposing compare-and-set internals."""
    lines = [title, f"Status: {_phase_label(goal.phase)}"]
    if goal.phase == "blocked":
        reason = goal.blocked_reason
        if reason is None:
            raise TypeError("blocked goal is missing its reason")
        lines.append(f"Blocker: {reason.code}: {reason.message}")
    lines.extend(
        [
            f"Objective: {goal.objective}",
            f"Rounds: {goal.rounds_started}/{goal.max_goal_rounds}",
            f"Activation: {goal.activation}",
            "",
            f"Commands: {_command_hint(goal)}",
        ]
    )
    return "\n".join(lines)


def _ref(goal: GoalView) -> GoalRef:
    return GoalRef(id=goal.id, revision=goal.revision)


def _missing(action: str) -> str:
    return f"No goal is currently set; /goal {action} requires one. {USAGE}"


def _live_agent(ctx: Context, session: Any) -> Any | None:
    agents = ctx.get("agents", None, strict=False)
    if agents is None:
        return None
    return agents.get(session.session_id)


def run_goal_command(ctx: Context, session: Any, raw_input: str) -> str:
    """Execute one ``/goal`` invocation and return its user-facing text."""
    goals: GoalService = ctx.goals
    agent = _live_agent(ctx, session)
    if agent is None:
        return f"/goal requires a live agent for this session. {USAGE}"
    command = parse_goal_command(raw_input)
    try:
        if command.kind == "show":
            goal = goals.get(agent)
            return (
                f"No goal is set. {USAGE}"
                if goal is None
                else render_goal("Current goal", goal)
            )
        if command.kind == "invalid-edit":
            return f"/goal edit requires an objective. {USAGE}"
        if command.kind == "create":
            created = goals.create(agent, command.objective)
            return render_goal("Created goal", created)
        current = goals.get(agent)
        if current is None:
            return _missing(command.kind)
        if command.kind == "edit":
            edited = goals.edit(agent, _ref(current), objective=command.objective)
            return render_goal("Edited goal", edited)
        if command.kind == "pause":
            return render_goal("Paused goal", goals.pause(agent, _ref(current)))
        if command.kind == "resume":
            return render_goal("Resumed goal", goals.resume(agent, _ref(current)))
        tombstone = goals.clear(agent, _ref(current))
        del tombstone
        return "Cleared the current goal."
    except GoalError as error:
        return f"/goal: {error.message}"


__all__ = [
    "USAGE",
    "parse_goal_command",
    "render_goal",
    "run_goal_command",
]
