"""``ctx.planMode``: logged per-agent planning state and step-boundary admission.

Ports ``PlanModeController`` from ``packages/plan/plan-mode/src/index.ts``.
State lives in the ``plan`` projection, so it survives resume and fork. A
selection made mid-turn is committed at the next accepted in-turn pre-step,
before the model request is assembled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from animas import Context, EventMode, Service

from magitech_harness.llm.message import create_user_message
from magitech_harness.plan.fold import PLAN_PROJECTION
from magitech_harness.plan.types import PlanError
from magitech_harness.session.projection_registry import ProjectionHandle

EXIT_PLAN_MODE = "exit_plan_mode"
REVIEW_ID = "plan-review"
APPROVE_LABEL = "Approve"
KEEP_PLANNING_LABEL = "Keep planning"

EXIT_DESCRIPTION = (
    "Use only in plan mode. Present your plan for the user's review and, on "
    "approval, leave plan mode. The user may approve (carry out the plan from "
    "your next step) or keep planning — their feedback comes back in the tool "
    "result; revise and present again."
)


def resolve_config(config: Mapping[str, Any]) -> str:
    """Validate deployment-owned plan guidance."""
    section = config.get("section")
    if not isinstance(section, str):
        raise PlanError(
            "PlanModeConfig needs a string `section`", "PLAN_INVALID_CONFIG"
        )
    if not section.strip():
        raise PlanError(
            "PlanModeConfig needs a non-empty `section`", "PLAN_INVALID_CONFIG"
        )
    unknown = set(config) - {"section"}
    if unknown:
        raise PlanError(
            f"PlanModeConfig has unknown key(s) {', '.join(sorted(unknown))}",
            "PLAN_INVALID_CONFIG",
        )
    return section


class PlanModeController(Service):
    """Owns logged plan state, prompt sections, and the conditional exit tool."""

    inject: ClassVar[list[str]] = [
        "tools",
        "systemPrompt",
        "sessionProjections",
        "agents",
        "userQuestions",
    ]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "planMode")
        self.section = resolve_config(config or {})
        self._handle: ProjectionHandle[dict[str, Any]] | None = None
        self._pending: dict[int, dict[str, Any]] = {}
        self._disposed = False

    async def init(self) -> Callable[[], None]:
        """Register the projection, prompt sections, and pre-step listener."""
        registry = self.ctx.get("sessionProjections", None, strict=True)
        if registry is None:
            raise RuntimeError("planMode requires the sessionProjections service")
        self._handle = registry.register(PLAN_PROJECTION)

        system_prompt = self.ctx.get("systemPrompt", None, strict=False)
        if system_prompt is not None:
            order = system_prompt.get_section_order("PLAN_POLICY")
            system_prompt.section(
                {
                    "name": "plan:policy",
                    "retirement": "Plan mode is no longer available. Its instructions no longer apply.",
                    "order": order,
                    "text": lambda context: self._policy_text(context),
                }
            )
            system_prompt.section(
                {
                    "name": "plan:status",
                    "order": order + 1,
                    "replacement": "",
                    "text": lambda context: self._status_text(context),
                }
            )

        def on_pre_step(payload: Mapping[str, Any], next_fn: Any) -> Any:
            return self._on_pre_step(payload, next_fn)

        dispose_listener = self.ctx.on(
            "agent/pre-step", on_pre_step, mode=EventMode.WATERFALL
        )

        def mark_disposed() -> None:
            self._disposed = True

        self.ctx.effect(lambda: mark_disposed, "dsh-plan-mode: close service lifetime")

        def dispose() -> None:
            self._disposed = True
            dispose_listener()
            if self._handle is not None:
                self._handle.dispose()
                self._handle = None

        return dispose

    # -- state -------------------------------------------------------------

    def _state(self, agent: Any) -> dict[str, Any]:
        if self._handle is None:
            raise PlanError(
                "plan mode is not ready; await plugin activation before running an agent",
                "PLAN_STATE_UNAVAILABLE",
            )
        return self._handle.state_of(agent.session)

    def _session_key(self, agent: Any) -> int:
        return id(agent.session)

    def logged_active(self, agent: Any) -> bool:
        """Whether plan mode is committed for the agent's session."""
        return bool(self._state(agent)["active"])

    def entered(self, agent: Any) -> bool:
        """Whether the session has ever entered plan mode."""
        return bool(self._state(agent)["entered"])

    def selected_active(self, agent: Any) -> bool:
        """The state a proposed step presents: pending selection, else logged."""
        pending = self._pending.get(self._session_key(agent))
        return pending["active"] if pending is not None else self.logged_active(agent)

    def set(self, agent: Any, active: bool, *, narrate: bool = True) -> str:
        """Select a mode; return ``committed``, ``queued``, ``cancelled`` or ``noop``."""
        wanted = self.selected_active(agent)
        if active == wanted and self._state(agent)["wanted"] is None:
            return "noop"
        session = agent.session
        # A committed selection can be appended outside an open turn; otherwise
        # stage it for the next accepted in-turn pre-step.
        if self._in_open_turn(session) or active != self.logged_active(agent):
            self._pending[self._session_key(agent)] = {
                "active": active,
                "narrate": narrate,
            }
            return "queued" if active != wanted else "cancelled"
        session.append("plan/mode", {"active": active})
        self._pending.pop(self._session_key(agent), None)
        inject = getattr(agent, "inject", None)
        if callable(inject):
            inject(self._narration(active))
        return "committed"

    def _in_open_turn(self, session: Any) -> bool:
        agents = self.ctx.get("agents", None, strict=False)
        if agents is None:
            return False
        boundary = getattr(agents, "turn_boundary_of", None)
        if boundary is None:
            return False
        value = boundary(session)
        return bool((value or {}).get("openTurnStartSeq") is not None)

    def _on_pre_step(self, payload: Mapping[str, Any], next_fn: Any) -> Any:
        """Commit a staged selection after the step is accepted."""
        import asyncio

        async def wrap() -> Any:
            decision = next_fn()
            if hasattr(decision, "__await__"):
                decision = await decision
            if decision.get("kind") == "reject":
                return decision
            agent = payload.get("agent")
            signal = payload.get("signal")
            if signal is not None and signal.aborted:
                return decision
            if agent is None:
                return decision
            key = self._session_key(agent)
            pending = self._pending.get(key)
            if pending is None:
                return decision
            # Delete only after the append succeeds so a later accepted
            # in-turn pre-step can retry a failed durable write.
            try:
                agent.session.append("plan/mode", {"active": pending["active"]})
            except Exception:  # noqa: BLE001 -- a failed append stays pending for a later step
                return decision
            self._pending.pop(key, None)
            if not pending["narrate"]:
                return decision
            message = self._narration(pending["active"])
            return {
                **decision,
                "messages": [*decision.get("messages", []), message],
            }

        return asyncio.ensure_future(wrap()) if callable(next_fn) else wrap()

    @staticmethod
    def _narration(target: bool) -> dict[str, Any]:
        text = (
            "The user switched this session to plan mode."
            if target
            else "The user switched this session back to the default mode."
        )
        return create_user_message(
            content=[{"type": "text", "text": text}],
            source={"kind": "plan-mode", "form": "notice", "summary": text},
        )

    # -- prompt sections ---------------------------------------------------

    def _policy_text(self, context: Mapping[str, Any]) -> str:
        agent = context.get("agent")
        if agent is None:
            return ""
        pending = self._pending.get(self._session_key(agent))
        state = self._state(agent)
        if not (state["entered"] or (pending is not None and pending["active"])):
            return ""
        return f"The following instructions apply only while plan mode is active.\n\n{self.section}"

    def _status_text(self, context: Mapping[str, Any]) -> str:
        agent = context.get("agent")
        if agent is None:
            return ""
        state = self._state(agent)
        active = self.selected_active(agent)
        if not state["entered"] and not active:
            return ""
        return (
            "Plan mode is active. Follow the plan-mode instructions."
            if active
            else "Plan mode is inactive. The plan-mode instructions no longer apply."
        )


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> PlanModeController:
    """Mount ``ctx.planMode`` (service entrypoint for the loader)."""
    return PlanModeController(ctx, config)


__all__ = [
    "APPROVE_LABEL",
    "EXIT_DESCRIPTION",
    "EXIT_PLAN_MODE",
    "KEEP_PLANNING_LABEL",
    "REVIEW_ID",
    "PlanModeController",
    "apply",
    "resolve_config",
]
