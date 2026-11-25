"""The ``plan`` projection: durable mode selections folded from the session log.

Ports dsh's ``planProjectionDefinition`` from ``packages/plan/plan-mode/src/index.ts``.
A ``plan/mode`` event is log-only and whole-value replace; ``command/run`` and
``command/done`` are honoured when a command registry is composed, so the same
fold works with or without slash commands.
"""

from __future__ import annotations

from typing import Any

from magitech_harness.plan.types import (
    init_state,
    pending_view,
    validate_state,
    validate_wire,
)
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    WireView,
)


def apply_event(state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
    """Pure transition: previous state + one committed event -> next state."""
    payload = event.payload
    if event.type == "command/run" and payload.get("name") == "plan":
        if "args" not in payload:
            return state
        wanted = str(payload["args"]).strip() != "off"
        return {
            **state,
            "running": {"commandId": payload.get("commandId"), "wanted": wanted},
        }
    if (
        event.type == "command/done"
        and state["running"] is not None
        and payload.get("commandId") == state["running"]["commandId"]
    ):
        wanted = (
            payload.get("kind") == "success"
            and state["running"]["wanted"] != state["active"]
        )
        return {**state, "wanted": wanted if wanted else None, "running": None}
    if event.type == "plan/mode":
        active = payload.get("active")
        if not isinstance(active, bool):
            return state
        return {
            **state,
            "active": active,
            "wanted": None,
            "entered": state["entered"] or active,
        }
    return state


PLAN_PROJECTION: ProjectionDefinition[dict[str, Any]] = ProjectionDefinition(
    key="plan",
    state_schema=validate_state,
    init=lambda header, inherited_event_count: init_state(),
    apply=apply_event,
    wire=WireView(view_schema=validate_wire, view=pending_view, summary=False),
    state_version=5,
)


__all__ = ["PLAN_PROJECTION", "apply_event"]
