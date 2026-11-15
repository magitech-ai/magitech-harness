"""Shared pure fold of a Session's turn and step boundary events."""

from __future__ import annotations

from typing import Any, Literal, cast

from magitech_harness.agent.contracts import StepBoundary, TurnBoundaryProjection
from magitech_harness.llm.values import is_non_negative_safe_integer
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import ProjectionDefinition


def _validate_optional_seq(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if not is_non_negative_safe_integer(value):
        raise ValueError(f"turnBoundary.{field} must be a non-negative int or None")
    return value


def _validate_turn_boundary_state(value: Any) -> TurnBoundaryProjection:
    """Validate serialized or live TurnBoundaryProjection state."""
    if not isinstance(value, dict):
        raise TypeError("turnBoundary state must be a dict")
    for required_key in ("openTurnStartSeq", "lastStepStartSeq", "lastStepBoundary"):
        if required_key not in value:
            raise ValueError(f"turnBoundary.{required_key} is required")
    open_seq = _validate_optional_seq(value["openTurnStartSeq"], "openTurnStartSeq")
    step_seq = _validate_optional_seq(value["lastStepStartSeq"], "lastStepStartSeq")
    raw_boundary = value["lastStepBoundary"]
    boundary: StepBoundary | None = None
    if raw_boundary is not None:
        if not isinstance(raw_boundary, dict):
            raise ValueError("turnBoundary.lastStepBoundary must be a dict or None")
        kind = raw_boundary.get("kind")
        if kind not in ("start", "end"):
            raise ValueError("turnBoundary.lastStepBoundary.kind invalid")
        seq = _validate_optional_seq(raw_boundary.get("seq"), "lastStepBoundary.seq")
        if seq is None:
            raise ValueError("turnBoundary.lastStepBoundary.seq must be an int")
        boundary = {"kind": cast(Literal["start", "end"], kind), "seq": seq}
    last_turn = value.get("lastTurn")
    if not is_non_negative_safe_integer(last_turn):
        raise ValueError("turnBoundary.lastTurn must be a non-negative int")
    return {
        "openTurnStartSeq": open_seq,
        "lastStepStartSeq": step_seq,
        "lastStepBoundary": boundary,
        "lastTurn": last_turn,
    }


def _init_turn_boundary(
    _header: dict[str, Any], _inherited_event_count: int
) -> TurnBoundaryProjection:
    return {
        "openTurnStartSeq": None,
        "lastStepStartSeq": None,
        "lastStepBoundary": None,
        "lastTurn": 0,
    }


def _apply_turn_boundary(
    state: TurnBoundaryProjection, event: SessionEvent
) -> TurnBoundaryProjection:
    if event.type == "turn/start":
        return {
            **state,
            "openTurnStartSeq": event.seq,
            "lastTurn": event.payload["turn"],
        }
    if event.type == "turn/end":
        return {**state, "openTurnStartSeq": None}
    if event.type == "step/start":
        return {
            **state,
            "lastStepStartSeq": event.seq,
            "lastStepBoundary": {"kind": "start", "seq": event.seq},
        }
    if event.type == "step/end":
        return {
            **state,
            "lastStepBoundary": {"kind": "end", "seq": event.seq},
        }
    return state


turn_boundary_projection_definition: ProjectionDefinition[TurnBoundaryProjection] = (
    ProjectionDefinition(
        key="turnBoundary",
        state_version=2,
        state_schema=_validate_turn_boundary_state,
        init=_init_turn_boundary,
        apply=_apply_turn_boundary,
    )
)
