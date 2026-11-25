"""Plan-mode vocabulary: stable errors and durable selection state."""

from __future__ import annotations

from typing import Any, Literal

from magitech_harness.llm.error import HarnessError

PlanPhase = Literal["inactive", "active"]


class PlanError(HarnessError):
    """Stable failure taxonomy for plan mode."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


def init_state() -> dict[str, Any]:
    """Initial fold state: no selection, no mode, never entered."""
    return {"active": False, "wanted": None, "running": None, "entered": False}


def pending_view(state: dict[str, Any]) -> dict[str, bool]:
    """The client-facing `{ active, pending }` view of plan state."""
    wanted = (
        state["running"]["wanted"]
        if state.get("running") is not None
        else state.get("wanted")
    )
    return {
        "active": state["active"],
        "pending": wanted is not None and wanted != state["active"],
    }


def validate_state(value: Any) -> dict[str, Any]:
    """Validate one persisted plan projection value."""
    if not isinstance(value, dict) or set(value) != {
        "active",
        "wanted",
        "running",
        "entered",
    }:
        raise TypeError(
            "plan projection must have exactly active, wanted, running, entered"
        )
    if not isinstance(value["active"], bool) or not isinstance(value["entered"], bool):
        raise TypeError("plan active and entered must be booleans")
    wanted = value["wanted"]
    if wanted is not None and not isinstance(wanted, bool):
        raise TypeError("plan wanted must be a boolean or null")
    running = value["running"]
    if running is not None and (
        not isinstance(running, dict)
        or set(running) != {"commandId", "wanted"}
        or not isinstance(running["commandId"], str)
        or not isinstance(running["wanted"], bool)
    ):
        raise TypeError("plan running must be a command selection or null")
    return {
        "active": value["active"],
        "wanted": wanted,
        "running": running,
        "entered": value["entered"],
    }


def validate_wire(value: Any) -> dict[str, bool]:
    """Validate the client view of one plan projection."""
    if not isinstance(value, dict) or set(value) != {"active", "pending"}:
        raise TypeError("plan wire value must have exactly active and pending")
    if not isinstance(value["active"], bool) or not isinstance(value["pending"], bool):
        raise TypeError("plan wire value must carry booleans")
    return value


__all__ = ["PlanError", "init_state", "pending_view", "validate_state", "validate_wire"]
