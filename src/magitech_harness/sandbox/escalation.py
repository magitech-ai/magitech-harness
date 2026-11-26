"""Sandbox escalation vocabulary and fail-closed approval sequence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from magitech_harness.sandbox.types import (
    SandboxError,
    SandboxMode,
)

WIDER_MODES: dict[str, tuple[SandboxMode, ...]] = {
    "read-only": ("workspace-write", "danger-full-access"),
    "workspace-write": ("danger-full-access",),
}
ESCALATION_TARGETS: tuple[SandboxMode, ...] = ("workspace-write", "danger-full-access")
SANDBOX_ESCALATION_INVALID_ARGUMENT = "SANDBOX_ESCALATION_INVALID_ARGUMENT"
SANDBOX_ESCALATION_UNAVAILABLE = "SANDBOX_ESCALATION_UNAVAILABLE"
SANDBOX_ESCALATION_REJECTED = "SANDBOX_ESCALATION_REJECTED"
SANDBOX_ESCALATION_CANCELLED = "SANDBOX_ESCALATION_CANCELLED"
SANDBOX_ESCALATION_ERROR_CODES = (
    SANDBOX_ESCALATION_INVALID_ARGUMENT,
    SANDBOX_ESCALATION_UNAVAILABLE,
    SANDBOX_ESCALATION_REJECTED,
    SANDBOX_ESCALATION_CANCELLED,
)


class SandboxEscalationError(SandboxError):
    """Stable failure taxonomy for a sandbox escalation."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


def validate_escalation_args(sandbox_permissions: Any, justification: Any) -> None:
    """Validate the pairing and the non-blank justification."""
    if sandbox_permissions is not None and justification is None:
        raise SandboxEscalationError(
            "invalid escalation: sandbox_permissions requires a justification",
            SANDBOX_ESCALATION_INVALID_ARGUMENT,
        )
    if justification is not None and sandbox_permissions is None:
        raise SandboxEscalationError(
            "invalid escalation: justification is only valid together with sandbox_permissions",
            SANDBOX_ESCALATION_INVALID_ARGUMENT,
        )
    if justification is not None and (
        not isinstance(justification, str) or not justification.strip()
    ):
        raise SandboxEscalationError(
            "invalid justification: expected a non-empty sentence",
            SANDBOX_ESCALATION_INVALID_ARGUMENT,
        )


def sandbox_denial_marker(mode: str) -> str:
    """The model-facing marker for a file-policy denial."""
    return f"[sandbox: file access denied under {mode} mode]"


def escalation_hint_marker(subject: str) -> str:
    """The model-facing one-shot escalation hint."""
    return (
        f"[sandbox: escalation available — retry this exact {subject} once with "
        "sandbox_permissions (the narrowest wider mode that suffices) + "
        "justification; the approval prompt asks the user]"
    )


def sandbox_permissions_description(subject: str) -> str:
    """Tool parameter description for ``sandbox_permissions``."""
    return (
        f"The narrowest wider sandbox mode for a one-shot retry of the exact "
        f"{subject} the sandbox just denied; the retry asks the user for approval."
    )


def escalation_display_reason(mode: str, justification: str) -> str:
    """Human-readable approval reason."""
    return f"Allow this operation with {mode} permissions: {justification}"


async def approve_escalation(
    request: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> str:
    """Resolve one escalation request through a fail-closed approver."""
    requested = request.get("requestedMode")
    effective = request.get("effectiveMode")
    if requested == effective:
        return str(requested)
    if requested not in WIDER_MODES.get(str(effective), ()):
        raise SandboxEscalationError(
            f'sandbox escalation to "{requested}" is not strictly wider than '
            f'this call\'s current "{effective}" mode',
            SANDBOX_ESCALATION_INVALID_ARGUMENT,
        )
    approver = approval.get("approver")
    if approver is None:
        raise SandboxEscalationError(
            "sandbox escalation requires approval, but no approval service is composed",
            SANDBOX_ESCALATION_UNAVAILABLE,
        )
    agent = approval.get("agent")
    if agent is None:
        raise SandboxEscalationError(
            "sandbox escalation requires approval, but the call has no agent to route it through",
            SANDBOX_ESCALATION_UNAVAILABLE,
        )
    justification = str(request.get("justification", ""))
    reason = f"escalate sandbox to {requested}: {justification}"
    payload: dict[str, Any] = {
        "agent": agent,
        "toolName": approval.get("toolName"),
        "callId": approval.get("callId"),
        "reason": reason,
        "displayReason": {
            "en": escalation_display_reason(str(requested), justification)
        },
        "signal": approval.get("signal"),
    }
    outcome = (
        approver.request(payload) if hasattr(approver, "request") else approver(payload)
    )
    if hasattr(outcome, "__await__"):
        outcome = await outcome
    status = (
        outcome.get("status")
        if isinstance(outcome, Mapping)
        else getattr(outcome, "status", outcome)
    )
    if status in ("allowed-once", "allowed_once"):
        return str(requested)
    if status == "rejected":
        raise SandboxEscalationError(
            f"the user rejected escalating this {request.get('subject', 'operation')} "
            f'to "{requested}"; it stays denied, so stop and explain instead of '
            "working around it",
            SANDBOX_ESCALATION_REJECTED,
        )
    if status == "cancelled":
        signal = approval.get("signal")
        if signal is not None and hasattr(signal, "throw_if_aborted"):
            signal.throw_if_aborted()
        raise SandboxEscalationError(
            f'approval for escalating to "{requested}" was cancelled',
            SANDBOX_ESCALATION_CANCELLED,
        )
    if status == "unavailable":
        raise SandboxEscalationError(
            "sandbox escalation requires approval, but no approval channel is available",
            SANDBOX_ESCALATION_UNAVAILABLE,
        )
    raise SandboxEscalationError(
        f"unknown sandbox escalation outcome {status!r}",
        SANDBOX_ESCALATION_INVALID_ARGUMENT,
    )


__all__ = [
    "ESCALATION_TARGETS",
    "SANDBOX_ESCALATION_CANCELLED",
    "SANDBOX_ESCALATION_ERROR_CODES",
    "SANDBOX_ESCALATION_INVALID_ARGUMENT",
    "SANDBOX_ESCALATION_REJECTED",
    "SANDBOX_ESCALATION_UNAVAILABLE",
    "WIDER_MODES",
    "SandboxEscalationError",
    "approve_escalation",
    "escalation_display_reason",
    "escalation_hint_marker",
    "sandbox_denial_marker",
    "sandbox_permissions_description",
    "validate_escalation_args",
]
