"""ApprovalService (ctx.approval): turn-enclosed authorization with audit.

The service applies a session's approval policy before answerers and logs
every ask/outcome pair to the requesting session. A grant is one-shot: the
only positive outcome is ``allowed-once``; there is no persistent cache keyed
on arguments (a grant never authorizes a different tool or a later call).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypedDict, cast

from anima import Context, Service

from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    ProjectionHandle,
)

ApprovalOutcome = Literal["allowed-once", "rejected", "cancelled", "unavailable"]
ApprovalPolicy = Literal["ask", "never"]

ALL_OUTCOMES: frozenset[str] = frozenset(
    {"allowed-once", "rejected", "cancelled", "unavailable"}
)
ALL_POLICIES: frozenset[str] = frozenset({"ask", "never"})


def calculate_argument_digest(arguments: dict[str, Any]) -> str:
    """Canonical sha256 digest over sorted argument JSON (identity only)."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


class ApprovalRequest(TypedDict, total=False):
    """A pending decision: the agent, tool identity, reason and signal."""

    agent: Any
    toolName: str
    callId: str
    reason: str
    displayReason: dict[str, str]
    signal: Any


class _ApprovalState(TypedDict):
    policy: ApprovalPolicy | None
    turn_open: bool


def _validate_approval_state(value: Any) -> _ApprovalState:
    """Validate a restored approval state cell."""
    if not isinstance(value, dict):
        raise TypeError("approval state must be a record")
    policy = value.get("policy")
    if policy is not None and policy not in ALL_POLICIES:
        raise ValueError(f"invalid approval policy {policy!r}")
    turn_open = value.get("turn_open")
    if not isinstance(turn_open, bool):
        raise TypeError("approval turn_open must be a boolean")
    return {"policy": policy, "turn_open": turn_open}


def _init_approval_state(
    header: dict[str, Any], inherited_event_count: int
) -> _ApprovalState:
    return {"policy": None, "turn_open": False}


def _apply_approval_state(state: _ApprovalState, event: SessionEvent) -> _ApprovalState:
    if event.type == EventType.APPROVAL_POLICY:
        policy = event.payload.get("policy")
        if policy in ALL_POLICIES and policy != state["policy"]:
            return {**state, "policy": policy}
        return state
    if event.type == EventType.TURN_START and not state["turn_open"]:
        return {**state, "turn_open": True}
    if event.type == EventType.TURN_END and state["turn_open"]:
        return {**state, "turn_open": False}
    return state


APPROVAL_STATE_PROJECTION: ProjectionDefinition[_ApprovalState] = ProjectionDefinition(
    key="approvalState",
    state_schema=_validate_approval_state,
    init=_init_approval_state,
    apply=_apply_approval_state,
    state_version=1,
)


class ApprovalService(Service):
    """Broker for approval asks, bound to a live turn and audit-logged."""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(ctx, "approval")
        cfg = config or {}
        self._default_policy: ApprovalPolicy = cfg.get("policy", "ask")
        if self._default_policy not in ALL_POLICIES:
            raise ValueError(
                f"approval policy must be ask|never, got {self._default_policy!r}"
            )
        self._handler: Callable[[ApprovalRequest], Awaitable[str]] | None = None
        self._lock = asyncio.Lock()
        self._projection: ProjectionHandle[_ApprovalState] | None = None

    @property
    def policy(self) -> ApprovalPolicy:
        """The deployment default policy for sessions without an override."""
        return self._default_policy

    def set_handler(
        self, handler: Callable[[ApprovalRequest], Awaitable[str]] | None
    ) -> None:
        """Register an answerer callback (interactive UI)."""
        self._handler = handler

    def _state_of(self, session: Any) -> _ApprovalState:
        if self._projection is None:
            registry = self.ctx.get("sessionProjections", None, strict=False)
            if registry is None:
                return {"policy": None, "turn_open": True}
            self._projection = registry.register(APPROVAL_STATE_PROJECTION)
        return self._projection.state_of(session)

    def effective_policy(self, session: Any) -> ApprovalPolicy:
        """The session's override, else the configured default."""
        return self._state_of(session)["policy"] or self._default_policy

    def set_policy(self, agent: Any, policy: ApprovalPolicy) -> None:
        """Switch one session's approval policy (log-only, durable)."""
        if policy not in ALL_POLICIES:
            raise ValueError(f"approval policy must be ask|never, got {policy!r}")
        session = agent.session
        if self._state_of(session)["policy"] == policy:
            return
        session.append(EventType.APPROVAL_POLICY, {"policy": policy})

    async def request(self, req: ApprovalRequest) -> ApprovalOutcome:
        """Ask for one decision, enclosed by a turn and logged as an audit pair.

        Raises when there is no open turn: an approval event between turns is
        tail garbage on reload. Returns the closed outcome; ``allowed-once``
        is the only grant.
        """
        agent = req.get("agent")
        if agent is None:
            raise ValueError("approval request requires an agent")
        session = agent.session
        if not self._state_of(session)["turn_open"]:
            raise RuntimeError(
                "approval.request() outside an open turn: the approval/asked + "
                "approval/decided audit pair must be turn-enclosed"
            )

        request_id = uuid.uuid4().hex
        session.append(
            EventType.APPROVAL_ASKED,
            {
                "id": request_id,
                "toolName": req.get("toolName", ""),
                **({"callId": req["callId"]} if req.get("callId") else {}),
                **({"reason": req["reason"]} if req.get("reason") else {}),
            },
        )

        outcome = await self._decide(req, session)
        session.append(
            EventType.APPROVAL_DECIDED, {"id": request_id, "outcome": outcome}
        )
        return outcome

    async def _decide(self, req: ApprovalRequest, session: Any) -> ApprovalOutcome:
        signal = req.get("signal")
        if signal is not None and signal.is_aborted:
            return "cancelled"
        if self.effective_policy(session) == "never":
            return "rejected"

        async def _resolve() -> ApprovalOutcome:
            if self._handler is not None:
                try:
                    verdict = await self._handler(req)
                except Exception:  # noqa: BLE001 - answerer errors fail closed
                    return "unavailable"
                return cast(
                    ApprovalOutcome,
                    verdict if verdict in ALL_OUTCOMES else "unavailable",
                )
            try:
                verdict = await self.ctx.waterfall(
                    "approval/request", req, default_handler=lambda _r: "unavailable"
                )
            except Exception:  # noqa: BLE001 - answerer errors fail closed
                return "unavailable"
            return cast(
                ApprovalOutcome, verdict if verdict in ALL_OUTCOMES else "unavailable"
            )

        answer = asyncio.ensure_future(_resolve())
        if signal is None:
            return await answer

        loop = asyncio.get_running_loop()
        cancelled = loop.create_future()

        def _on_abort() -> None:
            if not cancelled.done():  # pragma: no branch - defensive re-entry
                cancelled.set_result(None)

        disposer = signal.on_abort(_on_abort)
        try:
            done, _ = await asyncio.wait(
                {answer, cancelled}, return_when=asyncio.FIRST_COMPLETED
            )
            if cancelled in done and answer not in done:
                answer.cancel()
                return "cancelled"
            return await answer
        finally:
            disposer()


__all__ = [
    "ALL_OUTCOMES",
    "ALL_POLICIES",
    "ApprovalOutcome",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalService",
    "calculate_argument_digest",
]
