"""GoalService (``ctx.goals``): event-sourced, compare-and-set session goals.

Ports DSH ``GoalService`` at ``packages/goal/goal/src/index.ts``. Every
mutation appends exactly one ``goal/change`` event and returns the fresh view.
Durable phase is replay state; activation is process-local and never persisted
(a resumed or forked session starts disarmed).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from animas import Context, Service

from magitech_harness.goal.projection import (
    GOAL_PROJECTION,
    current_projection,
    snapshot_to_projection,
)
from magitech_harness.goal.types import (
    GOAL_CHANGE_VERSION,
    GoalActivation,
    GoalBlockReason,
    GoalError,
    GoalPhase,
    GoalProjection,
    GoalRef,
    GoalSnapshot,
    GoalView,
)
from magitech_harness.session.projection_registry import ProjectionHandle

DEFAULT_MAX_GOAL_ROUNDS = 256

_GOAL_ID_PREFIX = "goal-"
_CODE_MIN_LENGTH = 1


@dataclass
class _RuntimeState:
    activation: GoalActivation = "disarmed"


def _resolve_objective(objective: Any) -> str:
    if not isinstance(objective, str):
        raise GoalError("goal objective must be a string", "GOAL_INVALID_CREATE")
    trimmed = objective.strip()
    if not trimmed:
        raise GoalError(
            "goal objective must be a non-empty string", "GOAL_INVALID_CREATE"
        )
    if trimmed != objective:
        raise GoalError(
            "goal objective must be normalized (no surrounding whitespace)",
            "GOAL_INVALID_CREATE",
        )
    return objective


def _resolve_max_rounds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GoalError(
            "maxGoalRounds must be a positive safe integer", "GOAL_INVALID_CREATE"
        )
    return value


def _resolve_block_reason(reason: Any) -> GoalBlockReason:
    if not isinstance(reason, GoalBlockReason):
        raise GoalError("goal block reason is invalid", "GOAL_INVALID_TRANSITION")
    if not reason.code or len(reason.code) < _CODE_MIN_LENGTH:
        raise GoalError(
            "goal block reason code must be a non-empty string",
            "GOAL_INVALID_TRANSITION",
        )
    if not reason.message or reason.message != reason.message.strip():
        raise GoalError(
            "goal block reason message must be non-empty and normalized",
            "GOAL_INVALID_TRANSITION",
        )
    return reason


class GoalService(Service):
    """``ctx.goals``: the same-session goal domain."""

    inject: ClassVar[list[str]] = ["agents", "sessionProjections"]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "goals")
        cfg = dict(config or {})
        self._default_max_goal_rounds = _resolve_max_rounds(
            cfg.get("defaultMaxGoalRounds", DEFAULT_MAX_GOAL_ROUNDS)
        )
        self._runtime: dict[str, _RuntimeState] = {}
        self._pending: dict[str, tuple[int, GoalActivation]] = {}
        self._handle: ProjectionHandle[dict[str, Any]] | None = None
        self._listen: Callable[[], Any] | None = None

    async def init(self) -> Callable[[], None]:
        """Register the ``goal`` projection and subscribe to goal changes."""
        registry = self.ctx.get("sessionProjections", None, strict=True)
        if registry is None:
            raise RuntimeError("goals requires the sessionProjections service")
        self._handle = registry.register(GOAL_PROJECTION)
        self._listen = self.ctx.on("session/event", self._on_event)

        def dispose() -> None:
            if self._listen is not None:
                self._listen()
            if self._handle is not None:
                self._handle.dispose()

        return dispose

    # -- reads -----------------------------------------------------------

    def _state_of(self, session: Any) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("goals requires the sessionProjections service")
        return self._handle.state_of(session)

    def _projection(self, session: Any) -> GoalProjection | None:
        return current_projection(self._state_of(session))

    def _runtime_state(self, session: Any) -> _RuntimeState:
        key = session.session_id
        state = self._runtime.get(key)
        if state is None:
            state = _RuntimeState()
            self._runtime[key] = state
        return state

    def get(self, agent: Any) -> GoalView | None:
        """Read the current goal for one exact live agent."""
        self._assert_live(agent)
        projection = self._projection(agent.session)
        if projection is None:
            return None
        return self._view(projection, self._runtime_state(agent.session).activation)

    def disarm(self, agent: Any) -> GoalView | None:
        """Remove process-local continuation authority without a durable write."""
        self._assert_live(agent)
        runtime = self._runtime_state(agent.session)
        runtime.activation = "disarmed"
        projection = self._projection(agent.session)
        if projection is None:
            return None
        return self._view(projection, runtime.activation)

    @staticmethod
    def _view(projection: GoalProjection, activation: GoalActivation) -> GoalView:
        return GoalView(
            goal=projection.goal,
            rounds_started=projection.rounds_started,
            created_at=projection.created_at,
            updated_at=projection.updated_at,
            activation=activation,
        )

    # -- mutations --------------------------------------------------------

    def create(
        self, agent: Any, objective: str, max_goal_rounds: int | None = None
    ) -> GoalView:
        """Create and arm a goal; a completed goal may be replaced."""
        self._assert_live(agent)
        resolved_objective = _resolve_objective(objective)
        resolved_rounds = (
            self._default_max_goal_rounds
            if max_goal_rounds is None
            else _resolve_max_rounds(max_goal_rounds)
        )
        projection = self._projection(agent.session)
        if projection is not None and projection.goal.phase != "complete":
            raise GoalError(
                f'goal "{projection.goal.id}" already exists with phase '
                f'"{projection.goal.phase}"',
                "GOAL_ALREADY_EXISTS",
            )
        now = _now_ms()
        snapshot = GoalSnapshot(
            id=f"{_GOAL_ID_PREFIX}{uuid.uuid4()}",
            revision=1,
            objective=resolved_objective,
            phase="active",
            max_goal_rounds=resolved_rounds,
        )
        return self._commit_new(agent, snapshot, 0, now, now, "armed")

    def edit(
        self,
        agent: Any,
        ref: GoalRef,
        *,
        objective: str | None = None,
        max_goal_rounds: int | None = None,
    ) -> GoalView:
        """Edit the objective and/or round cap without changing phase."""
        self._assert_live(agent)
        projection = self._expect_current(agent.session, ref)
        if objective is None and max_goal_rounds is None:
            raise GoalError(
                "goal edit requires objective and/or maxGoalRounds",
                "GOAL_INVALID_EDIT",
            )
        current = projection.goal
        snapshot = GoalSnapshot(
            id=current.id,
            revision=current.revision + 1,
            objective=(
                _resolve_objective(objective)
                if objective is not None
                else current.objective
            ),
            phase=current.phase,
            max_goal_rounds=(
                _resolve_max_rounds(max_goal_rounds)
                if max_goal_rounds is not None
                else current.max_goal_rounds
            ),
            blocked_reason=current.blocked_reason,
        )
        return self._commit_current(
            agent, projection, snapshot, self._runtime_state(agent.session).activation
        )

    def pause(self, agent: Any, ref: GoalRef) -> GoalView:
        """Pause an active goal and disarm automatic continuation."""
        return self._transition(agent, ref, "pause", ("active",), "paused", "disarmed")

    def resume(self, agent: Any, ref: GoalRef) -> GoalView:
        """Resume and arm a stopped goal while its round budget has capacity."""
        self._assert_live(agent)
        projection = self._expect_current(agent.session, ref)
        current = projection.goal
        runtime = self._runtime_state(agent.session)
        resumable: tuple[GoalPhase, ...] = ("active", "paused", "blocked")
        if current.phase not in resumable:
            raise self._transition_error(current, "resume", resumable)
        if current.phase == "active" and runtime.activation == "armed":
            raise GoalError(
                f'goal "{current.id}" is already active and armed',
                "GOAL_INVALID_TRANSITION",
            )
        if projection.rounds_started >= current.max_goal_rounds:
            raise GoalError(
                f'goal "{current.id}" exhausted {current.max_goal_rounds} goal '
                "rounds; increase maxGoalRounds before resuming",
                "GOAL_INVALID_TRANSITION",
            )
        snapshot = _with_phase(current, "active")
        return self._commit_current(agent, projection, snapshot, "armed")

    def complete(self, agent: Any, ref: GoalRef) -> GoalView:
        """Mark a current non-complete goal complete and disarm it."""
        return self._transition(
            agent,
            ref,
            "complete",
            ("active", "paused", "blocked"),
            "complete",
            "disarmed",
        )

    def block(self, agent: Any, ref: GoalRef, reason: GoalBlockReason) -> GoalView:
        """Mark an active goal blocked and disarm it."""
        self._assert_live(agent)
        projection = self._expect_current(agent.session, ref)
        current = projection.goal
        if current.phase != "active":
            raise self._transition_error(current, "block", ("active",))
        snapshot = GoalSnapshot(
            id=current.id,
            revision=current.revision + 1,
            objective=current.objective,
            phase="blocked",
            max_goal_rounds=current.max_goal_rounds,
            blocked_reason=_resolve_block_reason(reason),
        )
        return self._commit_current(agent, projection, snapshot, "disarmed")

    def clear(self, agent: Any, ref: GoalRef) -> GoalRef:
        """Clear the current goal while retaining a durable tombstone."""
        self._assert_live(agent)
        projection = self._expect_current(agent.session, ref)
        current = projection.goal
        tombstone = GoalRef(id=current.id, revision=current.revision + 1)
        change = {
            "kind": "goal/change",
            "version": GOAL_CHANGE_VERSION,
            "operation": "clear",
            "cleared": {"id": tombstone.id, "revision": tombstone.revision},
            "clearedAt": self._next_mutation_time(projection),
        }
        self._append_change(agent, change, "disarmed")
        return tombstone

    # -- internals --------------------------------------------------------

    def _assert_live(self, agent: Any) -> None:
        agents = self.ctx.get("agents", None, strict=False)
        if agents is None:
            raise GoalError("goals requires the agent registry", "GOAL_AGENT_REQUIRED")
        if agents.get(agent.id) is not agent:
            raise GoalError(
                "goal mutation requires the exact live agent", "GOAL_AGENT_REQUIRED"
            )

    def _expect_current(self, session: Any, ref: GoalRef) -> GoalProjection:
        projection = self._projection(session)
        if projection is None:
            raise GoalError("no current goal", "GOAL_NOT_FOUND")
        current = projection.goal
        if current.id != ref.id or current.revision != ref.revision:
            raise GoalError(
                f'goal "{ref.id}" revision {ref.revision} is not the current '
                f'revision {current.revision} of goal "{current.id}"',
                "GOAL_STALE_REVISION",
            )
        return projection

    @staticmethod
    def _transition_error(
        current: GoalSnapshot, operation: str, allowed: tuple[GoalPhase, ...]
    ) -> GoalError:
        return GoalError(
            f'cannot {operation} goal "{current.id}" from phase "{current.phase}"; '
            f"allowed: {', '.join(allowed)}",
            "GOAL_INVALID_TRANSITION",
        )

    def _transition(
        self,
        agent: Any,
        ref: GoalRef,
        operation: str,
        allowed: tuple[GoalPhase, ...],
        phase: GoalPhase,
        activation: GoalActivation,
    ) -> GoalView:
        self._assert_live(agent)
        projection = self._expect_current(agent.session, ref)
        current = projection.goal
        if current.phase not in allowed:
            raise self._transition_error(current, operation, allowed)
        return self._commit_current(
            agent, projection, _with_phase(current, phase), activation
        )

    @staticmethod
    def _next_mutation_time(projection: GoalProjection) -> int:
        """Monotonic wall-clock stamp that never precedes the last update."""
        return max(_now_ms(), projection.updated_at)

    def _commit_new(
        self,
        agent: Any,
        snapshot: GoalSnapshot,
        rounds_started: int,
        created_at: int,
        updated_at: int,
        activation: GoalActivation,
    ) -> GoalView:
        change = {
            "kind": "goal/change",
            "version": GOAL_CHANGE_VERSION,
            "operation": "create",
            "goal": snapshot_to_projection(snapshot),
            "roundsStarted": rounds_started,
            "createdAt": created_at,
            "updatedAt": updated_at,
        }
        return self._commit_event(agent, change, activation)

    def _commit_current(
        self,
        agent: Any,
        projection: GoalProjection,
        snapshot: GoalSnapshot,
        activation: GoalActivation,
    ) -> GoalView:
        change = {
            "kind": "goal/change",
            "version": GOAL_CHANGE_VERSION,
            "operation": _operation_for(projection.goal, snapshot),
            "goal": snapshot_to_projection(snapshot),
            "roundsStarted": projection.rounds_started,
            "createdAt": projection.created_at,
            "updatedAt": self._next_mutation_time(projection),
        }
        return self._commit_event(agent, change, activation)

    def _commit_event(
        self, agent: Any, change: dict[str, Any], activation: GoalActivation
    ) -> GoalView:
        session = agent.session
        seq = session.seq
        self._pending[session.session_id] = (seq, activation)
        try:
            session.append("goal/change", change)
        finally:
            self._pending.pop(session.session_id, None)
        self._runtime_state(session).activation = activation
        projection = self._projection(session)
        if projection is None:
            raise GoalError("goal mutation left no current goal", "GOAL_REPLAY_INVALID")
        return self._view(projection, activation)

    def _append_change(
        self, agent: Any, change: dict[str, Any], activation: GoalActivation
    ) -> None:
        """Append one change without requiring a current goal (the clear path)."""
        session = agent.session
        seq = session.seq
        self._pending[session.session_id] = (seq, activation)
        try:
            session.append("goal/change", change)
        finally:
            self._pending.pop(session.session_id, None)
        self._runtime_state(session).activation = activation

    def _on_event(self, session: Any, event: Any) -> None:
        if event.type != "goal/change":
            return
        pending = self._pending.get(session.session_id)
        activation: GoalActivation = (
            pending[1]
            if pending is not None and pending[0] == event.seq
            else "disarmed"
        )
        self._runtime_state(session).activation = activation


def _with_phase(snapshot: GoalSnapshot, phase: GoalPhase) -> GoalSnapshot:
    return GoalSnapshot(
        id=snapshot.id,
        revision=snapshot.revision + 1,
        objective=snapshot.objective,
        phase=phase,
        max_goal_rounds=snapshot.max_goal_rounds,
        blocked_reason=None,
    )


def _operation_for(current: GoalSnapshot, nxt: GoalSnapshot) -> str:
    """Derive the durable operation name from the phase transition."""
    if nxt.phase == current.phase:
        return "edit"
    return {
        "paused": "pause",
        "active": "resume",
        "complete": "complete",
        "blocked": "block",
    }[nxt.phase]


def _now_ms() -> int:
    return int(time.time() * 1000)


__all__ = [
    "DEFAULT_MAX_GOAL_ROUNDS",
    "GoalService",
]
