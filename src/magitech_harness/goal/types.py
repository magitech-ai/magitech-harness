"""Same-session goal domain vocabulary (ctx.goals).

Ports the DSH ``@deepseek-ai/dsh-goal`` domain at
``packages/goal/goal/src/{types,fold,runtime}.ts``: an event-sourced goal with
compare-and-set revisions, a durable phase, and a process-local activation that
is never persisted.

The durable fact is one ``goal/change`` session event per mutation. The
``goal`` projection folds those events into the current view; a matching
``user/message`` whose source kind is ``goal`` advances the admitted-round
counter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from magitech_harness.llm.error import HarnessError

GOAL_CHANGE_VERSION = 1

GoalPhase = Literal["active", "paused", "blocked", "complete"]
GoalActivation = Literal["armed", "disarmed"]
GoalOperation = Literal[
    "create", "edit", "pause", "resume", "complete", "block", "clear"
]

PHASES: frozenset[str] = frozenset({"active", "paused", "blocked", "complete"})
SNAPSHOT_OPERATIONS: frozenset[str] = frozenset(
    {"create", "edit", "pause", "resume", "complete", "block"}
)


class GoalError(HarnessError):
    """Guarded refusal from the goal domain boundary."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


@dataclass(frozen=True)
class GoalRef:
    """Compare-and-set identity for one exact goal revision."""

    id: str
    revision: int


@dataclass(frozen=True)
class GoalBlockReason:
    """Machine-routable and human-readable explanation for a blocked goal."""

    code: str
    message: str


@dataclass(frozen=True)
class GoalSnapshot:
    """Full durable state written by every non-clear goal mutation."""

    id: str
    revision: int
    objective: str
    phase: GoalPhase
    max_goal_rounds: int
    blocked_reason: GoalBlockReason | None = None


@dataclass(frozen=True)
class GoalProjection:
    """Current durable goal plus replay counters (activation excluded)."""

    goal: GoalSnapshot
    rounds_started: int
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class GoalView:
    """Model-facing view: durable state plus process-local activation."""

    goal: GoalSnapshot
    rounds_started: int
    created_at: int
    updated_at: int
    activation: GoalActivation

    @property
    def id(self) -> str:
        return self.goal.id

    @property
    def revision(self) -> int:
        return self.goal.revision

    @property
    def objective(self) -> str:
        return self.goal.objective

    @property
    def phase(self) -> GoalPhase:
        return self.goal.phase

    @property
    def max_goal_rounds(self) -> int:
        return self.goal.max_goal_rounds

    @property
    def blocked_reason(self) -> GoalBlockReason | None:
        return self.goal.blocked_reason


@dataclass
class GoalFoldState:
    """Mutable accumulator kept private to the pure fold."""

    goal: GoalSnapshot | None = None
    rounds_started: int = 0
    created_at: int | None = None
    updated_at: int | None = None
    last_ref: GoalRef | None = None
    seen_goal_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class GoalSnapshotChange:
    """One decoded full-snapshot goal mutation."""

    operation: str
    goal: GoalSnapshot
    rounds_started: int
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class GoalClearChange:
    """One decoded clear tombstone."""

    cleared: GoalRef
    cleared_at: int


GoalChange = GoalSnapshotChange | GoalClearChange


def _is_record(value: Any) -> bool:
    return isinstance(value, Mapping)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GoalError(
            f"goal change {field_name} must be a positive safe integer",
            "GOAL_REPLAY_INVALID",
        )
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GoalError(
            f"goal change {field_name} must be a non-negative safe integer",
            "GOAL_REPLAY_INVALID",
        )
    return value


_CODE_PATTERN_ERROR = "goal change goal.blockedReason.code must be lower-kebab-case"


def _decode_block_reason(value: Any) -> GoalBlockReason:
    if not _is_record(value) or set(value) != {"code", "message"}:
        raise GoalError(
            "goal change goal.blockedReason must have exactly code and message fields",
            "GOAL_REPLAY_INVALID",
        )
    code = value["code"]
    message = value["message"]
    if not isinstance(code, str) or not _is_kebab(code):
        raise GoalError(_CODE_PATTERN_ERROR, "GOAL_REPLAY_INVALID")
    if (
        not isinstance(message, str)
        or not message.strip()
        or message != message.strip()
    ):
        raise GoalError(
            "goal change goal.blockedReason.message must be non-empty and normalized",
            "GOAL_REPLAY_INVALID",
        )
    return GoalBlockReason(code=code, message=message)


def _is_kebab(value: str) -> bool:
    if not value:
        return False
    parts = value.split("-")
    for part in parts:
        if not part or not part.isascii() or not part.isalnum():
            return False
        if not part[0].isalpha() or part != part.lower():
            return False
    return True


def _decode_snapshot(value: Any) -> GoalSnapshot:
    if not _is_record(value):
        raise GoalError("goal change goal must be a record", "GOAL_REPLAY_INVALID")
    goal_id = value.get("id")
    if not isinstance(goal_id, str) or not goal_id:
        raise GoalError(
            "goal change goal.id must be a non-empty string", "GOAL_REPLAY_INVALID"
        )
    objective = value.get("objective")
    if (
        not isinstance(objective, str)
        or not objective.strip()
        or objective != objective.strip()
    ):
        raise GoalError(
            "goal change goal.objective must be non-empty and normalized",
            "GOAL_REPLAY_INVALID",
        )
    phase = value.get("phase")
    if not isinstance(phase, str) or phase not in PHASES:
        raise GoalError("goal change goal.phase is invalid", "GOAL_REPLAY_INVALID")
    expected = (
        {"blockedReason", "id", "maxGoalRounds", "objective", "phase", "revision"}
        if phase == "blocked"
        else {"id", "maxGoalRounds", "objective", "phase", "revision"}
    )
    if set(value) != expected:
        raise GoalError(
            f"goal change goal for phase {phase} must have exactly "
            f"{', '.join(sorted(expected))} fields",
            "GOAL_REPLAY_INVALID",
        )
    blocked = (
        _decode_block_reason(value["blockedReason"]) if phase == "blocked" else None
    )
    return GoalSnapshot(
        id=goal_id,
        revision=_positive_int(value.get("revision"), "goal.revision"),
        objective=objective,
        phase=phase,  # type: ignore[arg-type]
        max_goal_rounds=_positive_int(value.get("maxGoalRounds"), "goal.maxGoalRounds"),
        blocked_reason=blocked,
    )


def _decode_ref(value: Any) -> GoalRef:
    if not _is_record(value) or set(value) != {"id", "revision"}:
        raise GoalError(
            "goal clear tombstone must have exactly id and revision fields",
            "GOAL_REPLAY_INVALID",
        )
    cleared_id = value.get("id")
    if not isinstance(cleared_id, str) or not cleared_id:
        raise GoalError(
            "goal clear tombstone id must be a non-empty string",
            "GOAL_REPLAY_INVALID",
        )
    return GoalRef(
        id=cleared_id, revision=_positive_int(value.get("revision"), "cleared.revision")
    )


def decode_goal_change(value: Any) -> GoalChange | None:
    """Decode a value that declares itself as a goal change.

    Unrelated values return ``None``; malformed goal changes fail loudly.
    """
    if not _is_record(value) or value.get("kind") != "goal/change":
        return None
    if value.get("version") != GOAL_CHANGE_VERSION:
        raise GoalError(
            f"unsupported goal change version {value.get('version')!r}",
            "GOAL_REPLAY_INVALID",
        )
    operation = value.get("operation")
    if operation == "clear":
        allowed = {"cleared", "clearedAt", "kind", "operation", "version"}
        if set(value) != allowed:
            raise GoalError(
                "goal clear change must have exactly "
                f"{', '.join(sorted(allowed))} fields",
                "GOAL_REPLAY_INVALID",
            )
        return GoalClearChange(
            cleared=_decode_ref(value["cleared"]),
            cleared_at=_non_negative_int(value.get("clearedAt"), "clearedAt"),
        )
    if not isinstance(operation, str) or operation not in SNAPSHOT_OPERATIONS:
        raise GoalError("goal change operation is invalid", "GOAL_REPLAY_INVALID")
    allowed = {
        "createdAt",
        "goal",
        "kind",
        "operation",
        "roundsStarted",
        "updatedAt",
        "version",
    }
    if set(value) != allowed:
        raise GoalError(
            "goal snapshot change must have exactly "
            f"{', '.join(sorted(allowed))} fields",
            "GOAL_REPLAY_INVALID",
        )
    created_at = _non_negative_int(value.get("createdAt"), "createdAt")
    updated_at = _non_negative_int(value.get("updatedAt"), "updatedAt")
    if updated_at < created_at:
        raise GoalError(
            "goal change updatedAt cannot precede createdAt", "GOAL_REPLAY_INVALID"
        )
    return GoalSnapshotChange(
        operation=operation,
        goal=_decode_snapshot(value.get("goal")),
        rounds_started=_non_negative_int(value.get("roundsStarted"), "roundsStarted"),
        created_at=created_at,
        updated_at=updated_at,
    )


def change_ref(change: GoalChange) -> GoalRef:
    """Return the revision identity carried by a snapshot or tombstone."""
    if isinstance(change, GoalClearChange):
        return change.cleared
    return GoalRef(id=change.goal.id, revision=change.goal.revision)


def _require_next_revision(
    current: GoalSnapshot, nxt: GoalRef | GoalSnapshot, operation: str
) -> None:
    if nxt.id != current.id or nxt.revision != current.revision + 1:
        raise GoalError(
            f"goal {operation} must advance the current goal by one revision",
            "GOAL_REPLAY_INVALID",
        )


def _require_same_definition(
    current: GoalSnapshot, nxt: GoalSnapshot, operation: str
) -> None:
    if (
        nxt.objective != current.objective
        or nxt.max_goal_rounds != current.max_goal_rounds
    ):
        raise GoalError(
            f"goal {operation} cannot change objective or maxGoalRounds",
            "GOAL_REPLAY_INVALID",
        )


def _validate_snapshot_transition(
    state: GoalFoldState, change: GoalSnapshotChange, current: GoalSnapshot
) -> None:
    nxt = change.goal
    _require_next_revision(current, nxt, change.operation)
    if state.updated_at is None:
        raise GoalError("current goal fold lacks updatedAt", "GOAL_REPLAY_INVALID")
    if (
        change.created_at != state.created_at
        or change.updated_at < state.updated_at
        or change.rounds_started != state.rounds_started
    ):
        raise GoalError(
            f"goal {change.operation} does not preserve the current counters "
            "and timestamps",
            "GOAL_REPLAY_INVALID",
        )
    operation = change.operation
    if operation == "edit":
        if nxt.phase != current.phase or nxt.blocked_reason != current.blocked_reason:
            raise GoalError(
                "goal edit cannot change phase or blocked reason",
                "GOAL_REPLAY_INVALID",
            )
    elif operation == "pause":
        _require_same_definition(current, nxt, operation)
        if current.phase != "active" or nxt.phase != "paused":
            raise GoalError(
                "goal pause has an invalid phase transition", "GOAL_REPLAY_INVALID"
            )
    elif operation == "resume":
        _require_same_definition(current, nxt, operation)
        if (
            current.phase not in ("active", "paused", "blocked")
            or nxt.phase != "active"
            or state.rounds_started >= nxt.max_goal_rounds
        ):
            raise GoalError(
                "goal resume has an invalid phase transition or exhausted round budget",
                "GOAL_REPLAY_INVALID",
            )
    elif operation == "complete":
        _require_same_definition(current, nxt, operation)
        if current.phase == "complete" or nxt.phase != "complete":
            raise GoalError(
                "goal complete has an invalid phase transition", "GOAL_REPLAY_INVALID"
            )
    elif operation == "block":
        _require_same_definition(current, nxt, operation)
        if current.phase != "active" or nxt.phase != "blocked":
            raise GoalError(
                "goal block has an invalid phase transition", "GOAL_REPLAY_INVALID"
            )
    else:
        raise GoalError(
            f"goal {operation} cannot be validated as a current-goal transition",
            "GOAL_REPLAY_INVALID",
        )


def apply_goal_change(state: GoalFoldState, change: GoalChange) -> None:
    """Validate and apply one decoded change to a mutable accumulator."""
    ref = change_ref(change)
    if isinstance(change, GoalClearChange):
        current = state.goal
        if current is None:
            raise GoalError("goal clear requires a current goal", "GOAL_REPLAY_INVALID")
        _require_next_revision(current, change.cleared, "clear")
        if state.updated_at is None:
            raise GoalError("current goal fold lacks updatedAt", "GOAL_REPLAY_INVALID")
        if change.cleared_at < state.updated_at:
            raise GoalError(
                "goal clear timestamp cannot precede the current goal update",
                "GOAL_REPLAY_INVALID",
            )
        state.goal = None
        state.rounds_started = 0
        state.created_at = None
        state.updated_at = None
        state.last_ref = ref
        return
    if change.operation == "create":
        if (
            change.goal.revision != 1
            or change.goal.phase != "active"
            or change.rounds_started != 0
            or (state.goal is not None and state.goal.phase != "complete")
            or change.goal.id in state.seen_goal_ids
        ):
            raise GoalError(
                "goal create requires a fresh active revision-one goal with "
                "zero rounds",
                "GOAL_REPLAY_INVALID",
            )
        state.seen_goal_ids.add(change.goal.id)
    else:
        current = state.goal
        if current is None:
            raise GoalError(
                f"goal {change.operation} requires a current goal",
                "GOAL_REPLAY_INVALID",
            )
        _validate_snapshot_transition(state, change, current)
    state.goal = change.goal
    state.rounds_started = change.rounds_started
    state.created_at = change.created_at
    state.updated_at = change.updated_at
    state.last_ref = ref


def apply_goal_event(state: GoalFoldState, event: Any) -> None:
    """Apply one session event to the strict durable goal fold."""
    if event.type == "goal/change":
        change = decode_goal_change(event.payload)
        if change is None:
            raise GoalError(
                f"goal change at session event {event.seq} has an invalid kind",
                "GOAL_REPLAY_INVALID",
            )
        apply_goal_change(state, change)
        return
    if event.type != "user/message":
        return
    source = event.payload.get("source")
    if not isinstance(source, Mapping) or source.get("kind") != "goal":
        return
    goal_id = source.get("goalId")
    revision = source.get("revision")
    round_number = source.get("round")
    if (
        not isinstance(goal_id, str)
        or not goal_id
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or isinstance(round_number, bool)
        or not isinstance(round_number, int)
        or round_number < 1
    ):
        raise GoalError("goal message source is invalid", "GOAL_REPLAY_INVALID")
    current = state.goal
    if (
        current is None
        or current.phase != "active"
        or goal_id != current.id
        or revision != current.revision
        or round_number != state.rounds_started + 1
        or round_number > current.max_goal_rounds
    ):
        raise GoalError(
            f"goal round at session event {event.seq} is not the next admitted "
            "round of the active goal",
            "GOAL_REPLAY_INVALID",
        )
    state.rounds_started = round_number


def fold_goal(events: list[Any]) -> GoalFoldState:
    """Fold the current goal state from a contiguous session event log."""
    state = GoalFoldState()
    for event in events:
        apply_goal_event(state, event)
    return state


__all__ = [
    "GOAL_CHANGE_VERSION",
    "PHASES",
    "SNAPSHOT_OPERATIONS",
    "GoalActivation",
    "GoalBlockReason",
    "GoalChange",
    "GoalClearChange",
    "GoalError",
    "GoalFoldState",
    "GoalOperation",
    "GoalPhase",
    "GoalProjection",
    "GoalRef",
    "GoalSnapshot",
    "GoalSnapshotChange",
    "GoalView",
    "apply_goal_change",
    "apply_goal_event",
    "change_ref",
    "decode_goal_change",
    "fold_goal",
]
