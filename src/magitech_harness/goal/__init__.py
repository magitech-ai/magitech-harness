"""Same-session goal domain (``ctx.goals``) and its model-facing tools."""

from __future__ import annotations

from magitech_harness.goal.service import DEFAULT_MAX_GOAL_ROUNDS, GoalService
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

__all__ = [
    "DEFAULT_MAX_GOAL_ROUNDS",
    "GOAL_CHANGE_VERSION",
    "GoalActivation",
    "GoalBlockReason",
    "GoalError",
    "GoalPhase",
    "GoalProjection",
    "GoalRef",
    "GoalService",
    "GoalSnapshot",
    "GoalView",
]
