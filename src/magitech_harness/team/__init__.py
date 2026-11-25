"""Agent Teams: shared task boards and model-facing task tools."""

from __future__ import annotations

from magitech_harness.team.service import DEFAULT_MAX_TASKS, TeamService, apply
from magitech_harness.team.types import TeamError

__all__ = ["DEFAULT_MAX_TASKS", "TeamError", "TeamService", "apply"]
