"""Same-session subagent domain and its model-facing tools."""

from __future__ import annotations

from magitech_harness.subagent.manager import SubagentManager
from magitech_harness.subagent.service import (
    ROOT_AGENT_NAME,
    SubagentError,
    SubagentProvider,
    SubagentService,
)
from magitech_harness.subagent.tools import apply
from magitech_harness.subagent.types import (
    RESEARCH_PRESET_TOOLS,
    SubagentRecord,
    SubagentSpec,
    SubagentState,
)

__all__ = [
    "RESEARCH_PRESET_TOOLS",
    "ROOT_AGENT_NAME",
    "SubagentError",
    "SubagentManager",
    "SubagentProvider",
    "SubagentRecord",
    "SubagentService",
    "SubagentSpec",
    "SubagentState",
    "apply",
]
