"""Types and data structures for subagent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

RESEARCH_PRESET_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "glob",
        "grep",
        "lsp_definition",
        "lsp_references",
    }
)


class SubagentState(StrEnum):
    """Execution lifecycle state of a delegated subagent."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERRORED = "errored"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class SubagentSpec:
    """Specification for spawning a child subagent."""

    role: str
    prompt: str
    agent_name: str | None = None
    model: str = "inherit"
    workspace: Literal["inherit", "branch"] = "inherit"
    allowed_tools: frozenset[str] | None = None
    wait: bool = True
    max_depth: int | None = None


@dataclass
class SubagentRecord:
    """Runtime tracking record for a spawned subagent."""

    subagent_id: str
    role: str
    agent_name: str
    model_id: str
    workspace_path: Path
    session_id: str
    session_path: Path
    state: SubagentState
    state_detail: str = ""
    summary: str | None = None
    error: str | None = None
    effective_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Converts record to model-facing dictionary format."""
        return {
            "subagent_id": self.subagent_id,
            "role": self.role,
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "workspace_path": str(self.workspace_path),
            "session_id": self.session_id,
            "session_path": str(self.session_path),
            "status": self.state.value,
            "state_detail": self.state_detail,
            "summary": self.summary,
            "error": self.error,
            "effective_tools": self.effective_tools,
        }
