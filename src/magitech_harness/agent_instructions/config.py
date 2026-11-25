"""Configuration normalization for workspace instruction discovery."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROJECT_ROOT_MARKERS = (".git",)
DEFAULT_INSTRUCTION_FILE_CANDIDATES = ("AGENTS.md", "CLAUDE.md")
DEFAULT_LOCAL_INSTRUCTION_FILE_CANDIDATES = ("AGENTS.local.md", "CLAUDE.local.md")
DEFAULT_MAX_SOURCE_BYTES = 1_048_576
RESERVED_SEGMENTS = frozenset({"", ".", ".."})


class AgentInstructionsError(ValueError):
    """Raised when the row configuration is invalid."""


@dataclass(frozen=True)
class ResolvedConfig:
    dsh_home: Path
    agents_home: Path
    project_root_markers: tuple[str, ...]
    instruction_file_candidates: tuple[str, ...]
    local_instruction_file_candidates: tuple[str, ...]
    max_bytes: int
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES

    def identity(self, cwd: Path, project_root: Path) -> str:
        """Stable serialized identity used to gate baseline reuse on resume."""
        try:
            relative = str(project_root.relative_to(cwd))
        except ValueError:
            relative = "."
        return json.dumps(
            {
                "projectRoot": relative,
                "projectRootMarkers": list(self.project_root_markers),
                "maxBytes": self.max_bytes,
                "maxSourceBytes": self.max_source_bytes,
                "instructionFileCandidates": list(self.instruction_file_candidates),
                "localInstructionFileCandidates": list(
                    self.local_instruction_file_candidates
                ),
            }
        )


def _home(env_name: str, default_suffix: str) -> Path:
    raw = os.environ.get(env_name)
    if raw and raw.strip():
        value = raw.strip()
    else:
        value = str(Path.home() / default_suffix)
    if value.startswith("~"):
        value = str(Path.home()) + value[1:]
    return Path(value).expanduser().resolve()


def _candidates(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AgentInstructionsError(
            "instruction file candidates must be an array of strings"
        )
    return tuple(
        item
        for item in value
        if item not in RESERVED_SEGMENTS and "/" not in item and "\\" not in item
    )


def resolve_config(config: dict[str, Any] | None = None) -> ResolvedConfig:
    """Validate and normalize the row configuration (homes are process policy)."""
    cfg = dict(config or {})
    max_bytes = cfg.get("maxBytes")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise AgentInstructionsError("maxBytes must be an integer")
    markers_value = cfg.get("projectRootMarkers")
    if markers_value is None:
        markers = DEFAULT_PROJECT_ROOT_MARKERS
    elif isinstance(markers_value, list) and all(
        isinstance(item, str) for item in markers_value
    ):
        markers = tuple(markers_value)
    else:
        raise AgentInstructionsError("projectRootMarkers must be an array of strings")
    max_source = cfg.get("maxSourceBytes", DEFAULT_MAX_SOURCE_BYTES)
    if (
        isinstance(max_source, bool)
        or not isinstance(max_source, int)
        or max_source < 1
    ):
        raise AgentInstructionsError("maxSourceBytes must be a positive integer")
    return ResolvedConfig(
        dsh_home=_home("DSH_HOME", ".dsh"),
        agents_home=_home("DSH_AGENTS_HOME", ".agents"),
        project_root_markers=markers,
        instruction_file_candidates=_candidates(
            cfg.get("instructionFileCandidates"), DEFAULT_INSTRUCTION_FILE_CANDIDATES
        ),
        local_instruction_file_candidates=_candidates(
            cfg.get("localInstructionFileCandidates"),
            DEFAULT_LOCAL_INSTRUCTION_FILE_CANDIDATES,
        ),
        max_bytes=max_bytes,
        max_source_bytes=max_source,
    )


__all__ = [
    "DEFAULT_INSTRUCTION_FILE_CANDIDATES",
    "DEFAULT_LOCAL_INSTRUCTION_FILE_CANDIDATES",
    "DEFAULT_MAX_SOURCE_BYTES",
    "DEFAULT_PROJECT_ROOT_MARKERS",
    "AgentInstructionsError",
    "ResolvedConfig",
    "resolve_config",
]
