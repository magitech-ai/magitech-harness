"""Workspace instruction discovery and bounded UTF-8 reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from magitech_harness.agent_instructions.config import ResolvedConfig
from magitech_harness.agent_instructions.digest import (
    instruction_digest,
    trimmed_instruction_digest,
)

GLOBAL_SCOPE_USER = "user-global"
GLOBAL_SCOPE_AGENTS = "agents-global"


@dataclass(frozen=True)
class InstructionFile:
    """One discovered instruction file in loadable form."""

    path: Path
    display_path: str
    scope: str
    content: str
    digest: str
    trimmed_digest: str
    key: str = ""  # NUL-separated directory\0candidate scope key


@dataclass(frozen=True)
class LocatedFile:
    """A candidate location before its content is read."""

    path: Path
    display_path: str
    scope: str


def find_project_root(cwd: Path, markers: tuple[str, ...]) -> Path:
    """Walk upward from ``cwd`` until a root marker is found."""
    current = cwd.resolve()
    while True:
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            return cwd.resolve()
        current = parent


def ancestor_chain(root: Path, cwd: Path) -> list[Path]:
    """Inclusive root-to-cwd directory chain."""
    root, cwd = root.resolve(), cwd.resolve()
    chain: list[Path] = []
    current = cwd
    while True:
        chain.append(current)
        if current == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return list(reversed(chain))


def scope_for_display_path(display_path: str) -> str:
    """Map symbolic global labels to scope names, else the parent directory."""
    if display_path in (
        "~/.dsh/AGENTS.md",
        "$DSH_HOME/AGENTS.md",
        "~/.agents/AGENTS.md",
        "$DSH_AGENTS_HOME/AGENTS.md",
    ):
        return (
            GLOBAL_SCOPE_AGENTS
            if ".agents" in display_path or "DSH_AGENTS_HOME" in display_path
            else GLOBAL_SCOPE_USER
        )
    parent = str(Path(display_path).parent)
    return "." if parent == "." else parent


def instruction_scope_key(directory: str, candidate: str) -> str:
    """NUL-separated logical scope key for one candidate in one directory."""
    return f"{directory}\x00{candidate}"


def _read_bounded(path: Path, max_bytes: int) -> str | None:
    """Read a UTF-8 file without exceeding the byte cap; None when skipped."""
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file() or stat.st_size > max_bytes:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    return data.decode("utf-8", errors="replace")


def discover_locations(config: ResolvedConfig, cwd: Path) -> list[LocatedFile]:
    """Enumerate global and project candidate locations in precedence order."""
    locations: list[LocatedFile] = []
    roots = [
        (config.dsh_home / "AGENTS.md", "~/.dsh/AGENTS.md", GLOBAL_SCOPE_USER),
        (config.agents_home / "AGENTS.md", "~/.agents/AGENTS.md", GLOBAL_SCOPE_AGENTS),
    ]
    for path, display, scope in roots:
        locations.append(LocatedFile(path, display, scope))

    project_root = find_project_root(cwd, config.project_root_markers)
    for directory in ancestor_chain(project_root, cwd):
        for candidate in config.instruction_file_candidates:
            path = directory / candidate
            relative = path.relative_to(project_root).as_posix()
            locations.append(
                LocatedFile(path, relative, scope_for_display_path(relative))
            )
        for candidate in config.local_instruction_file_candidates:
            path = directory / candidate
            relative = path.relative_to(project_root).as_posix()
            locations.append(
                LocatedFile(path, relative, scope_for_display_path(relative))
            )
    return locations


def load_instruction_files(config: ResolvedConfig, cwd: Path) -> list[InstructionFile]:
    """Discover, read, validate and per-group deduplicate instruction files."""
    if config.max_bytes <= 0:
        return []
    seen_trimmed: dict[str, str] = {}
    seen_paths: set[Path] = set()
    result: list[InstructionFile] = []
    seen_keys: set[str] = set()
    for located in discover_locations(config, cwd):
        try:
            resolved = located.path.resolve()
        except OSError:
            continue
        if resolved in seen_paths:
            continue
        content = _read_bounded(resolved, config.max_source_bytes)
        if content is None:
            continue
        trimmed = trimmed_instruction_digest(content)
        group = (
            GLOBAL_SCOPE_USER
            if located.scope in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS)
            else str(Path(located.display_path).parent)
        )
        dedup_key = f"{group}\x00{trimmed}"
        if dedup_key in seen_trimmed:
            continue
        candidate = located.path.name
        directory = (
            located.display_path
            if located.scope in (GLOBAL_SCOPE_USER, GLOBAL_SCOPE_AGENTS)
            else str(Path(located.display_path).parent)
        )
        candidate_key = instruction_scope_key(directory, candidate)
        if candidate_key in seen_keys:
            continue
        seen_trimmed[dedup_key] = located.display_path
        seen_keys.add(candidate_key)
        seen_paths.add(resolved)
        result.append(
            InstructionFile(
                path=resolved,
                display_path=located.display_path,
                scope=located.scope,
                content=content,
                digest=instruction_digest(content),
                trimmed_digest=trimmed,
                key=candidate_key,
            )
        )
    return result


__all__ = [
    "GLOBAL_SCOPE_AGENTS",
    "GLOBAL_SCOPE_USER",
    "InstructionFile",
    "LocatedFile",
    "ancestor_chain",
    "discover_locations",
    "find_project_root",
    "instruction_scope_key",
    "load_instruction_files",
    "scope_for_display_path",
]
