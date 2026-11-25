"""Workspace-rooted path confinement shared by the base filesystem tools.

Collapses ``..`` and resolves symlinks before checking containment against a
canonicalized workspace boundary. An escape raises :class:`PathEscapeError`;
tool handlers surface it as a model-facing error string.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PathEscapeError", "confined_path"]


class PathEscapeError(ValueError):
    """Raised when a requested path resolves outside the workspace root."""


def confined_path(root: Path | str, raw: str) -> Path:
    """Resolve *raw* under *root* and reject any path that escapes it."""
    base = Path(root).resolve(strict=True)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != base and base not in resolved.parents:
        raise PathEscapeError(f"path escapes workspace: {raw}")
    return resolved
