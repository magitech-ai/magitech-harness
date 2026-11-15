"""Cross-platform workspace path classification, resolution, and display helpers."""

from __future__ import annotations

import re
from typing import TypedDict

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")
_LEADING_REL_RE = re.compile(r"^(?:\.?[/\\])+")


class PathParts(TypedDict):
    """Directory prefix and final basename split for display."""

    directory: str
    name: str


def _is_windows_style_path(value: str) -> bool:
    return bool(_WIN_DRIVE_RE.match(value)) or value.startswith("\\\\")


def is_absolute_workspace_path(path: str) -> bool:
    """Return True for POSIX root, Windows drive, or UNC absolute paths."""
    return path.startswith("/") or _is_windows_style_path(path)


def resolve_workspace_path(cwd: str | None, path: str) -> str:
    """Resolve *path* relative to *cwd* using the host's native separator spelling."""
    if is_absolute_workspace_path(path):
        return path
    if not cwd:
        return path
    separator = "\\" if _is_windows_style_path(cwd) and "\\" in cwd else "/"
    base = cwd.rstrip("/\\")
    relative = _LEADING_REL_RE.sub("", path)
    return f"{base}{separator}{relative}"


def abbreviate_home_path(path: str, home: str | None = None) -> str:
    """Abbreviate a POSIX home directory prefix to ``~`` or ``~/...``."""
    if not home:
        return path
    if _is_windows_style_path(path) or _is_windows_style_path(home):
        return path
    root = home.rstrip("/")
    if not root or root == "/":
        return path
    if path.rstrip("/") == root:
        return "~"
    if path.startswith(f"{root}/"):
        return f"~{path[len(root) :]}"
    return path


def workspace_title_of(path: str) -> str:
    """Return the final non-empty segment of a workspace path."""
    trimmed = path.rstrip("/\\")
    separator = max(trimmed.rfind("/"), trimmed.rfind("\\"))
    return trimmed[separator + 1 :]


def path_parts_of(path: str) -> PathParts:
    """Split a path into its directory prefix through the last separator and basename."""
    trimmed = path.rstrip("/\\")
    if trimmed == "":
        return {"directory": "", "name": path}
    cut = max(trimmed.rfind("/"), trimmed.rfind("\\")) + 1
    return {"directory": trimmed[:cut], "name": trimmed[cut:]}


def file_address_for(session_id: str, cwd: str | None, path: str) -> str:
    """Format a canonical session file resource URI for *path*."""
    normalized = path.replace("\\", "/")
    if not is_absolute_workspace_path(normalized):
        return f"mth-resource://file/{session_id}/{normalized}"
    root = "" if cwd is None else cwd.replace("\\", "/").rstrip("/")
    if root and normalized == root:
        return f"mth-resource://file/{session_id}/"
    if root and normalized.startswith(f"{root}/"):
        return f"mth-resource://file/{session_id}/{normalized[len(root) + 1 :]}"
    return f"mth-resource://file/{session_id}/{normalized}"


__all__ = [
    "PathParts",
    "abbreviate_home_path",
    "file_address_for",
    "is_absolute_workspace_path",
    "path_parts_of",
    "resolve_workspace_path",
    "workspace_title_of",
]
