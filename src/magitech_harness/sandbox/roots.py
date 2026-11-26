"""Canonical writable-root derivation for the sandbox seam."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from magitech_harness.sandbox.types import SandboxInvalidRootError


def canonical_path(path: str) -> str:
    """Native realpath, falling back to the spelled path when unreadable."""
    try:
        return str(Path(path).resolve(strict=False))
    except OSError:
        return path


def additional_root(path: str) -> str:
    """Validate and canonicalize one extra writable directory (every call)."""
    physical = canonical_path(path)
    try:
        mode = os.stat(physical).st_mode
    except OSError as error:
        raise SandboxInvalidRootError(
            path, f"cannot read the directory: {error}"
        ) from error
    if not stat.S_ISDIR(mode):
        raise SandboxInvalidRootError(path, "expected an existing directory")
    if os.path.normpath(path) != os.path.normpath(physical):
        raise SandboxInvalidRootError(
            path,
            f'use the physical directory path "{physical}"; symlinked grants are not supported',
        )
    return physical


def workspace_writable_roots(policy: Mapping[str, Any]) -> list[str]:
    """Writable roots for ``workspace-write`` only, deduped after canonicalization."""
    if policy.get("mode") != "workspace-write":
        return []
    roots = [canonical_path(str(policy["workspaceRoot"]))]
    for extra in policy.get("additionalWritableRoots") or ():
        roots.append(additional_root(str(extra)))
    return list(dict.fromkeys(roots))


def writable_roots(policy: Mapping[str, Any]) -> list[str]:
    """Workspace roots plus the platform temp areas."""
    if policy.get("mode") != "workspace-write":
        return []
    roots = workspace_writable_roots(policy)
    for candidate in ("/tmp", tempfile.gettempdir()):
        roots.append(canonical_path(candidate))
    return list(dict.fromkeys(roots))


__all__ = [
    "additional_root",
    "canonical_path",
    "workspace_writable_roots",
    "writable_roots",
]
