"""Filesystem service definitions, local implementation, patching, and policy."""

from __future__ import annotations

from magitech_harness.fs.local import LocalFileSystem, expand_home_prefix
from magitech_harness.fs.observation_policy import ObservedStateGate
from magitech_harness.fs.observation_policy import apply as apply_observation_policy
from magitech_harness.fs.patch import (
    ApplyHunksResult,
    HunkApplyFailure,
    PatchHunk,
    PatchOperation,
    PatchParseError,
    apply_hunks,
    applyHunks,
    parse_patch,
    parsePatch,
)
from magitech_harness.fs.service import FileSystem
from magitech_harness.fs.types import (
    FS_ERROR_CODES,
    FsDeleteOutcome,
    FsDirEntry,
    FsEditOutcome,
    FsEditRequest,
    FsError,
    FsErrorCode,
    FsInfo,
    FsMoveOutcome,
    FsMoveSourceFailure,
    FsObservation,
    FsPathInfo,
    FsTarget,
    FsTargetKey,
    FsVersion,
    FsWriteIntent,
    FsWriteOutcome,
)
from magitech_harness.fs.workspace_path import (
    PathParts,
    abbreviate_home_path,
    file_address_for,
    is_absolute_workspace_path,
    path_parts_of,
    resolve_workspace_path,
    workspace_title_of,
)

__all__ = [
    "FS_ERROR_CODES",
    "ApplyHunksResult",
    "FileSystem",
    "FsDeleteOutcome",
    "FsDirEntry",
    "FsEditOutcome",
    "FsEditRequest",
    "FsError",
    "FsErrorCode",
    "FsInfo",
    "FsMoveOutcome",
    "FsMoveSourceFailure",
    "FsObservation",
    "FsPathInfo",
    "FsTarget",
    "FsTargetKey",
    "FsVersion",
    "FsWriteIntent",
    "FsWriteOutcome",
    "HunkApplyFailure",
    "LocalFileSystem",
    "ObservedStateGate",
    "PatchHunk",
    "PatchOperation",
    "PatchParseError",
    "PathParts",
    "abbreviate_home_path",
    "applyHunks",
    "apply_hunks",
    "apply_observation_policy",
    "expand_home_prefix",
    "file_address_for",
    "is_absolute_workspace_path",
    "parsePatch",
    "parse_patch",
    "path_parts_of",
    "resolve_workspace_path",
    "workspace_title_of",
]
