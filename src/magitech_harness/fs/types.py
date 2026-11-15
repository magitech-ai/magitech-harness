"""Vocabulary for the filesystem service seam (ctx.fs)."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from magitech_harness.llm.error import HarnessError

FsTargetKey = str
FsVersion = str

FS_ERROR_CODES: tuple[str, ...] = (
    "FS_NOT_FOUND",
    "FS_NOT_DIRECTORY",
    "FS_NOT_TEXT",
    "FS_NOT_REGULAR_FILE",
    "FS_TOO_LARGE",
    "FS_PERMISSION_DENIED",
    "FS_SANDBOX_DENIED",
    "FS_IO_ERROR",
    "FS_STALE_VERSION",
    "FS_NOT_OBSERVED",
    "FS_AMBIGUOUS_EDIT",
    "FS_EDIT_NOT_FOUND",
    "FS_ABORTED",
    "FS_PATCH_MISMATCH",
    "FS_SAME_TARGET",
)

FsErrorCode = Literal[
    "FS_NOT_FOUND",
    "FS_NOT_DIRECTORY",
    "FS_NOT_TEXT",
    "FS_NOT_REGULAR_FILE",
    "FS_TOO_LARGE",
    "FS_PERMISSION_DENIED",
    "FS_SANDBOX_DENIED",
    "FS_IO_ERROR",
    "FS_STALE_VERSION",
    "FS_NOT_OBSERVED",
    "FS_AMBIGUOUS_EDIT",
    "FS_EDIT_NOT_FOUND",
    "FS_ABORTED",
    "FS_PATCH_MISMATCH",
    "FS_SAME_TARGET",
]


class FsError(HarnessError):
    """Typed filesystem error carrying a stable FsErrorCode and optional cause."""

    code: FsErrorCode

    def __init__(
        self,
        message: str,
        code: FsErrorCode,
        *,
        cause: BaseException | object | None = None,
    ) -> None:
        super().__init__(message, code, cause=cause)
        self.code = code


class FsObservationPresent(TypedDict):
    """Positive authoritative observation carrying the observed version token."""

    kind: Literal["present"]
    version: FsVersion


class FsObservationAbsent(TypedDict):
    """Negative authoritative observation confirming the target is absent."""

    kind: Literal["absent"]


FsObservation = FsObservationPresent | FsObservationAbsent


class FsTarget(TypedDict):
    """Resolved backend target identity plus caller-facing display path."""

    targetKey: FsTargetKey
    displayPath: str


class FsInfo(TypedDict):
    """Metadata returned by FileSystem.stat."""

    version: FsVersion
    type: Literal["file", "directory", "other"]
    size: NotRequired[int]


class FsPathInfo(TypedDict):
    """Path metadata returned by FileSystem.lstat without following symlinks."""

    version: FsVersion
    type: Literal["file", "directory", "symlink", "other"]
    size: NotRequired[int]


class FsDirEntry(TypedDict):
    """One direct child entry returned by FileSystem.list_dir."""

    name: str
    type: Literal["file", "directory", "other"]
    target: FsTarget
    version: NotRequired[FsVersion]
    size: NotRequired[int]


class FsWriteIntentCreateIfAbsent(TypedDict):
    """Guarded write intent rejecting existing targets."""

    kind: Literal["createIfAbsent"]


class FsWriteIntentReplaceIfVersion(TypedDict):
    """Guarded write intent requiring exact target version."""

    kind: Literal["replaceIfVersion"]
    version: FsVersion


FsWriteIntent = FsWriteIntentCreateIfAbsent | FsWriteIntentReplaceIfVersion


class FsWriteOutcome(TypedDict):
    """Outcome of a full-file write with bounded LF-normalized diff basis."""

    operation: Literal["create", "update"]
    version: FsVersion
    before: str | None
    after: str


class FsEditRequest(TypedDict):
    """Literal replacement edit request."""

    oldString: str
    newString: str
    replaceAll: bool


class FsEditOutcome(TypedDict):
    """Outcome of a literal edit or patch with LF-normalized before/after text."""

    version: FsVersion
    before: str
    after: str


class FsDeleteOutcome(TypedDict):
    """Outcome of a guarded regular-file deletion."""

    before: str | None


class FsMoveSourceFailure(TypedDict):
    """Failure details when destination published but source deletion failed."""

    code: FsErrorCode
    message: str


class FsMoveOutcomeDeleted(TypedDict):
    """Move outcome when source removal succeeded."""

    sourceBefore: str | None
    destination: FsWriteOutcome
    sourceDeleted: Literal[True]


class FsMoveOutcomeRetained(TypedDict):
    """Move outcome when destination published but source remained."""

    sourceBefore: str | None
    destination: FsWriteOutcome
    sourceDeleted: Literal[False]
    sourceFailure: FsMoveSourceFailure


FsMoveOutcome = FsMoveOutcomeDeleted | FsMoveOutcomeRetained
