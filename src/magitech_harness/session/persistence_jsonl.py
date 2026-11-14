"""Plain-JSONL session persistence: layout, generation resolution, kernel lease.

One session lives under ``<root>/<project-dir>/<session-dir>/``; the
project dir is ``_no-cwd`` when the session has no working directory and
``--<slug>--`` otherwise, and the session dir is the encoded session id.
Generations are ``session.jsonl`` (v0) and ``session.vN.jsonl`` (v1+);
the current generation is the highest version present, found by listing
the directory — there is no separate index file.

Artifacts are plain JSONL: the only behavioral differences vs the
checksummed framing are the suffix, no checksum protection, and no frame
boundaries, so torn-tail detection rests on the JSONL newline/record
rules.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from anima import Context, Disposer, Service

from magitech_harness.session.events import (
    KNOWN_SESSION_EVENT_TYPES,
    SessionEvent,
    snapshot_json_value,
)
from magitech_harness.session.format import (
    CURRENT_FORMAT_VERSION,
    SESSION_FORMAT_CATALOG,
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatUnsupportedError,
    SessionFormatUnsupportedMigrationError,
    SessionPersistenceCorruptionError,
    assert_v4_row_admission,
    create_session_format_catalog_with_children,
    decode_envelope_row,
    parse_session_format_log_filename,
    session_format_log_filename,
    snapshot_header,
)
from magitech_harness.session.persistence import (
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
    SessionHandle,
    SessionHandleClosedError,
    SessionOwnershipLostError,
    SessionPersistence,
    SessionPersistenceNotFoundError,
    SessionPersistenceSnapshot,
    SessionReadOnlyError,
)
from magitech_harness.session.store import (
    _classify_header_line,
    _inherited_event_count,
    _parse_row,
    _to_session_event,
)
from magitech_harness.session.surface import validate_session_event_data

__all__ = [
    "JsonlLiveTracker",
    "JsonlSessionHandle",
    "JsonlSessionPersistence",
    "JsonlSessionPersistenceService",
    "KernelLease",
    "ReadResult",
    "ScanResult",
    "SessionAlreadyExistsError",
    "SessionAlreadyOwnedError",
    "SessionFormatUnsupportedError",
    "SessionHandleClosedError",
    "SessionOwnershipLostError",
    "SessionPersistenceCorruptionError",
    "SessionPersistenceNotFoundError",
    "SessionPersistenceSnapshot",
    "SessionReadOnlyError",
    "encode_segment",
    "find_session_dir",
    "generation_filename",
    "parse_generation_filename",
    "project_dir",
    "project_key",
    "read_generation",
    "resolve_current_generation",
    "scan_log_bytes",
    "session_dir",
    "verify_current_generation",
]

_LOCK_FILENAME = "session.lock"
"""Kernel lease file beside the log; never removed on release."""

logger = logging.getLogger("harness.session")
"""Module logger: the disposed-drain warning goes here, never raised."""

_MAX_LOCK_ATTEMPTS = 3
"""Bounded retries verifying the locked inode is still the file at the path."""

_PROJECT_KEY_MAX = 251
"""Bound on the project slug inside its ``--`` markers."""

# ---------------------------------------------------------------------------
# Path encoding
# ---------------------------------------------------------------------------

_SAFE_SEGMENT_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
)
"""Code units that stay literal in an encoded path segment."""


def _utf16_units(value: str) -> list[int]:
    """Split a string into its UTF-16 code units, lone surrogates included."""
    encoded = value.encode("utf-16-be", "surrogatepass")
    return [
        int.from_bytes(encoded[index : index + 2], "big")
        for index in range(0, len(encoded), 2)
    ]


def _escape_unit(code: int) -> str:
    """One ``~XXXX`` escape for a single UTF-16 code unit."""
    return f"~{code:04X}"


def encode_segment(raw: str) -> str:
    """Encode an arbitrary string as one safe path segment, injectively.

    Safe code units stay literal; every other unit, including ``~``,
    becomes a ``~XXXX`` escape. ``.`` and ``..`` are special-cased so an
    encoded segment can never traverse. Operating on UTF-16 code units
    keeps lone surrogates encodable.
    """
    if not raw:
        raise ValueError("cannot encode an empty path segment")
    if raw == ".":
        return "~002E"
    if raw == "..":
        return "~002E~002E"
    return "".join(
        chr(code)
        if chr(code) != "~" and chr(code) in _SAFE_SEGMENT_CHARS
        else _escape_unit(code)
        for code in _utf16_units(raw)
    )


def project_key(cwd: str) -> str:
    """Build the readable directory key for a project path.

    Separators and drive separators collapse to single ``-`` runs;
    unsafe code units use the same ``~XXXX`` escape as session ids.
    The slug is truncated to 251 chars; separator replacement and
    truncation are intentionally lossy.
    """
    if not cwd:
        raise ValueError("cannot encode an empty project path")
    readable = ""
    separator_run = False
    for code in _utf16_units(cwd):
        char = chr(code)
        if char in ("/", "\\", ":"):
            if not separator_run:
                readable += "-"
            separator_run = True
        elif char != "~" and char in _SAFE_SEGMENT_CHARS:
            readable += char
            separator_run = False
        else:
            readable += _escape_unit(code)
            separator_run = False
    slug = readable.lstrip("-") or "root"
    return f"--{slug[:_PROJECT_KEY_MAX]}--"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def project_dir(root: Path, cwd: str | None) -> Path:
    """The project directory under ``root``; ``_no-cwd`` when no cwd."""
    if cwd is None:
        return root / "_no-cwd"
    return root / project_key(cwd)


def session_dir(root: Path, cwd: str | None, session_id: str) -> Path:
    """The directory owned by one session under its project directory."""
    return project_dir(root, cwd) / encode_segment(session_id)


def find_session_dir(root: Path, session_id: str) -> Path | None:
    """Locate a session directory by scanning ``<root>/*/<encoded id>/``.

    The caller does not know the session's working directory, so every
    project directory is checked. Returns ``None`` when no match exists.
    """
    if not root.is_dir():
        return None
    segment = encode_segment(session_id)
    for project in root.iterdir():
        if not project.is_dir():
            continue
        candidate = project / segment
        if candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Generations
# ---------------------------------------------------------------------------


def generation_filename(version: int) -> str:
    """Canonical filename for one format generation in a session directory."""
    return session_format_log_filename(version)


def parse_generation_filename(name: str) -> int | None:
    """Read the generation named by one file, or ``None`` if noncanonical.

    Temporary, uppercase, leading-zero, and ``.v0``-tagged names do not
    identify committed generations.
    """
    return parse_session_format_log_filename(name)


def resolve_current_generation(session_dir: Path) -> tuple[int, Path] | None:
    """Resolve the current generation in a session directory.

    The highest version present wins; noncanonical names are ignored.
    A compressed artifact is a hard error. Returns ``None`` when the
    directory holds no committed generation.
    """
    if not session_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for entry in session_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name.endswith(".jsonl.zstd"):
            raise SessionPersistenceCorruptionError(
                f"session directory {session_dir} holds a compressed artifact "
                f"{entry} this build cannot read"
            )
        version = parse_generation_filename(entry.name)
        if version is None:
            continue
        candidates.append((version, entry))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


# ---------------------------------------------------------------------------
# Kernel lease
# ---------------------------------------------------------------------------


def _same_inode(fd: int, lock_path: Path) -> bool:
    """Whether the locked file descriptor is still the file at the path."""
    try:
        current = os.stat(lock_path)
    except OSError:
        return False
    locked = os.fstat(fd)
    return (locked.st_dev, locked.st_ino) == (current.st_dev, current.st_ino)


def _acquire_posix(lock_path: Path, session_id: str) -> int:
    """Take a non-blocking exclusive flock, verifying the locked inode."""
    if sys.platform == "win32":
        # The flock calls below are POSIX-only; the guard also tells the
        # type checker this body is unreachable on Windows.
        raise RuntimeError("POSIX lease requested on Windows")
    import fcntl

    for _attempt in range(_MAX_LOCK_ATTEMPTS):
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            if error.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise SessionAlreadyOwnedError(session_id) from error
            raise
        if _same_inode(fd, lock_path):
            return fd
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    raise SessionAlreadyOwnedError(session_id)


def _acquire_windows(
    lock_path: Path,
    locking: Callable[[int, int, int], None],
    lock_mode: int,
    session_id: str,
) -> int:
    """Take a non-blocking exclusive byte-range lock on byte 0."""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        locking(fd, lock_mode, 1)
    except OSError as error:
        os.close(fd)
        if error.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
            raise SessionAlreadyOwnedError(session_id) from error
        raise
    return fd


class KernelLease:
    """Cross-process single-writer exclusion on a session lock file.

    The lock file sits beside the log and is never removed on release,
    so its inode stays stable for later lockers. Readers never touch
    it. Release is idempotent.
    """

    def __init__(self, fd: int, lock_path: Path) -> None:
        self._fd: int | None = fd
        self._lock_path = lock_path

    @classmethod
    def acquire(cls, session_dir: Path, session_id: str) -> KernelLease:
        """Create the lock file if needed and take the exclusive lease."""
        lock_path = session_dir / _LOCK_FILENAME
        if sys.platform == "win32":
            import msvcrt

            fd = _acquire_windows(
                lock_path, msvcrt.locking, msvcrt.LK_NBLCK, session_id
            )
        else:
            fd = _acquire_posix(lock_path, session_id)
        return cls(fd, lock_path)

    def release(self) -> None:
        """Release the lease; the lock file stays in place. Idempotent."""
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Byte-level log scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanResult:
    """Split of raw log bytes into complete records and a torn tail.

    ``lines`` holds every newline-terminated record with the newline
    stripped. A final record without a trailing newline is never a
    complete record, even when it parses: it is the torn fragment.
    """

    lines: list[bytes]
    torn_truncate_to: int | None
    torn_fragment: bytes


def scan_log_bytes(data: bytes) -> ScanResult:
    """Split raw log bytes into complete records plus a torn tail, if any.

    Empty input has no lines and no torn fragment; a file ending exactly
    with a newline has no torn fragment either.
    """
    if not data:
        return ScanResult(lines=[], torn_truncate_to=None, torn_fragment=b"")
    parts = data.split(b"\n")
    if data.endswith(b"\n"):
        return ScanResult(lines=parts[:-1], torn_truncate_to=None, torn_fragment=b"")
    torn_fragment = parts[-1]
    return ScanResult(
        lines=parts[:-1],
        torn_truncate_to=len(data) - len(torn_fragment),
        torn_fragment=torn_fragment,
    )


# ---------------------------------------------------------------------------
# Generation read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadResult:
    """One decoded generation: logical header, events, and file identity.

    ``revision_base`` is the file identity at read time — ``(st_dev,
    st_ino, st_size, st_mtime_ns, st_ctime_ns)`` from one ``os.stat``
    taken before the bytes are read — so a later read can tell whether
    the file was replaced. It identifies the file, not its contents.
    ``recovered_tail`` stays empty for plain JSONL: a torn fragment
    holds no complete event, so there is nothing to recover from it.
    """

    header: dict[str, Any]
    events: list[SessionEvent]
    inherited_event_count: int
    format_status: Literal["current", "migration-required"]
    torn_truncate_to: int | None
    recovered_tail: list[SessionEvent]
    revision_base: tuple[int, int, int, int, int]


def _revision_base(path: Path) -> tuple[int, int, int, int, int]:
    """File identity for the revision token, from one stat call."""
    info = os.stat(path)
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _decode_line(line: bytes, line_number: int, path: Path) -> str:
    """Decode one scanned line as UTF-8; undecodable bytes are corruption."""
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SessionPersistenceCorruptionError(
            f"corrupt session row {line_number} in {path}: not valid UTF-8"
        ) from error


def _read_header_line(
    line: bytes, path: Path
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Classify the first complete line; the version gate runs first.

    Returns the physical header value, the logical header, and whether
    the log is already current-version. A header that is not valid
    UTF-8 or JSON is corruption, never an unsupported version: the gate
    needs a parsed header value, so it cannot run before parsing.
    """
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SessionPersistenceCorruptionError(
            f"corrupt session header in {path}: not valid UTF-8"
        ) from error
    return _classify_header_line(text, path)


def _record_issue(
    issue: BaseException,
    recovery: Literal["strict", "recoverable"],
    recorded: BaseException | None,
) -> BaseException:
    """Raise a bad-row issue now in strict mode; keep the first in recoverable mode."""
    if recovery == "strict":
        raise issue
    if recorded is not None:
        return recorded
    return issue


def _read_current_rows(
    lines: list[bytes],
    session_id: str,
    path: Path,
    recovery: Literal["strict", "recoverable"],
) -> list[SessionEvent]:
    """Envelope-decode current-version body rows with recoverable tail rules.

    Row admission runs before recovery can discard a row, so a refused
    row fails loud in both modes — even a row after an already recorded
    issue. The first other bad row is recorded in recoverable mode;
    later rows are suppressed until a ``turn/end`` row re-throws the
    recorded issue, or the read ends and raises it there.
    """
    events: list[SessionEvent] = []
    recorded: BaseException | None = None
    for body_index, raw in enumerate(lines[1:], start=1):
        try:
            row = _parse_row(_decode_line(raw, body_index, path), body_index, path)
        except SessionPersistenceCorruptionError as error:
            recorded = _record_issue(error, recovery, recorded)
            continue
        try:
            assert_v4_row_admission(row, KNOWN_SESSION_EVENT_TYPES)
        except SessionFormatUnsupportedMigrationError as error:
            raise SessionFormatUnsupportedError(str(error)) from error
        if recorded is not None:
            if row.get("type") == "turn/end":
                raise recorded
            continue
        try:
            decoded = decode_envelope_row(row, body_index, f"session {session_id}")
        except SessionFormatError as error:
            recorded = _record_issue(
                SessionPersistenceCorruptionError(str(error), error),
                recovery,
                recorded,
            )
            continue
        if decoded.get("seq") != len(events):
            recorded = _record_issue(
                SessionPersistenceCorruptionError(
                    f"corrupt session row {body_index} in {path}: seq gap "
                    f"(expected {len(events)}, got {decoded.get('seq')})"
                ),
                recovery,
                recorded,
            )
            continue
        try:
            event = _to_session_event(decoded, session_id, path)
        except SessionFormatError as error:
            recorded = _record_issue(
                SessionPersistenceCorruptionError(str(error), error),
                recovery,
                recorded,
            )
            continue
        events.append(event)
    if recorded is not None:
        raise recorded
    return events


def _run_migrated_restore(
    lines: list[bytes],
    header_value: dict[str, Any],
    path: Path,
    recovery: Literal["strict", "recoverable"],
) -> SessionFormatArtifact:
    """Decode historical rows through the catalog into a v4 artifact.

    The restore runs with the caller's recovery mode and owns the
    record/suppress/re-throw-at-``turn/end`` rules for rows it decodes.
    Rows that fail to parse never reach it: in strict mode they raise at
    once, in recoverable mode the first is recorded, later rows are
    suppressed, and the recorded issue is re-thrown at the next
    ``turn/end`` row or at end of read.
    """
    catalog = create_session_format_catalog_with_children([])
    restore = catalog.create_restore(
        header_value, validation="transformed", recovery=recovery
    )
    recorded: BaseException | None = None
    try:
        for body_index, raw in enumerate(lines[1:], start=1):
            try:
                row = _parse_row(_decode_line(raw, body_index, path), body_index, path)
            except SessionPersistenceCorruptionError as error:
                recorded = _record_issue(error, recovery, recorded)
                continue
            if recorded is not None:
                if row.get("type") == "turn/end":
                    raise recorded
                continue
            restore.decode_row(row)
        if recorded is not None:
            raise recorded
        return restore.finish()
    except SessionFormatUnsupportedMigrationError as error:
        raise SessionFormatUnsupportedError(f"{error} (raw log: {path})") from error
    except SessionFormatError as error:
        raise SessionPersistenceCorruptionError(str(error), error) from error


def _read_migrated_rows(
    lines: list[bytes],
    header_value: dict[str, Any],
    session_id: str,
    path: Path,
    recovery: Literal["strict", "recoverable"],
) -> tuple[list[SessionEvent], int]:
    """Restore a historical log through the catalog into v4 events.

    A recoverable restore records its first bad row and suppresses the
    rest, re-throwing only at a later ``turn/end``. When no ``turn/end``
    follows, the swallowed issue must still surface at end of read, so
    a clean recoverable finish is followed by a strict re-run over the
    same rows, which raises the first bad row instead of silently
    returning the preserved prefix.
    """
    artifact = _run_migrated_restore(lines, header_value, path, recovery)
    if recovery == "recoverable":
        _run_migrated_restore(lines, header_value, path, "strict")
    events = [_to_session_event(event, session_id, path) for event in artifact.events]
    return events, artifact.inherited_event_count


def read_generation(
    path: Path, *, recovery: Literal["strict", "recoverable"] = "recoverable"
) -> ReadResult:
    """Read one generation file: header-strict, body per ``recovery``.

    The header is classified first — the version gate runs before any
    structural check, so a newer version is refused as unsupported with
    the upgrade text, never as corruption. Body rows follow the
    recoverable tail rules (first bad row recorded, later rows
    suppressed, re-thrown at the next ``turn/end`` or at end of read)
    or raise at the first bad row in strict mode. ``revision_base`` is
    the file identity from one stat taken before the bytes are read.
    """
    revision_base = _revision_base(path)
    scan = scan_log_bytes(path.read_bytes())
    if not scan.lines:
        raise SessionPersistenceCorruptionError(
            f"corrupt session log: first line is not a session header in {path}"
        )
    header_value, header, is_current = _read_header_line(scan.lines[0], path)
    session_id = str(header["id"])
    format_status: Literal["current", "migration-required"] = (
        "current" if is_current else "migration-required"
    )
    if is_current:
        events = _read_current_rows(scan.lines, session_id, path, recovery)
        inherited = _inherited_event_count(events)
    else:
        events, inherited = _read_migrated_rows(
            scan.lines, header_value, session_id, path, recovery
        )
    return ReadResult(
        header=header,
        events=events,
        inherited_event_count=inherited,
        format_status=format_status,
        torn_truncate_to=scan.torn_truncate_to,
        recovered_tail=[],
        revision_base=revision_base,
    )


def verify_current_generation(path: Path) -> ReadResult:
    """Strictly read one generation, refusing a torn physical tail.

    The verification read migration publication uses: it repairs
    nothing, and a generation with a torn tail is refused outright,
    naming the torn offset.
    """
    result = read_generation(path, recovery="strict")
    if result.torn_truncate_to is not None:
        raise SessionPersistenceCorruptionError(
            "current session generation has a torn physical tail at byte "
            f"offset {result.torn_truncate_to} in {path}"
        )
    return result


# ---------------------------------------------------------------------------
# Session handle
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    """fsync a directory so a new file entry in it is durable (POSIX only)."""
    if sys.platform == "win32":
        return
    # POSIX-only body: on Windows this function returns early above, and a
    # directory cannot be opened with os.open there at all.
    fd = os.open(path, os.O_RDONLY)  # pragma: no cover
    try:  # pragma: no cover
        os.fsync(fd)  # pragma: no cover
    finally:  # pragma: no cover
        os.close(fd)  # pragma: no cover


def _ensure_session_dir(path: Path) -> None:
    """Create a session directory (and parents) with mode 0o700."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _encode_line(value: dict[str, Any]) -> bytes:
    """One JSON value as a newline-terminated line."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


class JsonlSessionHandle(SessionHandle):
    """One open session log: an in-memory log plus a durable JSONL file.

    Implements the frozen handle contract: ``read`` re-reads the
    materialized generation on every call and merges growth into the
    in-memory log (a stored log that shrank below a previously
    observed prefix is corruption); ``append`` validates and snapshots
    the batch before any mutation, repairs a torn tail first, then
    writes the batch durably; ``flush`` is the durability barrier;
    ``close`` is idempotent. All mutations serialize on one
    per-handle lock.
    """

    def __init__(
        self,
        *,
        session_id: str,
        header: dict[str, Any],
        inherited_event_count: int,
        access: Literal["read", "write"],
        generation_path: Path,
        lease: KernelLease | None,
        events: list[SessionEvent] | None = None,
        torn_truncate_to: int | None = None,
        recovered_tail: list[SessionEvent] | None = None,
        materialized: bool = False,
        header_written: bool = False,
        release_claim: Callable[[], None] | None = None,
    ) -> None:
        self._id = session_id
        self._header = header
        self._inherited_event_count = inherited_event_count
        self._access = access
        self._generation_path = generation_path
        self._lease = lease
        # The adopted log is a detached copy: the caller keeps its list.
        self._events: list[SessionEvent] = copy.deepcopy(list(events or []))
        self._recovered_tail: list[SessionEvent] = copy.deepcopy(
            list(recovered_tail or [])
        )
        self._torn_truncate_to = torn_truncate_to
        self._materialized = materialized
        self._header_written = header_written
        self._release_claim = release_claim
        self._lock = threading.Lock()
        self._closed = False
        # File-complete count observed at open: the adopted log holds
        # the recovered tail, which is not in the file yet.
        self._high_water = len(self._events) - len(self._recovered_tail)

    @property
    def id(self) -> str:
        """The session id this handle serves."""
        return self._id

    @property
    def header(self) -> dict[str, Any]:
        """The logical header, fixed at create/open."""
        return self._header

    @property
    def inherited_event_count(self) -> int:
        """The inherited cut the header was seeded with."""
        return self._inherited_event_count

    @property
    def access(self) -> Literal["read", "write"]:
        """Whether this handle may mutate the log."""
        return self._access

    @property
    def closed(self) -> bool:
        """Whether close has run on this handle."""
        return self._closed

    def _require_open(self, operation: str) -> None:
        """Refuse an operation on a closed handle."""
        if self._closed:
            raise SessionHandleClosedError(self._id, operation)

    def _require_write(self, operation: str) -> None:
        """Refuse a mutation on a read handle."""
        if self._access != "write":
            raise SessionReadOnlyError(self._id, operation)

    def read(
        self, offset: int = 0, length: int | None = None
    ) -> tuple[str, list[SessionEvent]]:
        """Serve a slice of the log, merging any growth of the stored file.

        A materialized handle re-reads the generation on every call; the
        stored log may only grow, so a file shorter than the previously
        observed prefix is corruption. An unmaterialized handle serves
        its in-memory log only. Returns ``("shared-frozen", events)``
        for a materialized log and ``("detached", events)`` for an
        in-memory-only one. An offset at or past the end yields an
        empty list. The returned list is a copy of the slice.
        """
        with self._lock:
            self._require_open("read")
            if self._materialized:
                stored = read_generation(self._generation_path, recovery="recoverable")
                if len(stored.events) < self._high_water:
                    raise SessionPersistenceCorruptionError(
                        f"session log {self._generation_path} shrank below "
                        f"the previously observed {self._high_water} events"
                    )
                if len(stored.events) > len(self._events):
                    self._events.extend(stored.events[len(self._events) :])
                self._high_water = max(self._high_water, len(stored.events))
                state = "shared-frozen"
            else:
                state = "detached"
            start = max(offset, 0)
            if length is None:
                return state, list(self._events[start:])
            return state, list(self._events[start : start + max(length, 0)])

    def append(self, events: list[SessionEvent]) -> None:
        """Validate, snapshot, and durably append one contiguous batch.

        Guard order: closed, then read-only, then the empty batch is a
        no-op. The batch is snapshotted and validated before any
        mutation; torn-tail repair runs before the batch lands; the
        batch is fsynced before the in-memory log advances.
        """
        with self._lock:
            self._require_open("append")
            self._require_write("append")
            if not events:
                return
            batch = self._snapshot_batch(events)
            self._assert_contiguous(batch)
            rows = [self._admission_row(event) for event in batch]
            self._repair_torn_tail_locked()
            self._persist_rows_locked(rows)
            self._events.extend(batch)
            self._high_water = len(self._events)

    def _snapshot_batch(self, events: list[SessionEvent]) -> list[SessionEvent]:
        """Deep-snapshot one batch; a non-JSON value refuses the batch."""
        batch: list[SessionEvent] = []
        for event in events:
            payload = snapshot_json_value(event.payload)
            if payload is None:
                raise ValueError(
                    f'session event "{event.type}" at seq {event.seq} '
                    "carries non-JSON-serializable data"
                )
            surface_op = event.surface_op
            if surface_op is not None:
                surface_op = snapshot_json_value(surface_op)
                if surface_op is None:
                    raise ValueError(
                        f'session event "{event.type}" at seq {event.seq} '
                        "carries non-JSON-serializable surface metadata"
                    )
            source_seqs = event.source_event_seqs
            if source_seqs is not None:
                source_seqs = snapshot_json_value(source_seqs)
                if source_seqs is None:
                    raise ValueError(
                        f'session event "{event.type}" at seq {event.seq} '
                        "carries non-JSON-serializable surface metadata"
                    )
            batch.append(
                SessionEvent(
                    id=event.id,
                    session_id=event.session_id,
                    type=event.type,
                    timestamp=event.timestamp,
                    payload=payload,
                    ignorable=event.ignorable,
                    seq=event.seq,
                    surface_op=surface_op,
                    source_event_seqs=source_seqs,
                )
            )
        return batch

    def _assert_contiguous(self, batch: list[SessionEvent]) -> None:
        """The batch must continue the stored log with no gap or duplicate."""
        expected = len(self._events)
        for event in batch:
            if event.seq != expected:
                raise SessionPersistenceCorruptionError(
                    f"session {self._id} append is not contiguous: expected "
                    f"seq {expected}, got {event.seq}"
                )
            expected += 1

    def _admission_row(self, event: SessionEvent) -> dict[str, Any]:
        """Build the exact physical row Session.append builds, then admit it.

        The append contract is the admission contract: an event that
        could not be reloaded must not be appended.
        """
        validate_session_event_data(
            event, f'session event "{event.type}" at seq {event.seq}'
        )
        row: dict[str, Any] = {
            "type": event.type,
            "seq": event.seq,
            "time": int(event.timestamp * 1000),
            "data": event.payload,
            **({"ignorable": True} if event.ignorable else {}),
            **({"surfaceOp": event.surface_op} if event.surface_op is not None else {}),
            **(
                {"sourceEventSeqs": event.source_event_seqs}
                if event.source_event_seqs is not None
                else {}
            ),
        }
        try:
            assert_v4_row_admission(row, KNOWN_SESSION_EVENT_TYPES)
        except (SessionFormatError, SessionFormatUnsupportedMigrationError) as error:
            raise ValueError(
                f'session event "{event.type}" at seq {event.seq} '
                f"fails v4 admission: {error}"
            ) from error
        return row

    def _repair_torn_tail_locked(self) -> None:
        """Truncate torn bytes, then durably rewrite the recovered tail.

        Runs on the first mutation after open only. Each step clears
        its state once it lands, so a failed step retries on the next
        mutation. For plain JSONL the recovered tail is empty.
        """
        if self._torn_truncate_to is None and not self._recovered_tail:
            return
        if not self._materialized:
            return
        rows = [self._admission_row(event) for event in self._recovered_tail]
        with open(self._generation_path, "r+b") as handle:
            if self._torn_truncate_to is not None:
                handle.truncate(self._torn_truncate_to)
                handle.flush()
                os.fsync(handle.fileno())
                self._torn_truncate_to = None
            if rows:
                handle.seek(0, os.SEEK_END)
                handle.writelines(
                    _encode_line(SESSION_FORMAT_CATALOG.encode_current_event(row))
                    for row in rows
                )
                handle.flush()
                os.fsync(handle.fileno())
                self._recovered_tail = []

    def _persist_rows_locked(self, rows: list[dict[str, Any]]) -> None:
        """Append the header (if not yet written) and the batch durably.

        One append-mode open carries the header line plus the batch;
        the file and, when it was created, its parent directory are
        fsynced before the caller advances the in-memory log. A handle
        created without a lease takes one here, right before its first
        materializing write, so a second writer is refused from that
        moment on; contention raises before anything is written.
        """
        created = not self._generation_path.exists()
        _ensure_session_dir(self._generation_path.parent)
        if self._lease is None:
            self._lease = KernelLease.acquire(self._generation_path.parent, self._id)
        wrote_header = not self._header_written
        with open(self._generation_path, "ab") as handle:
            if wrote_header:
                handle.write(
                    _encode_line(
                        SESSION_FORMAT_CATALOG.encode_current_header(
                            self._header, self._inherited_event_count
                        )
                    )
                )
            handle.writelines(
                _encode_line(SESSION_FORMAT_CATALOG.encode_current_event(row))
                for row in rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        if wrote_header:
            self._header_written = True
        if created:
            _fsync_dir(self._generation_path.parent)
        self._materialized = True

    def flush(self) -> None:
        """Make the log durable: materialize a pending handle, else fsync.

        Appends are already durable on resolution, so on a
        materialized handle flush only fsyncs the file. Torn-tail
        repair does not run here; it waits for the next mutation
        (append).
        """
        with self._lock:
            self._require_open("flush")
            self._require_write("flush")
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Durability barrier, assuming the handle lock is held."""
        if not self._materialized:
            self._persist_rows_locked([])
            return
        fd = os.open(self._generation_path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def close(self) -> None:
        """Flush (write handles), then release the lease and the claim.

        Idempotent: a second call succeeds silently. The handle is
        marked closed first, so operations racing the teardown fail
        closed. A flush failure still leaves the handle closed; it
        combines with any release failure into one ExceptionGroup.
        Never raises SessionHandleClosedError.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            failures: list[Exception] = []
            if self._access == "write":
                try:
                    self._flush_locked()
                except Exception as error:  # noqa: BLE001 - combined into one group
                    failures.append(error)
            if self._lease is not None:
                try:
                    self._lease.release()
                except Exception as error:  # noqa: BLE001 - combined into one group
                    failures.append(error)
                self._lease = None
            if self._release_claim is not None:
                try:
                    self._release_claim()
                except Exception as error:  # noqa: BLE001 - combined into one group
                    failures.append(error)
                self._release_claim = None
            if failures:
                raise ExceptionGroup(f'session "{self._id}" close failed', failures)


# ---------------------------------------------------------------------------
# Session persistence service
# ---------------------------------------------------------------------------


def _migration_event_row(event: SessionEvent, session_id: str) -> dict[str, Any]:
    """Build the physical current-version row for one restored event.

    Mirrors the handle's admission row, except the millisecond time is
    rounded instead of truncated: the restored timestamp is a float
    division of the stored millisecond, and truncation would drift
    some values down by a millisecond.
    """
    validate_session_event_data(
        event, f'session event "{event.type}" at seq {event.seq}'
    )
    row: dict[str, Any] = {
        "type": event.type,
        "seq": event.seq,
        "time": round(event.timestamp * 1000),
        "data": event.payload,
        **({"ignorable": True} if event.ignorable else {}),
        **({"surfaceOp": event.surface_op} if event.surface_op is not None else {}),
        **(
            {"sourceEventSeqs": event.source_event_seqs}
            if event.source_event_seqs is not None
            else {}
        ),
    }
    try:
        assert_v4_row_admission(row, KNOWN_SESSION_EVENT_TYPES)
    except (SessionFormatError, SessionFormatUnsupportedMigrationError) as error:
        raise ValueError(
            f'session event "{event.type}" at seq {event.seq} '
            f"fails v4 admission: {error}"
        ) from error
    return row


@dataclass
class _PendingEntry:
    """A created-but-unmaterialized session: header, cut, and its handle."""

    header: dict[str, Any]
    inherited_event_count: int
    handle: JsonlSessionHandle
    generation_path: Path


class JsonlSessionPersistence(SessionPersistence):
    """Plain-JSONL session storage: create/open/stat/list/flush.

    Implements the frozen persistence contract. ``root`` is required:
    no default keeps session files from scattering as the working
    directory changes. ``live_batch_delay`` is kept for the live
    routing tracker; no timer logic lives here. The backend owns its
    live tracker: the tracker is constructed here, uninstalled, and
    installed by the row that owns this backend. Pending entries
    (created but unmaterialized) and in-process write claims are
    guarded by one lock; handle operations are never run while it is
    held, so a handle closing can never deadlock against the service.
    """

    def __init__(self, root: Path, *, live_batch_delay: float = 0.2) -> None:
        self._root = root
        self._live_batch_delay = live_batch_delay
        self._tracker = JsonlLiveTracker(self)
        self._pending: dict[str, _PendingEntry] = {}
        self._claims: dict[str, JsonlSessionHandle] = {}
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        """The storage root this backend writes under."""
        return self._root

    def install_tracker(self, ctx: Any) -> Callable[[], None]:
        """Install the backend-owned live tracker on ``ctx``.

        Returns the tracker's disposer; the tracker stays uninstalled
        until this runs, so constructing a backend starts no timers.
        """
        return self._tracker.install(ctx)

    def drain_live(self, session_id: str | None = None) -> None:
        """Drain buffered live events through their write handles.

        Drains one session when an id is given, every session's when
        it is None. A drain failure raises loudly; sessions after a
        failing one stay buffered for the next call.
        """
        self._tracker.drain_live(session_id)

    # -- helpers ---------------------------------------------------------

    def _release_claim(self, session_id: str) -> None:
        """Drop one handle's in-process claim and pending entry, if any.

        Safe without an identity check: open holds the service lock
        atomically from its claim check to its registration, so a
        release can never land between another open's check and
        registration and drop that open's claim.
        """
        with self._lock:
            self._claims.pop(session_id, None)
            self._pending.pop(session_id, None)

    def _find_generation(self, session_id: str) -> tuple[int, Path] | None:
        """Resolve the current on-disk generation for one id, if any."""
        found = find_session_dir(self._root, session_id)
        if found is None:
            return None
        return resolve_current_generation(found)

    def _stat_generation(
        self, path: Path, session_id: str | None
    ) -> SessionPersistenceSnapshot:
        """Header-only snapshot of one generation file; never decodes rows.

        A newer-version header is intact but unreadable here: it is
        reported with no format status and the stored header as-is,
        never raised. A malformed header is damage and raises
        corruption. When a session id is given, the header must name
        it.
        """
        info = os.stat(path)
        revision = (
            f"{info.st_dev}:{info.st_ino}:{info.st_size}:"
            f"{info.st_mtime_ns}:{info.st_ctime_ns}"
        )
        data = path.read_bytes()
        newline = data.find(b"\n")
        if newline < 0:
            raise SessionPersistenceCorruptionError(
                f"corrupt session log: first line is not a session header in {path}"
            )
        try:
            header_value = json.loads(data[:newline].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SessionPersistenceCorruptionError(
                f"corrupt session header in {path}: {error}"
            ) from error
        read = SESSION_FORMAT_CATALOG.read_header(header_value)
        if read.status == "malformed":
            assert read.reason is not None
            raise SessionPersistenceCorruptionError(read.reason)
        if read.status == "unsupported":
            # The version gate ran before any shape check, so an
            # unsupported value is a JSON object: the stored header.
            header: dict[str, Any] = header_value
            format_status: Literal["current", "migration-required"] | None = None
        else:
            assert read.header is not None
            header = read.header
            format_status = (
                "current" if read.status == "current" else "migration-required"
            )
        if session_id is not None and str(header.get("id")) != session_id:
            raise SessionPersistenceCorruptionError(
                f"session log {path} header id does not match {session_id!r}"
            )
        return SessionPersistenceSnapshot(
            header=header,
            format_status=format_status,
            revision=revision,
            event_count=None,
            size_bytes=info.st_size,
            location=path,
        )

    def _prefix_digest(self, events: list[SessionEvent], session_id: str) -> str:
        """SHA-256 over the canonical rows of the first sixteen events."""
        lines = b"".join(
            _encode_line(
                SESSION_FORMAT_CATALOG.encode_current_event(
                    _migration_event_row(event, session_id)
                )
            )
            for event in events[:16]
        )
        return hashlib.sha256(lines).hexdigest()

    def _publish_migration(
        self, source_path: Path, session_id: str, result: ReadResult
    ) -> Path:
        """Publish one historical generation as the current generation.

        The migrated artifact is written to a temp file in the same
        directory, fsynced, atomically renamed to the current
        generation name, and verified in-process (id, event count,
        and expected prefix digest) before the source generation is
        removed. On any failure the new file is removed, the source
        stays untouched, and the error propagates.
        """
        directory = source_path.parent
        target_path = directory / session_format_log_filename(CURRENT_FORMAT_VERSION)
        tmp_path = directory / f"session.v4.jsonl.tmp-{os.getpid()}"
        header_line = _encode_line(
            SESSION_FORMAT_CATALOG.encode_current_header(
                result.header, result.inherited_event_count
            )
        )
        event_lines = [
            _encode_line(
                SESSION_FORMAT_CATALOG.encode_current_event(
                    _migration_event_row(event, session_id)
                )
            )
            for event in result.events
        ]
        expected_digest = self._prefix_digest(result.events, session_id)
        created_tmp = False
        try:
            try:
                fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                created_tmp = True
            except FileExistsError:
                # Only a stale regular file may be cleared for a retry.
                # Anything else at the path is not ours to remove, so the
                # original error propagates and the path stays untouched.
                # On Windows a directory at the temp path raises
                # PermissionError from os.open, never FileExistsError, so
                # this re-raise is unreachable there.
                if not tmp_path.is_file():
                    raise  # pragma: no cover
                tmp_path.unlink()
                fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                created_tmp = True
            with os.fdopen(fd, "wb") as handle:
                handle.write(header_line)
                for line in event_lines:
                    handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target_path)
            _fsync_dir(directory)
        except BaseException:
            if created_tmp:
                tmp_path.unlink(missing_ok=True)
            raise
        try:
            verified = verify_current_generation(target_path)
            if str(verified.header.get("id")) != session_id:
                raise SessionPersistenceCorruptionError(
                    f"migrated session {session_id} failed verification: "
                    f"header id is {verified.header.get('id')!r}"
                )
            if len(verified.events) != len(result.events):
                raise SessionPersistenceCorruptionError(
                    f"migrated session {session_id} failed verification: "
                    f"expected {len(result.events)} events, "
                    f"got {len(verified.events)}"
                )
            if self._prefix_digest(verified.events, session_id) != expected_digest:
                raise SessionPersistenceCorruptionError(
                    f"migrated session {session_id} failed verification: "
                    "prefix digest mismatch"
                )
        except BaseException:
            try:
                target_path.unlink(missing_ok=True)
            except OSError:
                # Best-effort cleanup; the verification error is the one
                # that matters.
                pass
            raise
        source_path.unlink()
        _fsync_dir(directory)
        return target_path

    # -- create ----------------------------------------------------------

    def create(
        self,
        header: dict[str, Any],
        inherited_event_count: int | None = None,
    ) -> JsonlSessionHandle:
        """Register a new session and return its unmaterialized write handle.

        The header is validated; a seeded header requires an explicit
        inherited count, an unseeded one requires none (or zero). An
        id that is pending or already on disk is refused. The handle
        has no lease and no filesystem footprint until its first
        materializing write or flush.
        """
        validated = snapshot_header(header)
        seeded = bool(validated.get("isSeeded"))
        if seeded:
            if (
                not isinstance(inherited_event_count, int)
                or isinstance(inherited_event_count, bool)
                or inherited_event_count < 0
            ):
                raise ValueError(
                    "a seeded session header requires inherited_event_count "
                    "as an int >= 0"
                )
            cut = inherited_event_count
        elif inherited_event_count not in (None, 0):
            raise ValueError(
                "an unseeded session header requires inherited_event_count None or 0"
            )
        else:
            cut = 0
        session_id = str(validated["id"])
        cwd = validated.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError("session header cwd must be a string")
        with self._lock:
            if session_id in self._pending:
                raise SessionAlreadyExistsError(session_id)
            if self._find_generation(session_id) is not None:
                raise SessionAlreadyExistsError(session_id)
            generation_path = session_dir(
                self._root, cwd, session_id
            ) / session_format_log_filename(CURRENT_FORMAT_VERSION)
            handle = JsonlSessionHandle(
                session_id=session_id,
                header=validated,
                inherited_event_count=cut,
                access="write",
                generation_path=generation_path,
                lease=None,
                materialized=False,
                header_written=False,
                release_claim=lambda: self._release_claim(session_id),
            )
            self._pending[session_id] = _PendingEntry(
                header=validated,
                inherited_event_count=cut,
                handle=handle,
                generation_path=generation_path,
            )
            self._claims[session_id] = handle
            return handle

    # -- open ------------------------------------------------------------

    def open(
        self, session_id: str, access: Literal["read", "write"]
    ) -> JsonlSessionHandle:
        """Open a session for reading or single-writer writing."""
        if access == "read":
            return self._open_read(session_id)
        if access == "write":
            return self._open_write(session_id)
        raise ValueError(f'unknown session access "{access}"')

    def _open_read(self, session_id: str) -> JsonlSessionHandle:
        """Read without taking ownership; works while write is held.

        A pending session serves its in-memory events through a new
        read handle. An on-disk historical log is primed with the
        migrated events and is never published by a read.
        """
        with self._lock:
            entry = self._pending.get(session_id)
        if entry is not None:
            try:
                _, events = entry.handle.read()
            except SessionHandleClosedError:
                entry = None
            else:
                return JsonlSessionHandle(
                    session_id=session_id,
                    header=entry.header,
                    inherited_event_count=entry.inherited_event_count,
                    access="read",
                    generation_path=entry.generation_path,
                    lease=None,
                    events=events,
                    materialized=False,
                    header_written=False,
                )
        resolved = self._find_generation(session_id)
        if resolved is None:
            raise SessionPersistenceNotFoundError(session_id)
        _version, path = resolved
        result = read_generation(path, recovery="recoverable")
        if str(result.header.get("id")) != session_id:
            raise SessionPersistenceCorruptionError(
                f"session log {path} header id does not match {session_id!r}"
            )
        return JsonlSessionHandle(
            session_id=session_id,
            header=result.header,
            inherited_event_count=result.inherited_event_count,
            access="read",
            generation_path=path,
            lease=None,
            events=result.events,
            torn_truncate_to=result.torn_truncate_to,
            recovered_tail=result.recovered_tail,
            materialized=True,
            header_written=True,
        )

    def _open_write(self, session_id: str) -> JsonlSessionHandle:
        """Claim single-writer ownership, then open the stored log.

        The service lock is held for the whole open, so the claim
        check and the registration below are one atomic step: no
        second writer can slip in between. A historical log has its
        migration published before the handle is built. Every failure
        releases the lease (and takes no claim).
        """
        with self._lock:
            if session_id in self._claims:
                raise SessionAlreadyOwnedError(session_id)
            resolved = self._find_generation(session_id)
            if resolved is None:
                raise SessionPersistenceNotFoundError(session_id)
            _version, path = resolved
            lease = KernelLease.acquire(path.parent, session_id)
            try:
                result = read_generation(path, recovery="recoverable")
                if str(result.header.get("id")) != session_id:
                    raise SessionPersistenceCorruptionError(
                        f"session log {path} header id does not match {session_id!r}"
                    )
                generation_path = path
                if result.format_status == "migration-required":
                    generation_path = self._publish_migration(path, session_id, result)
            except BaseException:
                lease.release()
                raise
            handle = JsonlSessionHandle(
                session_id=session_id,
                header=result.header,
                inherited_event_count=result.inherited_event_count,
                access="write",
                generation_path=generation_path,
                lease=lease,
                events=result.events,
                torn_truncate_to=result.torn_truncate_to,
                recovered_tail=result.recovered_tail,
                materialized=True,
                header_written=True,
                release_claim=lambda: self._release_claim(session_id),
            )
            self._claims[session_id] = handle
            return handle

    # -- stat / list -----------------------------------------------------

    def stat(self, session_id: str) -> SessionPersistenceSnapshot | None:
        """Header-only observation; ``None`` when no log is visible."""
        with self._lock:
            entry = self._pending.get(session_id)
        if entry is not None:
            try:
                _, events = entry.handle.read()
            except SessionHandleClosedError:
                entry = None
            else:
                return SessionPersistenceSnapshot(
                    header=entry.header,
                    format_status="current",
                    revision=f"pending:{session_id}",
                    event_count=len(events),
                    size_bytes=None,
                    location=None,
                )
        resolved = self._find_generation(session_id)
        if resolved is None:
            return None
        _version, path = resolved
        return self._stat_generation(path, session_id)

    def list(self) -> list[SessionPersistenceSnapshot]:
        """Every stored session visible to this process, no promised order.

        Pending entries are listed from memory; every project
        directory is scanned for session dirs holding a canonical
        generation. A materialized pending session is listed once,
        from its pending entry.
        """
        with self._lock:
            pending = dict(self._pending)
        snapshots: list[SessionPersistenceSnapshot] = []
        seen: set[str] = set()
        for session_id, entry in pending.items():
            try:
                _, events = entry.handle.read()
            except SessionHandleClosedError:
                continue
            snapshots.append(
                SessionPersistenceSnapshot(
                    header=entry.header,
                    format_status="current",
                    revision=f"pending:{session_id}",
                    event_count=len(events),
                    size_bytes=None,
                    location=None,
                )
            )
            seen.add(session_id)
        if self._root.is_dir():
            for project in self._root.iterdir():
                if not project.is_dir():
                    continue
                for candidate in project.iterdir():
                    if not candidate.is_dir():
                        continue
                    resolved = resolve_current_generation(candidate)
                    if resolved is None:
                        continue
                    _version, path = resolved
                    snapshot = self._stat_generation(path, None)
                    if str(snapshot.header.get("id")) in seen:
                        continue
                    snapshots.append(snapshot)
        return snapshots

    # -- flush -----------------------------------------------------------

    def flush(self) -> None:
        """Flush every active write handle owned by this instance.

        Each handle flushes exactly as its own ``flush`` would; a
        concurrently closed handle counts as flushed. Failures are
        collected and raised together, naming each failed session id;
        the rest still flush.
        """
        with self._lock:
            handles = list(self._claims.values())
        failures: list[Exception] = []
        failed_ids: list[str] = []
        for handle in handles:
            if handle.closed:
                continue
            try:
                handle.flush()
            except SessionHandleClosedError:
                continue
            except Exception as error:  # noqa: BLE001 - combined into one group
                wrapped = RuntimeError(f'session "{handle.id}" flush failed: {error}')
                wrapped.__cause__ = error
                failures.append(wrapped)
                failed_ids.append(handle.id)
        if failures:
            raise ExceptionGroup(
                f"session flush failed for: {', '.join(failed_ids)}", failures
            )


class JsonlSessionPersistenceService(Service):
    """Row service owning the JSONL persistence backend.

    A row service providing ``sessionPersistence``. With a storage root
    configured it constructs the backend and installs the backend's
    live tracker on the context; with no root it activates hollow —
    the slot exists but carries no backend, and sessions resolve
    nothing from it. Row unload uninstalls the tracker.
    """

    def __init__(
        self,
        ctx: Context,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Build the service; the backend exists only with a root configured.

        ``config`` carries ``root`` and ``live_batch_delay``; a falsy
        root activates the service hollow.
        """
        super().__init__(ctx, "sessionPersistence")
        cfg = config or {}
        root = cfg.get("root") or None
        self._backend: JsonlSessionPersistence | None = None
        self._tracker_dispose: Disposer | None = None
        if root is not None:
            backend = JsonlSessionPersistence(
                Path(root),
                live_batch_delay=cfg.get("live_batch_delay", 0.2),
            )
            self._backend = backend
            self._tracker_dispose = backend.install_tracker(ctx)
        # Row unload runs this: the tracker must stop routing once the
        # row is gone, which slot removal alone would not ensure.
        ctx.effect(lambda: self.dispose)

    @property
    def backend(self) -> JsonlSessionPersistence | None:
        """The owned backend, or ``None`` when the row is hollow."""
        return self._backend

    @property
    def active(self) -> bool:
        """Whether this row owns a live backend."""
        return self._backend is not None

    def dispose(self) -> None:
        """Uninstall the live tracker. Idempotent."""
        if self._tracker_dispose is not None:
            dispose = self._tracker_dispose
            self._tracker_dispose = None
            dispose()


# ---------------------------------------------------------------------------
# Live routing
# ---------------------------------------------------------------------------


class JsonlLiveTracker:
    """Route live session events to their write handles in batches.

    ``session/event`` arrivals are cloned and buffered per session; one
    timer per session drains its buffer through the session's write
    handle after ``live_batch_delay``. The first deadline wins: a
    pending timer is left alone when more events arrive. A drain
    failure keeps the batch buffered and pauses that session's timer
    until an explicit ``drain_live`` retries loudly. ``session/flush``
    drains then flushes; ``session/disposed`` drains a final time and
    only warns on failure. The disposer returned by ``install``
    unregisters the listeners and cancels pending timers; it is
    idempotent.
    """

    def __init__(self, persistence: JsonlSessionPersistence) -> None:
        """Hold the persistence whose write claims name the targets."""
        self._persistence = persistence
        self._buffers: dict[str, list[SessionEvent]] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._paused: set[str] = set()
        self._lock = threading.Lock()
        self._installed = False
        self._install_token: object | None = None

    @staticmethod
    def _session_id_of(session: Any) -> str:
        """The id of a session object, or a bare id string as-is."""
        if isinstance(session, str):
            return session
        return str(session.session_id)

    def _write_handle(self, session_id: str) -> JsonlSessionHandle | None:
        """The live write handle for one session, if one is claimed."""
        with self._persistence._lock:
            return self._persistence._claims.get(session_id)

    @staticmethod
    def _clone_event(event: SessionEvent) -> SessionEvent:
        """Detached copy of one event; a non-JSON value refuses loudly."""
        payload = snapshot_json_value(event.payload)
        if payload is None:
            raise ValueError(
                f'session event "{event.type}" at seq {event.seq} '
                "carries non-JSON-serializable data"
            )
        surface_op = event.surface_op
        if surface_op is not None:
            surface_op = snapshot_json_value(surface_op)
            if surface_op is None:
                raise ValueError(
                    f'session event "{event.type}" at seq {event.seq} '
                    "carries non-JSON-serializable surface metadata"
                )
        source_seqs = event.source_event_seqs
        if source_seqs is not None:
            source_seqs = snapshot_json_value(source_seqs)
            if source_seqs is None:
                raise ValueError(
                    f'session event "{event.type}" at seq {event.seq} '
                    "carries non-JSON-serializable surface metadata"
                )
        return SessionEvent(
            id=event.id,
            session_id=event.session_id,
            type=event.type,
            timestamp=event.timestamp,
            payload=payload,
            ignorable=event.ignorable,
            seq=event.seq,
            surface_op=surface_op,
            source_event_seqs=source_seqs,
        )

    def enqueue_live(self, session: Any, event: SessionEvent) -> None:
        """Buffer one cloned live event for its session's write handle.

        Events for a session with no claimed write handle are ignored:
        no durable target exists for them. A pending batching timer is
        left alone, so the batch drains at the first deadline.
        """
        clone = self._clone_event(event)
        session_id = self._session_id_of(session)
        if self._write_handle(session_id) is None:
            return
        with self._lock:
            self._buffers.setdefault(session_id, []).append(clone)
            self._schedule_locked(session_id)

    def _schedule_locked(self, session_id: str) -> None:
        """Start one batching timer, unless one runs or is paused."""
        if session_id in self._timers or session_id in self._paused:
            return
        timer = threading.Timer(
            self._persistence._live_batch_delay, self._on_timer, args=(session_id,)
        )
        timer.daemon = True
        self._timers[session_id] = timer
        timer.start()

    def _cancel_timer_locked(self, session_id: str) -> None:
        """Cancel one session's pending timer, if any."""
        timer = self._timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()

    def _on_timer(self, session_id: str) -> None:
        """Drain one session when its batching deadline arrives.

        A failure keeps the batch buffered and pauses the timer; the
        next explicit drain retries loudly. The warning is the only
        trace of a timer-thread failure.
        """
        with self._lock:
            self._timers.pop(session_id, None)
        try:
            self._drain_one(session_id)
        except Exception as error:  # noqa: BLE001 - warned, retried explicitly
            logger.warning('session "%s": live drain failed: %s', session_id, error)

    def drain_live(self, session_id: str | None = None) -> None:
        """Drain one session's buffer, or every session's when id is None.

        One durable append per buffered batch, preserving order. The
        first failing session raises loudly; sessions after it stay
        buffered for the next call.
        """
        with self._lock:
            if session_id is None:
                targets = [sid for sid, batch in self._buffers.items() if batch]
            else:
                targets = [session_id]
        for target in targets:
            self._drain_one(target)

    def _drain_one(self, session_id: str) -> None:
        """Move one session's buffer through its write handle.

        The batch leaves the buffer before the append runs, so a
        concurrent drain can never append it twice. A failure puts
        the batch back at the front, pauses the timer, and raises.
        With no claimed write handle the batch stays buffered and
        the call succeeds silently: there is nowhere to write it yet.
        """
        with self._lock:
            batch = self._buffers.get(session_id)
            if not batch:
                self._cancel_timer_locked(session_id)
                return
            handle = self._write_handle(session_id)
            if handle is None:
                return
            self._paused.discard(session_id)
            self._cancel_timer_locked(session_id)
            pending = self._buffers.pop(session_id)
        try:
            handle.append(pending)
        except Exception:
            with self._lock:
                self._buffers[session_id] = pending + self._buffers.get(session_id, [])
                self._paused.add(session_id)
            raise

    def _on_flush(self, session: Any) -> None:
        """Drain one session, then flush its write handle.

        Drain-then-flush ordering: the flush is the durability
        barrier for the drained batch. Failures propagate. A session
        with no claimed write handle has nothing to flush.
        """
        session_id = self._session_id_of(session)
        self.drain_live(session_id)
        handle = self._write_handle(session_id)
        if handle is not None:
            handle.flush()

    def _on_disposed(self, session: Any) -> None:
        """Final drain for a disposed session; failures only warn."""
        session_id = self._session_id_of(session)
        try:
            self.drain_live(session_id)
        except Exception as error:  # noqa: BLE001 - disposal must not raise
            logger.warning(
                'session "%s": final live drain failed: %s', session_id, error
            )

    def install(self, ctx: Any) -> Callable[[], None]:
        """Register the live-routing listeners; return their disposer.

        The disposer unregisters the listeners (or, where the context
        offers no unregister call, stops them by token), cancels
        pending timers, and is idempotent. A second install while one
        is active raises; after disposal a fresh install is allowed.
        """
        with self._lock:
            if self._installed:
                raise ValueError("live tracker is already installed")
            token = object()

        def _on_event(session: Any, event: SessionEvent) -> None:
            if self._install_token is not token:
                return
            self.enqueue_live(session, event)

        def _on_flush(session: Any) -> None:
            if self._install_token is not token:
                return
            self._on_flush(session)

        def _on_disposed(session: Any) -> None:
            if self._install_token is not token:
                return
            self._on_disposed(session)

        disposers = [
            ctx.on("session/event", _on_event),
            ctx.on("session/flush", _on_flush),
            ctx.on("session/disposed", _on_disposed),
        ]
        with self._lock:
            self._installed = True
            self._install_token = token
        state = {"done": False}

        def dispose() -> None:
            with self._lock:
                if state["done"] or self._install_token is not token:
                    return
                state["done"] = True
                self._installed = False
                self._install_token = None
                timers = list(self._timers.values())
                self._timers.clear()
            for disposer in disposers:
                if callable(disposer):
                    disposer()
            for timer in timers:
                timer.cancel()

        return dispose
