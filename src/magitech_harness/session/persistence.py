"""Durable session persistence: storage contract and ownership errors.

A persistence backend stores one append-only event log per session and
hands out one handle per open channel onto a log. Events are contiguous
from seq 0 and never rewritten; a torn physical tail is never returned
to a reader. ``append`` is best-effort: a resolved append is accepted,
ordered, and visible to later reads on the same backend instance, but
only a resolved ``flush`` promises the batch survives a crash.
``flush`` — per handle or service-wide — is the durability barrier.

A session created through ``create`` is observable through ``stat``,
``list``, and ``open`` in this process from the moment ``create``
resolves, even while a backend defers physical materialization; other
processes see the session only once it materializes, and a session
that never materialized before a crash never existed. Once an
``append`` or ``flush`` resolves, reads started afterwards on the same
backend instance observe at least that prefix.

Ownership: a ``read`` handle never takes ownership and works while
another handle holds ``write``. A ``write`` handle is the session's
single mutator; opening a second one while the first is active is
refused. A handle that loses ownership is closed and reopened, never
silently re-acquired.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.format import (
    SessionFormatUnsupportedError,
    SessionPersistenceCorruptionError,
)

__all__ = [
    "SessionAlreadyExistsError",
    "SessionAlreadyOwnedError",
    "SessionFormatUnsupportedError",
    "SessionHandle",
    "SessionHandleClosedError",
    "SessionOwnershipLostError",
    "SessionPersistence",
    "SessionPersistenceCorruptionError",
    "SessionPersistenceNotFoundError",
    "SessionPersistenceSnapshot",
    "SessionReadOnlyError",
]


class SessionPersistenceNotFoundError(Exception):
    """The requested session has no durable log visible to this caller."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f'session "{session_id}" not found')
        self.session_id = session_id


class SessionAlreadyExistsError(Exception):
    """``create`` targeted a session id that already exists in this backend."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f'session "{session_id}" already exists')
        self.session_id = session_id


class SessionAlreadyOwnedError(Exception):
    """A write open found the session already bound to an active write handle."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f'session "{session_id}" is already owned by an active write handle'
        )
        self.session_id = session_id


class SessionReadOnlyError(Exception):
    """A mutation (``append``/``flush``) was called on a read handle."""

    def __init__(self, session_id: str, operation: str) -> None:
        super().__init__(
            f'session "{session_id}": {operation} is not available on a read handle'
        )
        self.session_id = session_id
        self.operation = operation


class SessionOwnershipLostError(Exception):
    """A write handle's ownership is permanently gone.

    The lease expired, a renewal failed, or the durable ownership record
    no longer names this handle. Declared for the cross-process lease
    layer; the shipped backends never raise it. The handle never
    re-acquires ownership — close it and reopen for write.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f'session "{session_id}": write ownership was lost; close this handle'
            " and reopen"
        )
        self.session_id = session_id


class SessionHandleClosedError(Exception):
    """An operation was called on a handle after ``close()`` was called."""

    def __init__(self, session_id: str, operation: str) -> None:
        super().__init__(f'session "{session_id}": {operation} on a closed handle')
        self.session_id = session_id
        self.operation = operation


@dataclass(frozen=True)
class SessionPersistenceSnapshot:
    """Lightweight stored-session observation without reading the event log."""

    header: dict[str, Any]
    """Detached logical header for one stored session."""

    format_status: Literal["current", "migration-required"] | None
    """Header-only format classification; absent without versioned metadata."""

    revision: str
    """Opaque change token, comparable only within the same backend instance
    and session id. Equal revisions may be treated as an unchanged log;
    unequal revisions promise nothing. Ownership churn does not change a
    revision."""

    event_count: int | None = None
    """Logical event count, when the backend can provide it cheaply."""

    size_bytes: int | None = None
    """Physical artifact byte size, when the backend can provide it cheaply."""

    location: Path | None = None
    """Backend-resolved location of the materialized per-session artifact.

    Absent when the backend has no materialized per-session file,
    including an unmaterialized pending session.
    """


class SessionHandle(ABC):
    """One open channel onto a stored session's append-only event log.

    A handle is single-owner state, not a shared service: ``read`` never
    observes an older state than a prior read on this handle, a ``write``
    handle reads its own successful appends, and ``close()`` is the one
    teardown. Every operation on a closed handle raises
    :class:`SessionHandleClosedError`, except ``close`` itself, which is
    idempotent.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """The stored session this handle addresses."""

    @property
    @abstractmethod
    def header(self) -> dict[str, Any]:
        """The immutable stored header, fixed at ``create``/``open``."""

    @property
    @abstractmethod
    def inherited_event_count(self) -> int:
        """Exact fork-inherited prefix length stored with the log.

        ``0`` when the header is not seeded. Storage metadata paired with
        the header for every body read, never part of the replayable event
        log.
        """

    @property
    @abstractmethod
    def access(self) -> Literal["read", "write"]:
        """Whether this handle may mutate the log."""

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Whether ``close()`` has been called on this handle."""

    @abstractmethod
    def read(
        self, offset: int = 0, length: int | None = None
    ) -> tuple[str, list[SessionEvent]]:
        """Read a slice of the valid contiguous logical log.

        Returns ``(event_state, events)`` where ``event_state`` is
        ``'detached'`` or ``'shared-frozen'`` and ``events`` are the
        requested events in seq order, starting at ``offset`` for at most
        ``length`` events (the rest of the log when ``length`` is ``None``).
        An offset at or past the end returns an empty list. A torn
        physical tail is never returned, and a stored log that shrank
        below a previously observed prefix is refused with
        :class:`SessionPersistenceCorruptionError`.
        """

    @abstractmethod
    def append(self, events: list[SessionEvent]) -> None:
        """Append a contiguous batch continuing the current logical end.

        The first event's ``seq`` must equal the stored next-seq, else
        :class:`SessionPersistenceCorruptionError`. An empty batch is a
        no-op. Raises :class:`SessionReadOnlyError` on a read handle.
        """

    @abstractmethod
    def flush(self) -> None:
        """The durability barrier for this handle.

        On return every acknowledged append is durable and the session is
        materialized for other processes; an empty created session becomes
        durably listable here. Raises :class:`SessionReadOnlyError` on a
        read handle.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the handle.

        A read handle frees local resources; a write handle completes
        pending durability and releases write ownership. Idempotent:
        calling ``close`` on a closed handle succeeds.
        """


class SessionPersistence(ABC):
    """Durable append-only session storage addressed through per-session handles.

    Storage semantics shared by every backend: events are contiguous from
    seq 0 and never rewritten; a torn physical tail is never returned to a
    reader and is truncated by the write path before its first append;
    reads validate current-format records only and refuse unknown
    vocabulary fail-closed. ``append`` persists best-effort; ``flush``
    — per handle or service-wide — is the durability barrier.
    """

    @abstractmethod
    def create(
        self,
        header: dict[str, Any],
        inherited_event_count: int | None = None,
    ) -> SessionHandle:
        """Create a new stored session and take its write ownership.

        The header is validated via ``snapshot_header`` in the format
        layer. A seeded header requires ``inherited_event_count`` provided as an
        int >= 0; an unseeded header requires it ``None`` or ``0``. An
        existing id (pending or on-disk) raises
        :class:`SessionAlreadyExistsError`. Returns an unmaterialized
        write handle: no lock yet, no filesystem footprint. The handle
        holds in-process write ownership until it is closed.
        """

    @abstractmethod
    def open(self, session_id: str, access: Literal["read", "write"]) -> SessionHandle:
        """Open an existing stored session.

        ``read`` never takes ownership and works while another handle
        holds write ownership. ``write`` atomically claims single-writer
        ownership; an existing active owner raises
        :class:`SessionAlreadyOwnedError`. A missing id raises
        :class:`SessionPersistenceNotFoundError`. Write-open of a
        historical generation publishes its migration first.
        """

    @abstractmethod
    def flush(self) -> None:
        """Flush every active write handle owned by this instance in one barrier.

        Each handle flushes exactly as its own ``flush`` would. Failures
        raise an :class:`ExceptionGroup` whose message names each failed
        session id; the remaining handles still flush. A concurrently
        closed handle counts as flushed.
        """

    @abstractmethod
    def stat(self, session_id: str) -> SessionPersistenceSnapshot | None:
        """Observe one stored session without reading its event log.

        Returns the snapshot, or ``None`` when the session does not exist.
        Pending entries are observable.
        """

    @abstractmethod
    def list(self) -> list[SessionPersistenceSnapshot]:
        """List every stored session visible to this process.

        One snapshot per stored session, in no promised order.
        """
