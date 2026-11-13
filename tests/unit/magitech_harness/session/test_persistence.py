"""Contract tests for the session persistence ABCs.

No backend exists yet, so the tests prove the contract semantics against
a small in-memory stub backend defined here that implements the ABCs.
The stub is test scaffolding, not shipped behavior: it keeps one event
list per session, tracks the single active write handle per session,
and lets a test mark a handle's flush as failing.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

import magitech_harness.session as session_pkg
from magitech_harness.session import format as format_module
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.format import (
    CURRENT_FORMAT_VERSION,
    SessionFormatError,
    snapshot_header,
)
from magitech_harness.session.persistence import (
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
    SessionFormatUnsupportedError,
    SessionHandle,
    SessionHandleClosedError,
    SessionOwnershipLostError,
    SessionPersistence,
    SessionPersistenceCorruptionError,
    SessionPersistenceNotFoundError,
    SessionPersistenceSnapshot,
    SessionReadOnlyError,
)


def _header(
    session_id: str, *, seeded: bool = False, version: int = 4
) -> dict[str, Any]:
    """One minimal valid logical header."""
    return {
        "version": version,
        "id": session_id,
        "createdAt": 0,
        "isSeeded": seeded,
        "delegationDepth": 0,
    }


def _event(session_id: str, seq: int, event_type: str = "turn/start") -> SessionEvent:
    """One event addressed to ``session_id`` at ``seq``."""
    return SessionEvent(
        id=f"e{seq}", session_id=session_id, type=event_type, seq=seq, payload={}
    )


@dataclass
class _Record:
    """One stored session in the in-memory stub backend."""

    header: dict[str, Any]
    inherited_event_count: int
    events: list[SessionEvent] = field(default_factory=list)


class _MemoryHandle(SessionHandle):
    """In-memory handle proving the persistence contract against the ABCs.

    ``fail_flush`` marks the next flushes as failing so the service-wide
    barrier can be proven to aggregate failures; ``flush_count`` counts
    successful flushes so tests can observe that a flush happened.
    """

    def __init__(
        self,
        backend: _MemoryPersistence,
        record: _Record,
        access: Literal["read", "write"],
    ) -> None:
        self._backend = backend
        self._record = record
        self._access = access
        self._closed = False
        self.fail_flush = False
        self.flush_count = 0

    @property
    def id(self) -> str:
        return str(self._record.header["id"])

    @property
    def header(self) -> dict[str, Any]:
        return dict(self._record.header)

    @property
    def inherited_event_count(self) -> int:
        return self._record.inherited_event_count

    @property
    def access(self) -> Literal["read", "write"]:
        return self._access

    @property
    def closed(self) -> bool:
        return self._closed

    def read(
        self, offset: int = 0, length: int | None = None
    ) -> tuple[str, list[SessionEvent]]:
        if self._closed:
            raise SessionHandleClosedError(self.id, "read")
        selected = [
            event.model_copy(deep=True)
            for event in self._record.events
            if event.seq >= offset
        ]
        if length is not None:
            selected = selected[:length]
        return "detached", selected

    def append(self, events: list[SessionEvent]) -> None:
        if self._closed:
            raise SessionHandleClosedError(self.id, "append")
        if self._access == "read":
            raise SessionReadOnlyError(self.id, "append")
        if not events:
            return
        expected = self._record.inherited_event_count + len(self._record.events)
        for index, event in enumerate(events):
            if event.seq != expected + index:
                raise SessionPersistenceCorruptionError(
                    f'append seq mismatch for "{self.id}": expected '
                    f"{expected + index} at index {index}, got {event.seq}"
                )
        self._record.events.extend(event.model_copy(deep=True) for event in events)

    def flush(self) -> None:
        if self._closed:
            raise SessionHandleClosedError(self.id, "flush")
        if self._access == "read":
            raise SessionReadOnlyError(self.id, "flush")
        if self.fail_flush:
            raise RuntimeError(f'flush failed for session "{self.id}"')
        self.flush_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._backend._release(self)


class _MemoryPersistence(SessionPersistence):
    """In-memory backend proving the persistence contract against the ABCs."""

    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}
        self._write_handles: dict[str, _MemoryHandle] = {}

    def create(
        self,
        header: dict[str, Any],
        inherited_event_count: int | None = None,
    ) -> SessionHandle:
        stored = copy.deepcopy(snapshot_header(header))
        session_id = str(stored["id"])
        if session_id in self._records:
            raise SessionAlreadyExistsError(session_id)
        if stored["isSeeded"]:
            if (
                not isinstance(inherited_event_count, int)
                or isinstance(inherited_event_count, bool)
                or inherited_event_count < 0
            ):
                raise ValueError(
                    "seeded sessions require inherited_event_count as an int >= 0"
                )
            count = inherited_event_count
        else:
            if inherited_event_count is not None and inherited_event_count != 0:
                raise ValueError(
                    "unseeded sessions require inherited_event_count None or 0"
                )
            count = 0
        record = _Record(header=stored, inherited_event_count=count)
        self._records[session_id] = record
        handle = _MemoryHandle(self, record, "write")
        self._write_handles[session_id] = handle
        return handle

    def open(self, session_id: str, access: Literal["read", "write"]) -> SessionHandle:
        record = self._records.get(session_id)
        if record is None:
            raise SessionPersistenceNotFoundError(session_id)
        if access == "write":
            if session_id in self._write_handles:
                raise SessionAlreadyOwnedError(session_id)
            handle = _MemoryHandle(self, record, "write")
            self._write_handles[session_id] = handle
            return handle
        return _MemoryHandle(self, record, "read")

    def flush(self) -> None:
        failures: list[Exception] = []
        failed_ids: list[str] = []
        for session_id, handle in self._write_handles.items():
            try:
                handle.flush()
            except Exception as error:  # noqa: BLE001 - the barrier aggregates
                # every per-handle failure by design; the rest still flush.
                failures.append(error)
                failed_ids.append(session_id)
        if failures:
            raise ExceptionGroup(
                f"session flush failed for: {', '.join(sorted(failed_ids))}",
                failures,
            )

    def stat(self, session_id: str) -> SessionPersistenceSnapshot | None:
        record = self._records.get(session_id)
        if record is None:
            return None
        return self._snapshot(session_id, record)

    def list(self) -> list[SessionPersistenceSnapshot]:
        return [
            self._snapshot(session_id, record)
            for session_id, record in self._records.items()
        ]

    def _snapshot(self, session_id: str, record: _Record) -> SessionPersistenceSnapshot:
        return SessionPersistenceSnapshot(
            header=dict(record.header),
            format_status=(
                "current"
                if record.header["version"] == CURRENT_FORMAT_VERSION
                else "migration-required"
            ),
            revision=f"{session_id}:{len(record.events)}",
            event_count=len(record.events),
        )

    def _release(self, handle: _MemoryHandle) -> None:
        owner = self._write_handles.get(handle.id)
        if owner is handle:
            del self._write_handles[handle.id]


# ---------------------------------------------------------------------------
# Errors: taxonomy, messages, and exports
# ---------------------------------------------------------------------------


def test_errors_carry_session_id_and_stable_message() -> None:
    """Verifies each ownership error names its session and operation."""
    not_found = SessionPersistenceNotFoundError("missing")
    assert not_found.session_id == "missing"
    assert str(not_found) == 'session "missing" not found'

    exists = SessionAlreadyExistsError("taken")
    assert exists.session_id == "taken"
    assert str(exists) == 'session "taken" already exists'

    owned = SessionAlreadyOwnedError("busy")
    assert owned.session_id == "busy"
    assert str(owned) == 'session "busy" is already owned by an active write handle'

    read_only = SessionReadOnlyError("ro", "append")
    assert read_only.session_id == "ro"
    assert read_only.operation == "append"
    assert str(read_only) == 'session "ro": append is not available on a read handle'

    closed = SessionHandleClosedError("gone", "flush")
    assert closed.session_id == "gone"
    assert closed.operation == "flush"
    assert str(closed) == 'session "gone": flush on a closed handle'


def test_ownership_lost_error_exists_but_is_never_thrown() -> None:
    """Verifies the declared-but-never-thrown error is importable and distinct.

    No shipped backend raises it; the test asserts only that the class
    exists, carries its session, and is not any other error class.
    """
    error = SessionOwnershipLostError("lost")
    assert error.session_id == "lost"
    assert isinstance(error, Exception)
    assert not isinstance(error, SessionAlreadyOwnedError)
    assert not isinstance(error, SessionHandleClosedError)
    assert str(error) == (
        'session "lost": write ownership was lost; close this handle and reopen'
    )


def test_format_errors_are_re_exported_not_redefined() -> None:
    """Verifies the two format errors are the same objects, not copies."""
    assert (
        SessionPersistenceCorruptionError
        is format_module.SessionPersistenceCorruptionError
    )
    assert SessionFormatUnsupportedError is format_module.SessionFormatUnsupportedError


def test_package_exports_match_the_canonical_module() -> None:
    """Verifies the session package re-exports every new contract name."""
    assert session_pkg.SessionAlreadyExistsError is SessionAlreadyExistsError
    assert session_pkg.SessionAlreadyOwnedError is SessionAlreadyOwnedError
    assert session_pkg.SessionHandle is SessionHandle
    assert session_pkg.SessionHandleClosedError is SessionHandleClosedError
    assert session_pkg.SessionOwnershipLostError is SessionOwnershipLostError
    assert session_pkg.SessionPersistence is SessionPersistence
    assert (
        session_pkg.SessionPersistenceNotFoundError is SessionPersistenceNotFoundError
    )
    assert session_pkg.SessionPersistenceSnapshot is SessionPersistenceSnapshot
    assert session_pkg.SessionReadOnlyError is SessionReadOnlyError


def test_abstract_bases_cannot_be_instantiated() -> None:
    """Verifies both ABCs refuse direct instantiation."""
    with pytest.raises(TypeError):
        SessionHandle()
    with pytest.raises(TypeError):
        SessionPersistence()


def test_snapshot_defaults_and_frozen() -> None:
    """Verifies snapshot defaults and that snapshots are immutable values."""
    snapshot = SessionPersistenceSnapshot(
        header=_header("s"), format_status="current", revision="rev-1"
    )
    assert snapshot.header == _header("s")
    assert snapshot.format_status == "current"
    assert snapshot.revision == "rev-1"
    assert snapshot.event_count is None
    assert snapshot.size_bytes is None
    assert snapshot.location is None
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = "rev-2"  # type: ignore[misc]

    located = SessionPersistenceSnapshot(
        header=_header("s"),
        format_status="migration-required",
        revision="rev-2",
        event_count=3,
        size_bytes=128,
        location=Path("/sessions/s/session.jsonl"),
    )
    assert located.event_count == 3
    assert located.size_bytes == 128
    assert located.location == Path("/sessions/s/session.jsonl")


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_returns_unmaterialized_write_handle() -> None:
    """Verifies create returns a write handle with the fixed header."""
    persistence = _MemoryPersistence()
    handle = persistence.create(_header("s1"))

    assert handle.id == "s1"
    assert handle.header == _header("s1")
    assert handle.inherited_event_count == 0
    assert handle.access == "write"
    assert not handle.closed


def test_create_duplicate_id_raises_already_exists() -> None:
    """Verifies create on an existing id refuses, pending or on-disk."""
    persistence = _MemoryPersistence()
    persistence.create(_header("s1"))

    with pytest.raises(SessionAlreadyExistsError) as excinfo:
        persistence.create(_header("s1"))
    assert excinfo.value.session_id == "s1"


def test_create_validates_header() -> None:
    """Verifies create refuses a header that fails header validation."""
    persistence = _MemoryPersistence()

    with pytest.raises(SessionFormatError):
        persistence.create({"version": 4, "createdAt": 0})
    with pytest.raises(SessionFormatError):
        persistence.create({**_header("bad"), "version": -1})


def test_create_seeded_requires_inherited_count() -> None:
    """Verifies a seeded header requires an int >= 0 inherited count."""
    persistence = _MemoryPersistence()

    with pytest.raises(ValueError, match="inherited_event_count"):
        persistence.create(_header("seeded", seeded=True))
    with pytest.raises(ValueError, match="inherited_event_count"):
        persistence.create(_header("seeded", seeded=True), -1)

    handle = persistence.create(_header("seeded", seeded=True), 3)
    assert handle.inherited_event_count == 3


def test_create_unseeded_requires_no_inherited_count() -> None:
    """Verifies an unseeded header requires None or 0 inherited count."""
    persistence = _MemoryPersistence()

    with pytest.raises(ValueError, match="inherited_event_count"):
        persistence.create(_header("plain"), 2)

    assert persistence.create(_header("plain")).inherited_event_count == 0
    assert persistence.create(_header("plain-zero"), 0).inherited_event_count == 0


# ---------------------------------------------------------------------------
# open
# ---------------------------------------------------------------------------


def test_open_missing_id_raises_not_found() -> None:
    """Verifies open of a missing id refuses with the session id."""
    persistence = _MemoryPersistence()

    with pytest.raises(SessionPersistenceNotFoundError) as excinfo:
        persistence.open("missing", "read")
    assert excinfo.value.session_id == "missing"

    with pytest.raises(SessionPersistenceNotFoundError):
        persistence.open("missing", "write")


def test_open_read_works_while_write_is_held() -> None:
    """Verifies read opens never take ownership, even under a writer."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    reader_one = persistence.open("s1", "read")
    reader_two = persistence.open("s1", "read")

    assert reader_one.access == "read"
    assert reader_two.access == "read"
    assert not writer.closed
    assert reader_one.header == _header("s1")


def test_open_write_while_owned_raises_already_owned() -> None:
    """Verifies a second write open is refused while the first is active."""
    persistence = _MemoryPersistence()
    persistence.create(_header("s1"))

    with pytest.raises(SessionAlreadyOwnedError) as excinfo:
        persistence.open("s1", "write")
    assert excinfo.value.session_id == "s1"


def test_open_write_succeeds_after_owner_closes() -> None:
    """Verifies closing the write handle releases write ownership."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    writer.close()

    reopened = persistence.open("s1", "write")
    assert reopened.access == "write"
    assert reopened.header == _header("s1")


def test_open_returns_header_fixed_at_create() -> None:
    """Verifies an opened handle carries the header fixed at create."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    writer.append([_event("s1", 0)])
    writer.close()

    reader = persistence.open("s1", "read")
    assert reader.header == _header("s1")
    assert reader.inherited_event_count == 0


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_returns_detached_events_in_order() -> None:
    """Verifies read returns the stored prefix as detached copies."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    writer.append([_event("s1", 0), _event("s1", 1, "turn/end")])

    event_state, events = writer.read()
    assert event_state == "detached"
    assert [event.seq for event in events] == [0, 1]

    events[0].payload["tampered"] = True
    _, reread = writer.read()
    assert "tampered" not in reread[0].payload


def test_read_slices_by_offset_and_length() -> None:
    """Verifies offset/length slicing and the past-the-end empty read."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    writer.append([_event("s1", seq) for seq in range(3)])

    _, all_events = writer.read()
    assert [event.seq for event in all_events] == [0, 1, 2]

    _, tail = writer.read(1)
    assert [event.seq for event in tail] == [1, 2]

    _, one = writer.read(1, 1)
    assert [event.seq for event in one] == [1]

    _, empty = writer.read(3)
    assert empty == []
    _, empty_length = writer.read(0, 0)
    assert empty_length == []


def test_read_on_seeded_handle_uses_seq_offsets() -> None:
    """Verifies read offsets are seqs, not positions, for seeded logs."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("seeded", seeded=True), 3)
    writer.append([_event("seeded", 3), _event("seeded", 4)])

    _, from_zero = writer.read(0)
    assert [event.seq for event in from_zero] == [3, 4]
    _, from_four = writer.read(4)
    assert [event.seq for event in from_four] == [4]
    _, past_end = writer.read(5)
    assert past_end == []


def test_read_observes_other_handles_appends() -> None:
    """Verifies a read started after an append observes at least that prefix."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    reader = persistence.open("s1", "read")

    _, before = reader.read()
    assert before == []

    writer.append([_event("s1", 0)])
    _, after = reader.read()
    assert [event.seq for event in after] == [0]


def test_reads_are_monotonic() -> None:
    """Verifies repeated reads never observe an older state than before."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    writer.append([_event("s1", 0)])

    _, first = writer.read()
    writer.append([_event("s1", 1)])
    _, second = writer.read()
    assert [event.seq for event in first] == [0]
    assert [event.seq for event in second] == [0, 1]
    assert second[:1] == first


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


def test_append_continues_the_log_and_write_handle_reads_own_appends() -> None:
    """Verifies appends land contiguously and are visible to the writer."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))

    writer.append([_event("s1", 0)])
    writer.append([_event("s1", 1), _event("s1", 2)])
    _, events = writer.read()
    assert [event.seq for event in events] == [0, 1, 2]


def test_append_empty_batch_is_a_no_op() -> None:
    """Verifies an empty append changes nothing and raises nothing."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    writer.append([])

    _, events = writer.read()
    assert events == []
    writer.append([_event("s1", 0)])
    _, events = writer.read()
    assert [event.seq for event in events] == [0]


def test_append_gap_raises_corruption() -> None:
    """Verifies a batch that skips a seq is refused as corruption."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    writer.append([_event("s1", 0)])

    with pytest.raises(SessionPersistenceCorruptionError, match="expected 1"):
        writer.append([_event("s1", 2)])
    _, events = writer.read()
    assert [event.seq for event in events] == [0]


def test_append_duplicate_seq_raises_corruption() -> None:
    """Verifies a batch repeating the last seq is refused as corruption."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    writer.append([_event("s1", 0)])

    with pytest.raises(SessionPersistenceCorruptionError, match="expected 1"):
        writer.append([_event("s1", 0)])


def test_append_wrong_first_seq_on_seeded_handle_raises_corruption() -> None:
    """Verifies a seeded log's first append must start at the inherited cut."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("seeded", seeded=True), 3)

    with pytest.raises(SessionPersistenceCorruptionError, match="expected 3"):
        writer.append([_event("seeded", 0)])


@pytest.mark.parametrize("operation", ["append", "flush"])
def test_read_handle_rejects_mutations(operation: str) -> None:
    """Verifies append/flush on a read handle raise SessionReadOnlyError."""
    persistence = _MemoryPersistence()
    persistence.create(_header("s1"))
    reader = persistence.open("s1", "read")

    with pytest.raises(SessionReadOnlyError) as excinfo:
        if operation == "append":
            reader.append([_event("s1", 0)])
        else:
            reader.flush()
    assert excinfo.value.session_id == "s1"
    assert excinfo.value.operation == operation


# ---------------------------------------------------------------------------
# flush (handle and service-wide)
# ---------------------------------------------------------------------------


def test_handle_flush_marks_durability() -> None:
    """Verifies a write handle flush succeeds and is observable."""
    persistence = _MemoryPersistence()
    handle = persistence.create(_header("s1"))
    assert isinstance(handle, _MemoryHandle)

    handle.flush()
    assert handle.flush_count == 1


def test_service_flush_flushes_every_active_write_handle() -> None:
    """Verifies the service-wide barrier flushes all owned write handles."""
    persistence = _MemoryPersistence()
    first = persistence.create(_header("s1"))
    second = persistence.create(_header("s2"))
    assert isinstance(first, _MemoryHandle)
    assert isinstance(second, _MemoryHandle)

    persistence.flush()
    assert first.flush_count == 1
    assert second.flush_count == 1


def test_service_flush_with_no_write_handles_succeeds() -> None:
    """Verifies the barrier is a no-op when nothing is owned."""
    persistence = _MemoryPersistence()
    persistence.create(_header("s1")).close()
    persistence.flush()


def test_service_flush_aggregates_failures_and_flushes_the_rest() -> None:
    """Verifies one failing flush does not stop the others from flushing."""
    persistence = _MemoryPersistence()
    failing = persistence.create(_header("s1"))
    healthy = persistence.create(_header("s2"))
    assert isinstance(failing, _MemoryHandle)
    assert isinstance(healthy, _MemoryHandle)
    failing.fail_flush = True

    with pytest.raises(ExceptionGroup) as excinfo:
        persistence.flush()
    assert "s1" in str(excinfo.value)
    assert "s2" not in str(excinfo.value)
    assert len(excinfo.value.exceptions) == 1
    assert isinstance(excinfo.value.exceptions[0], RuntimeError)
    assert healthy.flush_count == 1
    assert failing.flush_count == 0


def test_service_flush_names_every_failed_session() -> None:
    """Verifies the group message names each failed session id."""
    persistence = _MemoryPersistence()
    first = persistence.create(_header("s1"))
    second = persistence.create(_header("s2"))
    assert isinstance(first, _MemoryHandle)
    assert isinstance(second, _MemoryHandle)
    first.fail_flush = True
    second.fail_flush = True

    with pytest.raises(ExceptionGroup) as excinfo:
        persistence.flush()
    assert "s1" in str(excinfo.value)
    assert "s2" in str(excinfo.value)
    assert len(excinfo.value.exceptions) == 2


def test_service_flush_treats_a_closed_failing_handle_as_flushed() -> None:
    """Verifies a concurrently closed handle counts as flushed."""
    persistence = _MemoryPersistence()
    failing = persistence.create(_header("s1"))
    assert isinstance(failing, _MemoryHandle)
    failing.fail_flush = True
    failing.close()

    persistence.flush()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", ["read", "append", "flush"])
def test_closed_handle_rejects_every_operation(operation: str) -> None:
    """Verifies every operation on a closed handle raises HandleClosed."""
    persistence = _MemoryPersistence()
    handle = persistence.create(_header("s1"))
    handle.close()
    assert handle.closed

    with pytest.raises(SessionHandleClosedError) as excinfo:
        if operation == "read":
            handle.read()
        elif operation == "append":
            handle.append([])
        else:
            handle.flush()
    assert excinfo.value.session_id == "s1"
    assert excinfo.value.operation == operation


def test_close_is_idempotent() -> None:
    """Verifies double-close succeeds, on write and read handles alike."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))
    reader = persistence.open("s1", "read")

    writer.close()
    writer.close()
    reader.close()
    reader.close()
    assert writer.closed
    assert reader.closed


def test_closed_read_handle_rejects_read() -> None:
    """Verifies a closed read handle rejects reads like any closed handle."""
    persistence = _MemoryPersistence()
    persistence.create(_header("s1"))
    reader = persistence.open("s1", "read")
    reader.close()

    with pytest.raises(SessionHandleClosedError) as excinfo:
        reader.read()
    assert excinfo.value.operation == "read"


# ---------------------------------------------------------------------------
# stat / list
# ---------------------------------------------------------------------------


def test_stat_returns_none_for_missing_session() -> None:
    """Verifies stat of a missing id returns None instead of raising."""
    persistence = _MemoryPersistence()
    assert persistence.stat("missing") is None


def test_stat_observes_pending_session() -> None:
    """Verifies a created-but-unmaterialized session is already observable."""
    persistence = _MemoryPersistence()
    persistence.create(_header("s1"))

    snapshot = persistence.stat("s1")
    assert snapshot is not None
    assert snapshot.header == _header("s1")
    assert snapshot.format_status == "current"
    assert snapshot.event_count == 0
    assert snapshot.size_bytes is None
    assert snapshot.location is None


def test_stat_reports_migration_required_for_historical_header() -> None:
    """Verifies the header-only classification reaches the snapshot."""
    persistence = _MemoryPersistence()
    persistence.create(_header("old", version=3))

    snapshot = persistence.stat("old")
    assert snapshot is not None
    assert snapshot.format_status == "migration-required"


def test_stat_revision_changes_on_append_not_on_ownership_churn() -> None:
    """Verifies the revision tracks log change, not ownership churn."""
    persistence = _MemoryPersistence()
    writer = persistence.create(_header("s1"))

    first = persistence.stat("s1")
    assert first is not None
    writer.append([_event("s1", 0)])
    second = persistence.stat("s1")
    assert second is not None
    assert second.revision != first.revision
    assert second.event_count == 1

    writer.close()
    persistence.open("s1", "write").close()
    third = persistence.stat("s1")
    assert third is not None
    assert third.revision == second.revision


def test_stat_header_is_a_detached_copy() -> None:
    """Verifies mutating a returned header cannot corrupt the stored one."""
    persistence = _MemoryPersistence()
    handle = persistence.create(_header("s1"))

    handle.header["id"] = "tampered"
    snapshot = persistence.stat("s1")
    assert snapshot is not None
    assert snapshot.header["id"] == "s1"

    snapshot.header["id"] = "tampered"
    again = persistence.stat("s1")
    assert again is not None
    assert again.header["id"] == "s1"


def test_list_returns_every_stored_session() -> None:
    """Verifies list reports one snapshot per stored session."""
    persistence = _MemoryPersistence()
    assert persistence.list() == []

    persistence.create(_header("s1"))
    persistence.create(_header("s2"))
    snapshots = persistence.list()
    assert {snapshot.header["id"] for snapshot in snapshots} == {"s1", "s2"}
    assert all(snapshot.format_status == "current" for snapshot in snapshots)
