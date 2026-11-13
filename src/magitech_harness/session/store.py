"""Session storage and deterministic model history projection (derive_messages).

Ownership split: a Session is purely in-memory and never touches the
filesystem. Durable writes flow from Session.append through
``session/event`` publication to the live tracker, which batches them
onto the write handle the service opened for that session. The service
layer owns the persistence backend and the write handles; the store
owns live-session membership (prepare / enter / announce).
``Session.load`` is a strict file-level restore of one stored log: the
stored header is authoritative for the session id, a missing or empty
file is not found, and a migrated log refuses append until it is
republished through the backend.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from magitech_harness.session.events import (
    CORE_EVENT_TYPES,
    KNOWN_SESSION_EVENT_TYPES,
    OWNER_EVENT_TYPES,
    SURFACE_EVENT_TYPES,
    EventType,
    SessionEvent,
    is_lossless_json,
    is_plugin_record_type,
    snapshot_json_value,
)
from magitech_harness.session.fork import build_fork_seed
from magitech_harness.session.format import (
    CURRENT_FORMAT_VERSION,
    SESSION_FORMAT_CATALOG,
    SessionFormatError,
    SessionFormatUnsupportedError,
    SessionFormatUnsupportedMigrationError,
    SessionPersistenceCorruptionError,
    assert_v4_row_admission,
    create_session_format_catalog_with_children,
    decode_envelope_row,
    is_json_object,
    require_count,
    require_safe_integer,
    session_format_version_refusal,
    snapshot_header,
)
from magitech_harness.session.persistence import SessionPersistenceNotFoundError
from magitech_harness.session.repair import (
    interrupted_turn_closers,
    open_turn_closers,
)
from magitech_harness.session.request_header import fold_request_header
from magitech_harness.session.surface import (
    MessageProjection,
    SurfaceError,
    SurfaceManager,
    derive_event_message,
    fold_surface,
    validate_session_event_data,
)
from magitech_harness.session.tool_history import ToolHistoryProjection

logger = logging.getLogger("harness.session")


def derive_messages(
    events: list[SessionEvent],
    projections: list[MessageProjection] | None = None,
) -> list[dict[str, Any]]:
    """Project current model-visible messages from a session log.

    Pure function over the surface fold: one derived message per
    surface node that projects non-None, in surface order. No merging,
    no synthesis, no slicing. The returned messages are detached
    copies of the stored messages; the caller may mutate them freely.
    """
    fold = fold_surface(events, projections)
    messages: list[dict[str, Any]] = []
    for seq in fold.nodes:
        message = derive_event_message(events[seq], fold.projected_messages)
        if message is not None:
            messages.append(copy.deepcopy(message))
    return messages


# ---------------------------------------------------------------------------
# Plugin records
# ---------------------------------------------------------------------------


class PluginRecord:
    """One committed experimental plugin record, read from the log."""

    def __init__(self, type: str, seq: int, time: float, data: Any) -> None:
        self.type = type
        self.seq = seq
        self.time = time
        self.data = data


def plugin_record_of(event: SessionEvent) -> PluginRecord | None:
    """Read one committed event as a plugin record, or ``None``.

    A record is an ignorable event whose type follows the ``plugin:``
    grammar; the owner validates ``data`` before use.
    """
    if event.ignorable is not True or not is_plugin_record_type(event.type):
        return None
    return PluginRecord(
        type=event.type, seq=event.seq, time=event.timestamp, data=event.payload
    )


def append_plugin_record(session: Session, type: str, data: dict[str, Any]) -> int:
    """Commit one plugin record to a session; returns its sequence number."""
    return session.append_plugin_record(type, data)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class _StoreAttachment:
    """One store-owned session's publication state (store-private)."""

    def __init__(self, store: SessionStore, status: str) -> None:
        self.store = store
        self.status = status
        self.appending = False
        self.detach_requested = False
        self.session: Session | None = None
        self.detach: Callable[[], None] = lambda: None


def _snapshot_event(source: SessionEvent) -> SessionEvent | None:
    """Deep-copy one event through the lossless-JSON domain, or ``None``."""
    record = {
        "type": source.type,
        "seq": source.seq,
        "time": source.timestamp,
        "data": source.payload,
        "ignorable": source.ignorable,
        "surfaceOp": source.surface_op,
        "sourceEventSeqs": source.source_event_seqs,
    }
    copied = snapshot_json_value(record)
    if copied is None:
        return None
    return SessionEvent(
        id=source.id,
        session_id=source.session_id,
        type=copied["type"],
        timestamp=copied["time"],
        payload=copied["data"],
        ignorable=copied["ignorable"],
        seq=copied["seq"],
        surface_op=copied.get("surfaceOp"),
        source_event_seqs=copied.get("sourceEventSeqs"),
    )


def _event_is_lossless_json(event: SessionEvent) -> bool:
    """Whether an event round-trips through the lossless-JSON domain."""
    return is_lossless_json(
        {
            "type": event.type,
            "seq": event.seq,
            "time": event.timestamp,
            "data": event.payload,
            "ignorable": event.ignorable,
            "surfaceOp": event.surface_op,
            "sourceEventSeqs": event.source_event_seqs,
        }
    )


class Session:
    """A durable, append-only conversation session."""

    def __init__(
        self,
        session_id: str,
        *,
        seed: list[SessionEvent] | None = None,
        header: dict[str, Any] | None = None,
        inherited_event_count: int | None = None,
        event_state: str = "snapshot",
        projections: list[MessageProjection] | None = None,
    ) -> None:
        self.session_id = session_id
        self.events: list[SessionEvent] = []
        self._surface_manager = SurfaceManager(self.events, projections)
        # Incremental request-fold watermarks: each fold consumes only
        # events appended since its last read, starting from 0 so a
        # restored log (including inherited events) folds exactly like
        # the offline fold on first read.
        self._header_fold: dict[str, Any] | None = None
        self._header_fold_seq: int = 0
        self._context_fold: dict[str, Any] | None = None
        self._context_fold_seq: int = 0
        self._tool_history_projection = ToolHistoryProjection()
        self._tool_history_seq: int = 0
        # Cached derived messages, invalidated by a surface
        # content-generation bump (a replace or projection change).
        self._derived: list[dict[str, Any]] = []
        self._derived_nodes: int = 0
        self._derived_generation: int = 0
        self._appending = False
        self._detached = False
        self._attachment: _StoreAttachment | None = None
        self._publish_hook: Callable[[SessionEvent], None] | None = None
        # Retained v4 logical header (set on load, built on first append).
        self._header: dict[str, Any] | None = None
        # True when the log this session was restored from is already
        # current-version v4. A session loaded from a migrated log
        # refuses append until the log is republished as current-version
        # v4 through the backend; fresh sessions are current by
        # construction.
        self._on_disk_current: bool = True
        if header is not None:
            self._header = header
        if self._header is None:
            self._header = {
                "version": CURRENT_FORMAT_VERSION,
                "id": session_id,
                "createdAt": int(time.time() * 1000),
                "isSeeded": False,
                "delegationDepth": 0,
            }
        snapshot_header(self._header)
        if self._header.get("origin") not in (None, "subagent"):
            raise SessionFormatError('Session header origin must be "subagent"')
        if isinstance(self._header.get("cwd"), str) and not os.path.isabs(
            self._header["cwd"]
        ):
            raise SessionFormatError("Session header cwd must be absolute")
        self._is_seeded = self._header["isSeeded"] is True
        if seed is None and self._is_seeded:
            raise ValueError("seeded session requires an explicit constructor seed")
        if self._is_seeded and inherited_event_count is None:
            raise ValueError("seeded session requires an inherited event count")
        inherited = inherited_event_count if inherited_event_count is not None else 0
        if not self._is_seeded and inherited != 0:
            raise ValueError("unseeded session inherited event count must be 0")
        if seed is not None:
            for index, source in enumerate(seed):
                if event_state == "snapshot":
                    candidate = _snapshot_event(source)
                    if candidate is None:
                        raise ValueError(
                            f"seed event at index {index} is not losslessly "
                            "JSON-serializable"
                        )
                else:
                    if not _event_is_lossless_json(source):
                        raise ValueError(
                            f"seed event at index {index} is not losslessly "
                            "JSON-serializable"
                        )
                    candidate = source
                if candidate.seq != index:
                    raise ValueError(
                        f"seed event at index {index} has seq {candidate.seq} "
                        f"(expected {index}); seed must be contiguous from 0"
                    )
                # A seed is a persistence/replay boundary: the same
                # event-data validation `append` enforces applies here,
                # so a bad seed cannot construct a live log that no
                # persistence backend could store.
                try:
                    validate_session_event_data(
                        candidate, f"seed {candidate.type} at index {index}"
                    )
                except SurfaceError as error:
                    raise ValueError(
                        f"invalid seed event at index {index}: {error}"
                    ) from error
                try:
                    self._surface_manager.validate_next(candidate)
                except SurfaceError as error:
                    raise ValueError(
                        f"invalid seed event at index {index}: {error}"
                    ) from error
                self.events.append(candidate)
        self.first_live_seq = len(self.events)
        if inherited > len(self.events):
            raise ValueError("session inherited event count exceeds its event log")
        marker = (
            self.events[inherited]
            if seed is not None and inherited < len(self.events)
            else None
        )
        marked = (
            marker is not None
            and marker.type == "session/end-seed"
            and marker.payload.get("inherited") is True
        )
        if (
            event_state == "snapshot"
            and self._is_seeded
            and inherited != len(self.events)
            and not marked
        ):
            raise ValueError(
                "seeded session constructor seed must equal its inherited prefix "
                "or mark its inherited cut"
            )
        if marked and any(
            event.type == "session/end-seed" and event.payload.get("inherited") is True
            for event in self.events[inherited + 1 :]
        ):
            raise ValueError(
                "session inherited event count must identify the final inherited marker"
            )
        self.inherited_event_count = inherited
        self.first_lifecycle_seq = (
            inherited
            if event_state == "snapshot" and self._is_seeded
            else self.first_live_seq
        )
        # A fresh seeded child always owns one tagged marker at its
        # inherited cut, even when the copied prefix already ends in an
        # ancestor marker. Otherwise, if the log's last event is not
        # already ``session/end-seed``, an untagged marker is appended.
        # Restore adopts the stored log exactly and appends no marker.
        if seed is not None and event_state == "snapshot":
            if self._is_seeded and not marked:
                self._commit_marker({"inherited": True})
            elif not self._is_seeded and (
                not self.events or self.events[-1].type != "session/end-seed"
            ):
                self._commit_marker({})

    @property
    def header(self) -> dict[str, Any]:
        """The session's detached creation metadata (never in the log)."""
        assert self._header is not None
        return self._header

    @property
    def id(self) -> str:
        """The session identity, derived from its header's single copy."""
        session_id = self.header["id"]
        assert isinstance(session_id, str)
        return session_id

    @property
    def seq(self) -> int:
        """The next event's sequence number — always the log length."""
        return len(self.events)

    def is_own_seq(self, seq: int) -> bool:
        """Whether one event position is outside the fork-inherited prefix."""
        return self.inherited_event_count <= seq < len(self.events)

    @classmethod
    def create(
        cls,
        session_id: str,
        seed: list[SessionEvent] | None = None,
        header: dict[str, Any] | None = None,
        inherited_event_count: int | None = None,
        projections: list[MessageProjection] | None = None,
    ) -> Session:
        """Create a detached session, snapshotting borrowed seed events."""
        return cls(
            session_id,
            seed=seed,
            header=header,
            inherited_event_count=inherited_event_count,
            event_state="snapshot",
            projections=projections,
        )

    @classmethod
    def from_restore(
        cls,
        session_id: str,
        seed: list[SessionEvent],
        header: dict[str, Any],
        inherited_event_count: int,
        event_state: str,
        projections: list[MessageProjection] | None = None,
    ) -> Session:
        """Restore a detached session, adopting independently owned events.

        ``event_state`` is ``'detached'`` or ``'shared-frozen'``; the seed
        is adopted without copying. The stored log is kept exactly as
        restored — no constructor marker is appended.
        """
        if event_state not in ("detached", "shared-frozen"):
            raise ValueError(
                f"unknown restore event state {event_state!r}; expected "
                "'detached' or 'shared-frozen'"
            )
        return cls(
            session_id,
            seed=seed,
            header=header,
            inherited_event_count=inherited_event_count,
            event_state=event_state,
            projections=projections,
        )

    def _commit_marker(self, data: dict[str, Any]) -> None:
        """Append a constructor ``session/end-seed`` marker in memory only."""
        candidate = SessionEvent(
            id=f"{self.session_id}:{len(self.events)}",
            session_id=self.session_id,
            type="session/end-seed",
            timestamp=time.time(),
            payload=data,
            seq=len(self.events),
        )
        self._surface_manager.validate_next(candidate)
        self.events.append(candidate)

    def _publication_entry(self) -> _StoreAttachment | None:
        """Require announced membership for store-owned sessions."""
        if self._detached:
            raise ValueError(
                f'session "{self.session_id}" cannot append outside a SessionStore'
            )
        attachment = self._attachment
        if attachment is not None and attachment.status != "complete":
            raise ValueError(
                f'session "{self.session_id}" cannot append before session/created '
                "completes"
            )
        if self._appending:
            raise ValueError(
                "session append cannot reenter while another append is being published"
            )
        return attachment

    def _require_appendable(self) -> None:
        """Refuse appends to a session restored from a migrated log.

        A migrated log is republished as current-version v4 through the
        backend before it accepts new events; appending in memory would
        leave the restored session ahead of a log it can no longer
        extend.
        """
        if not self._on_disk_current:
            raise SessionFormatUnsupportedError(
                "cannot append to a migrated session until it is republished "
                f"as current-version v4 (session {self.session_id!r})"
            )

    def append(
        self,
        event_type: EventType | str,
        data: dict[str, Any],
        *,
        surface_op: str | dict[str, Any] | None = None,
        source_event_seqs: list[int] | None = None,
        ignorable: bool = False,
    ) -> SessionEvent:
        """Append one typed event to the log; returns the logged event.

        Surface types require a surface operation; non-surface types
        forbid surface metadata. Data is snapshotted before commit, so
        the caller can never mutate logged data afterwards. The event is
        assigned ``seq = log.length`` and ``time`` at append.
        """
        type_value = (
            event_type.value if isinstance(event_type, EventType) else str(event_type)
        )
        self._require_appendable()
        entry = self._publication_entry()
        if type_value not in KNOWN_SESSION_EVENT_TYPES:
            raise ValueError(f'unknown session event type "{type_value}"')
        if type_value in SURFACE_EVENT_TYPES and surface_op is None:
            raise ValueError(
                f'session event "{type_value}" is surface-eligible and requires '
                "a surfaceOp marker"
            )
        if type_value not in SURFACE_EVENT_TYPES and (
            surface_op is not None or source_event_seqs is not None
        ):
            raise ValueError(
                f'session event "{type_value}" is not surface-eligible and cannot '
                "carry surface metadata"
            )
        data_snapshot = snapshot_json_value(data)
        if data_snapshot is None:
            raise ValueError(
                f'session event "{type_value}" carries non-JSON-serializable data'
            )
        metadata_snapshot = snapshot_json_value(
            {
                **(
                    {"sourceEventSeqs": source_event_seqs}
                    if source_event_seqs is not None
                    else {}
                ),
                **({"surfaceOp": surface_op} if surface_op is not None else {}),
            }
        )
        if metadata_snapshot is None:
            raise ValueError(
                f'session event "{type_value}" carries non-JSON-serializable '
                "surface metadata"
            )
        candidate = SessionEvent(
            id=f"{self.session_id}:{len(self.events)}",
            session_id=self.session_id,
            type=type_value,
            timestamp=time.time(),
            payload=data_snapshot,
            ignorable=ignorable,
            seq=len(self.events),
            surface_op=metadata_snapshot.get("surfaceOp"),
            source_event_seqs=metadata_snapshot.get("sourceEventSeqs"),
        )
        validate_session_event_data(
            candidate,
            f'session event "{type_value}" at seq {candidate.seq}',
            strict=True,
        )
        # The append contract is the admission contract: an event that
        # could not be reloaded must not be appended. The row is the
        # exact physical shape the log persists.
        row: dict[str, Any] = {
            "type": candidate.type,
            "seq": candidate.seq,
            "time": int(candidate.timestamp * 1000),
            "data": data_snapshot,
            **({"ignorable": True} if candidate.ignorable else {}),
            **(
                {"surfaceOp": candidate.surface_op}
                if candidate.surface_op is not None
                else {}
            ),
            **(
                {"sourceEventSeqs": candidate.source_event_seqs}
                if candidate.source_event_seqs is not None
                else {}
            ),
        }
        try:
            assert_v4_row_admission(row, KNOWN_SESSION_EVENT_TYPES)
        except (SessionFormatError, SessionFormatUnsupportedMigrationError) as error:
            raise ValueError(
                f'session event "{type_value}" at seq {candidate.seq} '
                f"fails v4 admission: {error}"
            ) from error
        self._commit(candidate, entry)
        return candidate

    def append_plugin_record(self, type: str, data: dict[str, Any]) -> int:
        """Commit one ``plugin:`` record; returns its sequence number.

        Records are always ``ignorable: true`` and never members of the
        known event vocabulary; the owner validates the payload.
        """
        self._require_appendable()
        entry = self._publication_entry()
        if not is_plugin_record_type(type):
            raise ValueError(f'plugin record type "{type}" is outside the grammar')
        data_snapshot = snapshot_json_value(data)
        if data_snapshot is None:
            raise ValueError(
                f'plugin record "{type}" carries non-JSON-serializable data'
            )
        candidate = SessionEvent(
            id=f"{self.session_id}:{len(self.events)}",
            session_id=self.session_id,
            type=type,
            timestamp=time.time(),
            payload=data_snapshot,
            ignorable=True,
            seq=len(self.events),
        )
        self._commit(candidate, entry)
        return candidate.seq

    def _commit(self, candidate: SessionEvent, entry: _StoreAttachment | None) -> None:
        """Accept one candidate at the next sequence number and publish it.

        The surface transition is planned before the event enters the
        log, so a failure cannot partially mutate the surface. Once the
        event enters the log the append is committed: observer failures
        are contained and do not change the return value.
        """
        self._surface_manager.validate_next(candidate)
        if entry is not None:
            entry.appending = True
        self._appending = True
        try:
            self.events.append(candidate)
            if self._publish_hook is not None:
                self._publish_hook(candidate)
        finally:
            self._appending = False
            if entry is not None:
                entry.appending = False
                if entry.detach_requested and entry.status != "dispatching":
                    entry.detach()

    def derive_messages(self) -> list[dict[str, Any]]:
        """Project current model-visible messages from the event log.

        Cached on the live session: pure tail growth costs O(new
        nodes); a content-generation bump rebuilds from scratch. The
        returned messages are read-only cached copies: callers must
        not mutate them.
        """
        nodes = self._surface_manager.nodes
        generation = self._surface_manager.content_generation
        if generation != self._derived_generation:
            self._derived = []
            self._derived_nodes = 0
            self._derived_generation = generation
        for seq in nodes[self._derived_nodes :]:
            message = self._surface_manager.derive_event_message(self.events[seq])
            if message is not None:
                self._derived.append(copy.deepcopy(message))
        self._derived_nodes = len(nodes)
        return list(self._derived)

    def request_header(self) -> dict[str, Any] | None:
        """The latest folded request header, or ``None`` before any header.

        Incrementally maintained: each header event is folded once.
        The returned header is a detached copy.
        """
        self._header_fold = fold_request_header(
            self.events[self._header_fold_seq :], from_=self._header_fold
        )
        self._header_fold_seq = len(self.events)
        return (
            copy.deepcopy(self._header_fold) if self._header_fold is not None else None
        )

    def request_context(self) -> dict[str, Any] | None:
        """The latest logged request context, or ``None`` before any context.

        Incrementally maintained; the returned context is a detached copy.
        """
        for event in self.events[self._context_fold_seq :]:
            if event.type == "request/context":
                self._context_fold = copy.deepcopy(event.payload)
        self._context_fold_seq = len(self.events)
        return (
            copy.deepcopy(self._context_fold)
            if self._context_fold is not None
            else None
        )

    def tool_history(self) -> dict[str, Any]:
        """The current tool-history snapshot for request projection."""
        for event in self.events[self._tool_history_seq :]:
            self._tool_history_projection.apply(event)
        self._tool_history_seq = len(self.events)
        return self._tool_history_projection.snapshot()

    @property
    def surface_content_generation(self) -> int:
        """The surface fold's content generation (append/replace signal)."""
        return self._surface_manager.content_generation

    def surface_nodes(self) -> list[int]:
        """Current surface event sequences in model-visible order."""
        return self._surface_manager.nodes

    @classmethod
    def load(
        cls,
        storage_path: Path,
        projections: list[MessageProjection] | None = None,
    ) -> Session:
        """Load a session from one stored log file.

        Strict file-level restore. Refuses by default: a newer-version
        log, a malformed header, a corrupt row, a gapped or duplicated
        seq, or an unknown non-ignorable event raises instead of being
        silently truncated or repaired. A missing, empty, or blank-only
        file raises not-found instead of synthesizing an empty session.
        The stored header is authoritative for the session id. Unknown
        ignorable events are carried opaquely. A log holding
        ``image/offload`` needs its message projection supplied here, or
        the fold refuses the event. The returned session is detached and
        in-memory only: nothing written here reaches a durable log, and
        a migrated log refuses append until it is republished through
        the backend.
        """
        if not storage_path.exists() or storage_path.stat().st_size == 0:
            raise SessionPersistenceNotFoundError(str(storage_path))
        lines = [
            line.strip()
            for line in storage_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise SessionPersistenceNotFoundError(str(storage_path))
        header, events, on_disk_current, inherited = _load_lines(lines, storage_path)
        sess = cls.from_restore(
            header["id"],
            events,
            header,
            inherited,
            "shared-frozen",
            projections=projections,
        )
        sess._on_disk_current = on_disk_current
        return sess


def _parse_row(line: str, row_index: int, storage_path: Path) -> dict[str, Any]:
    """Parse one JSONL row; corrupt JSON is corruption, never truncation."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise SessionPersistenceCorruptionError(
            f"corrupt session row {row_index} in {storage_path}: {error}"
        ) from error
    if not isinstance(row, dict):
        raise SessionPersistenceCorruptionError(
            f"corrupt session row {row_index} in {storage_path}: "
            "row must be a JSON object"
        )
    return row


def _check_retired_shapes(event: dict[str, Any], session_id: str) -> None:
    """Refuse retired shapes hiding under a known event type."""
    event_type = event.get("type")
    data = event.get("data")
    if event_type == "request/header" and isinstance(data, dict):
        reason = data.get("reason")
        if reason == "fallback" or (
            isinstance(reason, dict) and reason.get("kind") == "fallback"
        ):
            raise SessionFormatUnsupportedError(
                f'session "{session_id}" request/header at seq {event.get("seq")} '
                'carries retired reason "fallback"'
            )
        header = data.get("header")
        if isinstance(header, dict) and "system" in header:
            raise SessionFormatUnsupportedError(
                f'session "{session_id}" request/header at seq {event.get("seq")} '
                "carries retired header.system"
            )
    if event_type in ("user/message", "system/message") and isinstance(data, dict):
        message = data if event_type == "user/message" else data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool-result"
            for block in content
        ):
            raise SessionFormatUnsupportedError(
                f'session "{session_id}" {event_type} at seq {event.get("seq")} '
                "carries a retired tool-result wrapper block"
            )


def _to_session_event(
    event: dict[str, Any], session_id: str, storage_path: Path
) -> SessionEvent:
    """Map one v4 logical event to a SessionEvent.

    Unknown non-ignorable events are refused; unknown ignorable events
    (including ``plugin:`` records) are carried opaquely, never dropped.
    Retired shapes under a known type are refused.
    """
    event_type = event.get("type")
    _check_retired_shapes(event, session_id)
    if event_type not in KNOWN_SESSION_EVENT_TYPES:
        if event.get("ignorable") is True:
            pass
        else:
            raise SessionFormatUnsupportedError(
                f"session {session_id!r} contains unknown event type "
                f"{json.dumps(event_type)}; this build cannot load it "
                f"(raw log: {storage_path})"
            )
    # Native v4 row admission runs on every loaded row, current-version
    # and migrated alike — the same admission the v4 decoder applies.
    # It runs after the retired-shape and vocabulary checks above so
    # their refusal errors are preserved.
    assert_v4_row_admission(event, KNOWN_SESSION_EVENT_TYPES)
    seq = require_count(event.get("seq"), f"session event {event_type} seq")
    data = event.get("data")
    return SessionEvent(
        id=f"{session_id}:{seq}",
        session_id=session_id,
        type=str(event_type),
        timestamp=require_safe_integer(
            event.get("time"), f"session event {event_type} time"
        )
        / 1000.0,
        payload=data if is_json_object(data) else {"value": data},
        ignorable=event.get("ignorable") is True,
        seq=seq,
        surface_op=event.get("surfaceOp"),
        source_event_seqs=event.get("sourceEventSeqs"),
    )


def _inherited_event_count(events: list[SessionEvent]) -> int:
    """Recover the inherited cut from the last tagged end-seed marker."""
    inherited = 0
    for event in events:
        if event.type == "session/end-seed" and event.payload.get("inherited") is True:
            inherited = event.seq
    return inherited


def _load_current_rows(
    lines: list[str], session_id: str, storage_path: Path
) -> tuple[list[SessionEvent], int]:
    """Envelope-only read of a current-version file's event rows.

    Seq density is enforced on the raw rows: each row's seq must equal
    its position, so gaps and duplicates are corruption.
    """
    events: list[SessionEvent] = []
    for index, line in enumerate(lines[1:], start=1):
        row = _parse_row(line, index, storage_path)
        try:
            decoded = decode_envelope_row(row, index, f"session {session_id}")
        except SessionFormatError as error:
            raise SessionPersistenceCorruptionError(str(error), error) from error
        if decoded.get("seq") != index - 1:
            raise SessionPersistenceCorruptionError(
                f"corrupt session row {index} in {storage_path}: seq gap "
                f"(expected {index - 1}, got {decoded.get('seq')})"
            )
        events.append(_to_session_event(decoded, session_id, storage_path))
    return events, _inherited_event_count(events)


def _load_migrated_rows(
    lines: list[str],
    header_value: dict[str, Any],
    session_id: str,
    storage_path: Path,
) -> tuple[list[SessionEvent], int]:
    """Restore a historical file through the catalog into v4 events."""
    catalog = create_session_format_catalog_with_children([])
    try:
        restore = catalog.create_restore(
            header_value, validation="transformed", recovery="strict"
        )
        for index, line in enumerate(lines[1:], start=1):
            restore.decode_row(_parse_row(line, index, storage_path))
        artifact = restore.finish()
    except SessionFormatUnsupportedMigrationError as error:
        raise SessionFormatUnsupportedError(
            f"{error} (raw log: {storage_path})"
        ) from error
    except SessionFormatError as error:
        raise SessionPersistenceCorruptionError(str(error), error) from error
    events: list[SessionEvent] = []
    for event in artifact.events:
        events.append(_to_session_event(event, session_id, storage_path))
    return events, artifact.inherited_event_count


def _classify_header_line(
    line: str, storage_path: Path
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Classify one stored log's header line without decoding event rows.

    Returns the physical header value, the logical header, and whether the
    log is already current-version v4. Unsupported and malformed headers
    raise the same errors Session.load raises for them.
    """
    try:
        header_value = json.loads(line)
    except json.JSONDecodeError as error:
        raise SessionPersistenceCorruptionError(
            f"corrupt session header in {storage_path}: {error}"
        ) from error
    read = SESSION_FORMAT_CATALOG.read_header(header_value)
    if read.status == "unsupported":
        if (
            read.stored_version is not None
            and read.stored_version > CURRENT_FORMAT_VERSION
        ):
            header_id = (
                header_value.get("id")
                if is_json_object(header_value)
                else storage_path.stem
            )
            message = session_format_version_refusal(
                str(header_id), read.stored_version
            )
        else:
            # Unreachable with the shipped catalog: a header that decodes
            # cannot fail a later edge validation (all edges validate the
            # same header fields), and the missing-codec path is excluded
            # at catalog construction.
            message = read.reason or "unsupported session format"  # pragma: no cover
        raise SessionFormatUnsupportedError(f"{message} (raw log: {storage_path})")
    if read.status == "malformed":
        raise SessionFormatError(read.reason or "malformed session header")
    assert read.header is not None
    return header_value, read.header, read.status == "current"


def _load_lines(
    lines: list[str], storage_path: Path
) -> tuple[dict[str, Any], list[SessionEvent], bool, int]:
    """Classify one stored log's header, then decode or migrate its rows.

    Returns the logical header, the events, whether the on-disk log is
    already current-version v4, and the inherited event count recovered
    from the tagged ``session/end-seed`` marker.
    """
    header_value, header, on_disk_current = _classify_header_line(
        lines[0], storage_path
    )
    session_id = header["id"]
    if on_disk_current:
        events, inherited = _load_current_rows(lines, session_id, storage_path)
    else:
        events, inherited = _load_migrated_rows(
            lines, header_value, session_id, storage_path
        )
    return header, events, on_disk_current, inherited


# ---------------------------------------------------------------------------
# SessionPreparation
# ---------------------------------------------------------------------------


class SessionPreparation:
    """One exact unpublished session and the provider state behind it.

    Disposal is synchronous and idempotent. Publication consumes the
    preparation, making a later disposal a no-op.
    """

    def __init__(
        self, session: Session, release: Callable[[], None] | None = None
    ) -> None:
        self.session = session
        self._release = release
        self._released = False

    @classmethod
    def create(
        cls, session: Session, release: Callable[[], None] | None = None
    ) -> SessionPreparation:
        """Wrap an unpublished session in one preparation lifetime."""
        return cls(session, release)

    def dispose(self) -> None:
        """Release provider state once when this preparation leaves its caller."""
        if self._released:
            return
        self._released = True
        if self._release is not None:
            self._release()

    def _publish(self) -> None:
        """Mark the preparation consumed by publication."""
        self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.dispose()


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionForkError(Exception):
    """A session fork was rejected; ``code`` names the rejection."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class SessionStore:
    """In-memory store of live sessions: prepare / enter / announce.

    Persistence is intentionally not in the store: the service layer
    owns the persistence backend and the write handles, and durable
    writes reach a handle through ``session/event`` publication and the
    live tracker the service installs. A session published outside
    that lifecycle persists nothing.
    """

    def __init__(self, ctx: Any | None = None) -> None:
        self._ctx = ctx
        self._sessions: dict[str, _StoreAttachment] = {}
        self._counter = 0
        self._projections: list[MessageProjection] = []
        self._detaches: dict[str, Callable[[], None]] = {}

    @property
    def message_projections(self) -> list[MessageProjection]:
        """Borrowed message-projection definitions for live replay."""
        return self._projections

    def register_message_projection(
        self, projection: MessageProjection
    ) -> Callable[[], None]:
        """Register one event interpreter; returns its disposer.

        One definition per event type. Disposing it makes sessions that
        used it refuse further derivation.
        """
        if any(item.type == projection.type for item in self._projections):
            raise ValueError(
                f'session message projection "{projection.type}" is already registered'
            )
        self._projections.append(projection)

        def disposer() -> None:
            if projection in self._projections:
                self._projections.remove(projection)

        return disposer

    def _listeners(self, event_name: str) -> list[Callable[..., Any]]:
        """The context's listeners for one session lifecycle event.

        This is the context event bus's own listener list — the same
        callbacks ``ctx.emit`` / ``ctx.parallel`` would dispatch — read
        directly so ``announce`` can veto on a synchronous throw and
        ``flush`` can settle every listener before throwing the first
        failure.
        """
        if self._ctx is None:
            return []
        bus = getattr(self._ctx, "_event_bus", None)
        if bus is None:
            return []
        try:
            listeners = bus._filter_listeners(event_name, this_arg=self._ctx)
        except AttributeError:
            return []
        return [listener.callback for listener in listeners]

    def prepare(
        self, session_id: str | None = None, options: dict[str, Any] | None = None
    ) -> SessionPreparation:
        """Build a session without entering it into the store.

        Mints ``session-<n>`` ids, validates ``meta.cwd`` absolute, and
        builds the immutable header. With ``event_state`` the seed is
        restored via :meth:`Session.from_restore`; otherwise it is
        snapshotted via :meth:`Session.create`. Writes stay disabled
        until ``announce`` completes.
        """
        opts = options or {}
        if session_id is None:
            while True:
                self._counter += 1
                candidate = f"session-{self._counter}"
                if candidate not in self._sessions:
                    break
            session_id = candidate
        elif session_id in self._sessions:
            raise ValueError(f'session "{session_id}" already exists')
        meta = opts.get("meta") or {}
        if not isinstance(meta, dict):
            raise TypeError("session meta must be an object")
        cwd = meta.get("cwd")
        if cwd is not None and (not isinstance(cwd, str) or not os.path.isabs(cwd)):
            raise ValueError("session meta.cwd must be an absolute path")
        header: dict[str, Any] = {
            "version": CURRENT_FORMAT_VERSION,
            "id": session_id,
            "createdAt": meta.get("createdAt", int(time.time() * 1000)),
            **({"cwd": cwd} if cwd is not None else {}),
            **(
                {"parentSession": meta["parentSession"]}
                if meta.get("parentSession") is not None
                else {}
            ),
            "isSeeded": meta.get("isSeeded", False),
            **({"origin": meta["origin"]} if meta.get("origin") is not None else {}),
            "delegationDepth": meta.get("delegationDepth", 0),
            **(
                {"agentPreset": meta["agentPreset"]}
                if meta.get("agentPreset") is not None
                else {}
            ),
        }
        snapshot_header(header)
        if header.get("origin") not in (None, "subagent"):
            raise SessionFormatError('Session header origin must be "subagent"')
        event_state = opts.get("event_state")
        seed = opts.get("seed")
        inherited = opts.get("inherited_event_count")
        if event_state is not None:
            if event_state not in ("detached", "shared-frozen"):
                raise ValueError(
                    f"unknown restore event state {event_state!r}; expected "
                    "'detached' or 'shared-frozen'"
                )
            if seed is None or inherited is None:
                raise ValueError(
                    "restored session preparation requires a seed and an "
                    "inherited event count"
                )
            session = Session.from_restore(
                session_id,
                seed,
                header,
                inherited,
                event_state,
                projections=self._projections,
            )
        else:
            session = Session.create(
                session_id,
                seed=seed,
                header=header,
                inherited_event_count=inherited,
                projections=self._projections,
            )
        attachment = _StoreAttachment(self, "prepared")
        attachment.session = session
        session._attachment = attachment
        session._detached = False

        def release() -> None:
            if session.session_id not in self._sessions:
                session._attachment = None
                session._detached = True

        return SessionPreparation.create(session, release=release)

    def enter(
        self, preparation_or_session: SessionPreparation | Session
    ) -> Callable[[], None]:
        """Enter a session into the store and install publication hooks.

        Re-checks the id for a duplicate: a stale prepared session must
        not overwrite a live same-id entry. Returns the detach disposer.
        Writes remain disabled until ``announce`` completes.
        """
        session = (
            preparation_or_session.session
            if isinstance(preparation_or_session, SessionPreparation)
            else preparation_or_session
        )
        if session.session_id in self._sessions:
            raise ValueError(f'session "{session.session_id}" already exists')
        attachment = session._attachment
        if attachment is not None and attachment.store is not self:
            raise ValueError(
                f'session "{session.session_id}" is already attached to a store'
            )
        if attachment is not None and attachment.status != "prepared":
            # Unreachable: any non-prepared attachment is either live in
            # this store (caught above) or was cleared on detach, which
            # also clears the attachment.
            raise ValueError(  # pragma: no cover
                f'session "{session.session_id}" is already attached to a store'
            )
        entry = _StoreAttachment(self, "pending")
        entry.session = session
        session._attachment = entry
        self._sessions[session.session_id] = entry
        if self._ctx is not None:
            ctx = self._ctx

            def publish(event: SessionEvent, session: Session = session) -> None:
                ctx.emit("session/event", session, event)

            session._publish_hook = publish

        def detach() -> None:
            if entry.status == "dispatching" or entry.appending:
                entry.detach_requested = True
                return
            self._detach_entry(entry)

        entry.detach = detach
        return detach

    def _detach_entry(self, entry: _StoreAttachment) -> None:
        """Remove one exact entered session and emit its paired disposal."""
        session = entry.session
        if session is None or self._sessions.get(session.session_id) is not entry:
            return
        del self._sessions[session.session_id]
        entry.detach_requested = False
        session._attachment = None
        session._detached = True
        session._publish_hook = None
        if entry.status != "pending" and self._ctx is not None:
            self._ctx.emit("session/disposed", session)

    def remove(self, session: Session) -> bool:
        """Detach a live session from the store; returns whether it was live."""
        entry = session._attachment
        if entry is None or entry.store is not self:
            return False
        if self._sessions.get(session.session_id) is not entry:
            return False
        entry.detach()
        return True

    def announce(self, session: Session) -> None:
        """Emit ``session/created`` exactly once for an entered session.

        Synchronous throws veto publication; async rejections are
        logged. A failed announcement leaves the session read-only and
        unretryable.
        """
        entry = self._live_entry(session)
        if entry.status != "pending":
            raise ValueError(f'session "{session.session_id}" was already announced')
        entry.status = "dispatching"
        try:
            for callback in self._listeners("session/created"):
                result = callback(session)
                if inspect.isawaitable(result):
                    self._watch_async_created(result, session.session_id)
            entry.status = "complete"
        except Exception:
            entry.status = "failed"
            raise
        finally:
            if entry.detach_requested:
                self._detach_entry(entry)

    def _watch_async_created(self, result: Any, session_id: str) -> None:
        """Observe an async ``session/created`` listener: rejection is logged."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if inspect.iscoroutine(result):
                try:
                    asyncio.run(result)
                except Exception as error:  # noqa: BLE001
                    logger.warning(
                        'session "%s": session/created listener rejected: %s',
                        session_id,
                        error,
                    )
            return
        task = loop.create_task(result)

        def _done(done_task: asyncio.Task[Any]) -> None:
            if done_task.cancelled():
                # The listener task is not exposed, so no caller can cancel
                # it; kept so a cancelled task stays silent.
                return  # pragma: no cover
            error = done_task.exception()
            if error is not None:
                logger.warning(
                    'session "%s": session/created listener rejected: %s',
                    session_id,
                    error,
                )

        task.add_done_callback(_done)

    async def flush(self, session: Session) -> bool:
        """Dispatch the ``session/flush`` durability checkpoint.

        The single durability entry point: every listener settles
        (all-settled), then the first failure is thrown. Returns whether
        at least one listener participated. Durability itself is
        performed by the service layer: the service installs the live
        tracker as a ``session/flush`` listener, so dispatch drains the
        session's buffered live events and flushes its write handle;
        the service-wide barrier flushes every handle it owns.
        """
        self._live_entry(session)
        callbacks = self._listeners("session/flush")
        if not callbacks:
            return False

        async def run(callback: Callable[..., Any]) -> BaseException | None:
            try:
                result = callback(session)
                if inspect.isawaitable(result):
                    await result
            except BaseException as error:  # noqa: BLE001
                return error
            return None

        results = await asyncio.gather(*(run(cb) for cb in callbacks))
        failures = [result for result in results if result is not None]
        if failures:
            first = failures[0]
            if isinstance(first, BaseException):
                raise first
            # run() only ever returns BaseException instances, so this
            # fallback cannot run.
            raise RuntimeError(str(first))  # pragma: no cover
        return True

    def _live_entry(self, session: Session) -> _StoreAttachment:
        entry = session._attachment
        if (
            entry is None
            or entry.store is not self
            or self._sessions.get(session.session_id) is not entry
        ):
            raise ValueError(
                f'session "{session.session_id}" is not live in this store'
            )
        return entry

    def get(self, session_id: str) -> Session | None:
        """Look up a live session."""
        entry = self._sessions.get(session_id)
        return entry.session if entry is not None else None

    def list(self) -> list[Session]:
        """All live sessions, in creation order."""
        return [
            entry.session
            for entry in self._sessions.values()
            if entry.session is not None
        ]

    def create(
        self, session_id: str | None = None, options: dict[str, Any] | None = None
    ) -> Session:
        """Create a live session: prepare, enter, then announce.

        A throwing ``session/created`` listener rolls the attach back
        instead of leaking the store entry.
        """
        preparation = self.prepare(session_id, options)
        session = preparation.session
        detach: Callable[[], None] | None = None
        try:
            detach = self.enter(preparation)
            self._detaches[session.session_id] = detach
            self.announce(session)
        except Exception:
            if detach is not None:
                detach()
            else:
                # Enter cannot fail inside create: prepare already checks
                # the duplicate-id condition enter re-checks, and the
                # attachment is freshly prepared.
                self._detaches.pop(session.session_id, None)  # pragma: no cover
            preparation.dispose()
            raise
        preparation._publish()
        return session

    def fork(
        self,
        source: Session | str,
        boundary: int | None = None,
        child_session_id: str | None = None,
    ) -> Session:
        """Create an independent live session from an exact source prefix.

        ``boundary`` is an inclusive source event seq; omitted means the
        source's current last event, and omitted on an empty source forks
        an empty child. An open tail receives synthetic tool results and
        step/turn closers with the forked cause. Closed steps and turns
        remain unchanged.
        """
        if child_session_id is not None and self.get(child_session_id) is not None:
            raise SessionForkError(
                f'session "{child_session_id}" already exists',
                "SESSION_ALREADY_EXISTS",
            )
        if isinstance(source, str):
            live = self.get(source)
            if live is None:
                raise SessionForkError(
                    f'session "{source}" was not found', "SESSION_NOT_FOUND"
                )
        elif isinstance(source, Session):
            live = source
            if self.get(live.session_id) is not live:
                raise SessionForkError(
                    f'session "{live.session_id}" is not live', "SESSION_NOT_LIVE"
                )
        else:
            raise TypeError("fork source must be a live session or its id")
        events = list(live.events)
        if boundary is None:
            resolved = events[-1].seq if events else None
        else:
            resolved = boundary
            if (
                not isinstance(resolved, int)
                or isinstance(resolved, bool)
                or resolved < 0
                or resolved >= len(events)
                or events[resolved].seq != resolved
            ):
                raise SessionForkError(
                    f"fork boundary {boundary} does not exist in session "
                    f'"{live.session_id}"',
                    "INVALID_BOUNDARY",
                )
        seed = build_fork_seed(events, resolved) if resolved is not None else []
        inherited = resolved + 1 if resolved is not None else 0
        meta: dict[str, Any] = {"isSeeded": True}
        if live.header.get("cwd") is not None:
            meta["cwd"] = live.header["cwd"]
        return self.create(
            child_session_id,
            options={
                "seed": seed,
                "inherited_event_count": inherited,
                "meta": meta,
            },
        )


__all__ = [
    "CORE_EVENT_TYPES",
    "KNOWN_SESSION_EVENT_TYPES",
    "OWNER_EVENT_TYPES",
    "MessageProjection",
    "PluginRecord",
    "Session",
    "SessionForkError",
    "SessionPreparation",
    "SessionStore",
    "append_plugin_record",
    "build_fork_seed",
    "derive_messages",
    "fold_surface",
    "interrupted_turn_closers",
    "open_turn_closers",
    "plugin_record_of",
]
