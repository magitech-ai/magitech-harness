"""
SessionService (ctx.sessions): Service for session management.

Ownership split: this service owns the persistence backend and the
write handles; the store owns live-session membership (prepare / enter /
announce). A Session is purely in-memory: its durable writes flow from
``session/event`` publication through the live tracker this service
installs on the context, which batches them onto the write handle
opened for that session. ``flush`` is the durability barrier.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anima import Context, Service

from magitech_harness.session.events import (
    SURFACE_EVENT_TYPES,
    EventType,
    SessionEvent,
)
from magitech_harness.session.format import (
    CURRENT_FORMAT_VERSION,
    snapshot_header,
)
from magitech_harness.session.persistence import (
    SessionAlreadyOwnedError,
    SessionPersistenceNotFoundError,
)
from magitech_harness.session.persistence_jsonl import (
    JsonlLiveTracker,
    JsonlSessionHandle,
    JsonlSessionPersistence,
    find_session_dir,
    generation_filename,
    resolve_current_generation,
    session_dir,
)
from magitech_harness.session.projection_registry import (
    ProjectionHandle,
    SessionProjectionRegistry,
)
from magitech_harness.session.projections import (
    SESSION_STATS_PROJECTION,
    TITLE_INPUT_PROJECTION,
    TITLE_PROJECTION,
    TURN_OUTLINE_PROJECTION,
    WORKING_DIRECTORY_PROJECTION,
)
from magitech_harness.session.store import Session, SessionStore

logger = logging.getLogger("harness.session")


class SessionService(Service):
    """Service providing durable session storage and event broadcasting.

    Live sessions are owned by a :class:`SessionStore` (prepare / enter /
    announce); this service is the shell over it that also owns the
    persistence backend, the write handles, and the live tracker. The
    backend root comes from the ``base_storage_dir`` config value; with
    no root configured the service runs in memory only — sessions work
    exactly the same, nothing is persisted, and no tracker is installed.
    """

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "sessions")
        cfg = config or {}
        root = cfg.get("base_storage_dir")
        self.base_storage_dir = Path(root) if root is not None else None
        self._store = SessionStore(ctx)
        self._projections = SessionProjectionRegistry(ctx)
        self._title_handle: ProjectionHandle[str | None] = self._projections.register(
            TITLE_PROJECTION
        )
        self._title_input_handle: ProjectionHandle[dict[str, Any]] = (
            self._projections.register(TITLE_INPUT_PROJECTION)
        )
        self._session_stats_handle: ProjectionHandle[dict[str, Any]] = (
            self._projections.register(SESSION_STATS_PROJECTION)
        )
        self._turn_outline_handle: ProjectionHandle[dict[str, Any]] = (
            self._projections.register(TURN_OUTLINE_PROJECTION)
        )
        self._working_directory_handle: ProjectionHandle[str | None] = (
            self._projections.register(WORKING_DIRECTORY_PROJECTION)
        )
        self._active_sessions: dict[str, Session] = {}
        self._handles: dict[str, JsonlSessionHandle] = {}
        self._persistence: JsonlSessionPersistence | None = None
        self._tracker: JsonlLiveTracker | None = None
        self._tracker_dispose: Callable[[], None] | None = None
        if self.base_storage_dir is not None:
            delay = cfg.get("live_batch_delay", 0.2)
            self._persistence = JsonlSessionPersistence(
                self.base_storage_dir, live_batch_delay=delay
            )
            self._tracker = JsonlLiveTracker(self._persistence)
            self._tracker_dispose = self._tracker.install(ctx)

    @property
    def store(self) -> SessionStore:
        """The live-session store this service shells over."""
        return self._store

    @property
    def projections(self) -> SessionProjectionRegistry:
        """The registry driving the five units over committed events."""
        return self._projections

    def _new_header(self, session_id: str) -> dict[str, Any]:
        """The fresh unseeded header for one new session."""
        return snapshot_header(
            {
                "version": CURRENT_FORMAT_VERSION,
                "id": session_id,
                "createdAt": int(time.time() * 1000),
                "isSeeded": False,
                "delegationDepth": 0,
            }
        )

    def create(self, session_id: str | None = None) -> Session:
        """Create a live session and register its durable write handle.

        The store entry comes first so a vetoed ``session/created``
        leaves nothing behind; the persistence handle is then created
        for the same header. Creating over an id that already exists on
        disk raises ``SessionAlreadyExistsError`` and rolls the store
        entry back.
        """
        sid = session_id or str(uuid.uuid4())
        if self._store.get(sid) is not None or sid in self._active_sessions:
            raise ValueError(f'session "{sid}" already exists')
        if self._persistence is None:
            session = self._store.create(sid)
            self._active_sessions[sid] = session
            return session
        header = self._new_header(sid)
        session = self._store.create(
            sid, options={"meta": {"createdAt": header["createdAt"]}}
        )
        try:
            handle = self._persistence.create(header)
        except Exception:
            self._store.remove(session)
            raise
        self._handles[sid] = handle
        self._active_sessions[sid] = session
        return session

    def fork(
        self,
        source: Session | str,
        boundary: int | None = None,
        child_session_id: str | None = None,
    ) -> Session:
        """Fork a live session and persist the child's inherited prefix.

        The store forks in memory first; the persistence handle is then
        created for the seeded child header and the inherited events
        are appended to it at once, so the first live event after the
        fork continues the log contiguously. A persistence failure
        rolls the store entry back.
        """
        session = self._store.fork(source, boundary, child_session_id)
        if self._persistence is None:
            self._active_sessions[session.session_id] = session
            return session
        header = dict(session.header)
        try:
            handle = self._persistence.create(
                header,
                inherited_event_count=session.inherited_event_count,
            )
        except Exception:
            self._store.remove(session)
            raise
        try:
            handle.append(list(session.events))
        except Exception:
            try:
                handle.close()
            except Exception as error:  # noqa: BLE001 - best-effort teardown
                # Close releases the claim even when its flush fails, and
                # the append error is the one that matters.
                logger.warning(
                    'session "%s": fork rollback close failed: %s',
                    session.session_id,
                    error,
                )
            assert self.base_storage_dir is not None
            directory = find_session_dir(self.base_storage_dir, session.session_id)
            if directory is not None:
                shutil.rmtree(directory)
            self._store.remove(session)
            raise
        self._handles[session.session_id] = handle
        self._active_sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._store.get(session_id) or self._active_sessions.get(session_id)

    def load(self, session_id: str) -> Session | None:
        """Load a session by ID as a live session, or ``None`` if missing.

        A cold load opens the stored log for write — publishing a
        historical generation first — and enters the restored session
        into the store, so later appends publish through the live
        tracker onto the write handle. A log whose stored header names
        a different id is refused as corruption, never adopted.
        """
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]
        if self._store.get(session_id) is not None:
            return self._store.get(session_id)
        if self._persistence is None:
            return None
        try:
            handle = self._persistence.open(session_id, "write")
        except SessionPersistenceNotFoundError:
            return None
        try:
            event_state, events = handle.read()
            session = self._enter_restored(session_id, handle, event_state, events)
        except Exception:
            handle.close()
            raise
        self._handles[session_id] = handle
        self._active_sessions[session_id] = session
        return session

    def _enter_restored(
        self,
        session_id: str,
        handle: JsonlSessionHandle,
        event_state: str,
        events: list[SessionEvent],
    ) -> Session:
        """Enter a handle's restored events into the store as live."""
        header = handle.header
        meta: dict[str, Any] = {"createdAt": header["createdAt"]}
        for key in (
            "cwd",
            "parentSession",
            "isSeeded",
            "origin",
            "delegationDepth",
            "agentPreset",
        ):
            if header.get(key) is not None:
                meta[key] = header[key]
        return self._store.create(
            session_id,
            options={
                "meta": meta,
                "event_state": event_state,
                "seed": events,
                "inherited_event_count": handle.inherited_event_count,
            },
        )

    def read(self, session_id: str) -> Session | None:
        """Inspect a session without taking ownership of it.

        A live session is returned as-is; otherwise the stored log is
        opened for read and restored detached — no store entry, no
        announcement, no write claim, and no migration is published.
        Returns ``None`` when the session is not stored anywhere.
        """
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]
        if self._store.get(session_id) is not None:
            return self._store.get(session_id)
        if self._persistence is None:
            return None
        try:
            handle = self._persistence.open(session_id, "read")
        except SessionPersistenceNotFoundError:
            return None
        event_state, events = handle.read()
        session = Session.from_restore(
            session_id,
            events,
            handle.header,
            handle.inherited_event_count,
            event_state,
            projections=list(self._store.message_projections),
        )
        handle.close()
        return session

    def location(self, session_id: str) -> Path | None:
        """The materialized log path for one session, if it has one.

        Resolved from the layout directly: a live session's pending
        entry reports no location, so the backend's header-only
        observation cannot answer this.
        """
        if self.base_storage_dir is None:
            return None
        directory = find_session_dir(self.base_storage_dir, session_id)
        if directory is None:
            return None
        resolved = resolve_current_generation(directory)
        return resolved[1] if resolved is not None else None

    def session_path(self, session_id: str) -> Path:
        """The current-generation log path for one session.

        A session created but not yet written has no file on disk;
        the path returned is the generation file its first write will
        use. Raises ``SessionPersistenceNotFoundError`` for an unknown
        id, or when no persistence backend is configured.
        """
        if self.base_storage_dir is None:
            raise SessionPersistenceNotFoundError(session_id)
        directory = find_session_dir(self.base_storage_dir, session_id)
        if directory is not None:
            resolved = resolve_current_generation(directory)
            if resolved is not None:
                return resolved[1]
            return directory / generation_filename(CURRENT_FORMAT_VERSION)
        handle = self._handles.get(session_id)
        if handle is None:
            raise SessionPersistenceNotFoundError(session_id)
        return session_dir(
            self.base_storage_dir, handle.header.get("cwd"), session_id
        ) / generation_filename(CURRENT_FORMAT_VERSION)

    def list_sessions(self) -> list[str]:
        """Returns a sorted list of all active and persisted session IDs."""
        ids = {session.session_id for session in self._store.list()}
        ids.update(self._active_sessions.keys())
        if self._persistence is not None:
            for snapshot in self._persistence.list():
                ids.add(snapshot.header["id"])
        return sorted(ids)

    def close(self, session_id: str) -> bool:
        """Detach a live session and release its write handle.

        The store detach emits ``session/disposed``, so the live
        tracker drains a final time; the handle close then flushes and
        releases the lease and the single-writer claim. Returns whether
        the session was live.
        """
        session = self._active_sessions.get(session_id)
        if session is None:
            return False
        handle = self._handles.pop(session_id, None)
        self._store.remove(session)
        del self._active_sessions[session_id]
        if handle is not None:
            handle.close()
        return True

    def delete(self, session_id: str) -> bool:
        """Delete a session's stored log and forget it.

        Administrative operation, not part of the session lifecycle: it
        removes the session directory outright. It is not a tombstone
        and writes no log event. A session with a live write handle is
        refused (``SessionAlreadyOwnedError``): close the session first
        so the lease is released instead of broken.
        """
        handle = self._handles.get(session_id)
        if handle is not None and not handle.closed:
            raise SessionAlreadyOwnedError(session_id)
        existed = False
        session = self._active_sessions.pop(session_id, None)
        if session is not None:
            self._store.remove(session)
            existed = True
        elif self._store.get(session_id) is not None:
            self._store.remove(self._store.get(session_id))  # type: ignore[arg-type]
            existed = True
        if self.base_storage_dir is not None:
            directory = find_session_dir(self.base_storage_dir, session_id)
            if directory is not None:
                shutil.rmtree(directory)
                existed = True
        self._handles.pop(session_id, None)
        return existed

    async def flush(self, session: Session | None = None) -> None:
        """Durability barrier.

        One session: dispatch ``session/flush`` through the store, so
        the live tracker drains that session's buffered events and
        flushes its write handle. All sessions: drain every live
        buffer, then flush every write handle owned by this service;
        failures are raised together, naming each failed session id.
        """
        if session is not None:
            await self._store.flush(session)
            return
        if self._tracker is not None:
            self._tracker.drain_live()
        if self._persistence is not None:
            self._persistence.flush()

    def log_event(
        self,
        session: Session,
        event_type: EventType,
        payload: dict,
        *,
        surface_op: str | dict[str, Any] | None = None,
        source_event_seqs: list[int] | None = None,
    ) -> SessionEvent:
        """Append one event; surface types default to an ``append`` op.

        Store-owned sessions already publish ``session/event`` through
        the store's publication hook; independent sessions are broadcast
        here.
        """
        type_value = (
            event_type.value if isinstance(event_type, EventType) else str(event_type)
        )
        if type_value in SURFACE_EVENT_TYPES and surface_op is None:
            surface_op = "append"
        event = session.append(
            event_type,
            payload,
            surface_op=surface_op,
            source_event_seqs=source_event_seqs,
        )
        if session._publish_hook is None:
            # Broadcast durable session event
            self.ctx.emit("session/event", session, event)
        return event
