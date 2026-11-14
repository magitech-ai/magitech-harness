"""
SessionService (ctx.sessions): Service for session management.

Ownership split: this service owns the write handles; the store owns
live-session membership (prepare / enter / announce). A Session is
purely in-memory: its durable writes flow from ``session/event``
publication through the live tracker the persistence row installs on
the context, which batches them onto the write handle opened for that
session. ``flush`` is the durability barrier. The persistence backend
and the projection cache arrive from their own rows and are resolved
lazily on every use, so either row may land after this service
constructs; a configured storage dir with no persistence row fails
loud at use instead of silently running memory-only.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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
    JsonlSessionHandle,
    find_session_dir,
    generation_filename,
    resolve_current_generation,
    session_dir,
)
from magitech_harness.session.projection_registry import ProjectionSnapshot
from magitech_harness.session.store import Session, SessionStore

if TYPE_CHECKING:
    from magitech_harness.session.persistence_jsonl import (
        JsonlSessionPersistence,
    )
    from magitech_harness.session.projection_cache import SessionProjectionCache
    from magitech_harness.session.projection_registry import (
        SessionProjectionRegistry,
    )

logger = logging.getLogger("harness.session")


class SessionService(Service):
    """Service providing durable session storage and event broadcasting.

    Live sessions are owned by a :class:`SessionStore` (prepare / enter /
    announce); this service is the shell over it that also owns the
    write handles. The backend root comes from the ``base_storage_dir``
    config value; with no root configured the service runs in memory
    only — sessions work exactly the same, nothing is persisted, and
    no tracker is installed.

    The persistence backend and the projection cache are not owned
    here: they arrive from their rows and are resolved lazily on every
    use, so a row landing after this service constructs is still
    picked up. They are optional in the lazy sense only: when a
    storage dir is configured but no persistence row is active, the
    session op fails loud instead of silently running memory-only.
    A required injection would pend this fiber forever in memory-only
    mode, so the lazy resolution is the shape for an optional
    dependency.

    The projection registry is not owned here either: it arrives from
    the ``sessionProjections`` row, declared as a required injection, so
    the fiber stays pending until the registry is active and the row
    order cannot leave this service constructed without one. A host
    that constructs this service directly, bypassing the loader,
    still gets a working service: the projection cache is installed
    only when a registry is present; without one, projection reads
    fail loud at use while the store and persistence keep working.
    """

    inject: ClassVar[list[str]] = ["sessionProjections"]

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "sessions")
        cfg = config or {}
        # An unset ${MAGITECH_SESSIONS_DIR} interpolates to an empty
        # string, and Path("") is Path("."): normalise a falsy value to
        # None so the service stays memory-only instead of persisting
        # into the working directory.
        root = cfg.get("base_storage_dir") or None
        self.base_storage_dir = Path(root) if root is not None else None
        self._store = SessionStore(ctx)
        # Through the loader the injection guarantees the registry is
        # active before this constructs. Direct construction bypasses
        # that guarantee: a context that carries the slot but does not
        # resolve it is a framework inconsistency, not a missing row:
        # fail loud instead of silently running without projections.
        if ctx.has("sessionProjections"):
            registry = ctx.get("sessionProjections", None, strict=False)
            if registry is None:
                raise RuntimeError(
                    "sessions service: sessionProjections is present but "
                    "unresolvable on this context"
                )
            self._projections: SessionProjectionRegistry | None = registry
        else:
            self._projections = None
        self._active_sessions: dict[str, Session] = {}
        self._handles: dict[str, JsonlSessionHandle] = {}

    @property
    def store(self) -> SessionStore:
        """The live-session store this service shells over."""
        return self._store

    def _require_projections(self) -> SessionProjectionRegistry:
        """The registry, or a loud failure when the context carries none."""
        if self._projections is None:
            raise RuntimeError(
                "session projections are unavailable: no "
                "sessionProjections service on this context"
            )
        return self._projections

    def _resolve_persistence(self) -> JsonlSessionPersistence | None:
        """The persistence row's backend, or ``None`` for memory-only.

        Resolved fresh on every use: the row may land after this
        service constructs. When a storage dir is configured but no
        persistence row is active, the session op fails loud instead
        of silently running memory-only.
        """
        if self.base_storage_dir is None:
            return None
        service = self.ctx.get("sessionPersistence", None, strict=False)
        backend: JsonlSessionPersistence | None = (
            getattr(service, "backend", None) if service is not None else None
        )
        if backend is None:
            raise RuntimeError(
                "sessions service: base_storage_dir is configured but no "
                "sessionPersistence row is active"
            )
        return backend

    def _resolve_cache(self) -> SessionProjectionCache | None:
        """The projection-cache row's cache, or ``None`` when there is none.

        Resolved fresh on every use: the row may land after this
        service constructs. A hollow row carries no cache.
        """
        service = self.ctx.get("sessionProjectionCache", None, strict=False)
        cache: SessionProjectionCache | None = (
            getattr(service, "cache", None) if service is not None else None
        )
        return cache

    @property
    def projection_cache(self) -> SessionProjectionCache | None:
        """The persisted projection cache, or ``None`` for a memory-only service."""
        return self._resolve_cache()

    def flush_log_for_cache(self, session: Session) -> None:
        """Synchronous log drain + flush ahead of a cache write.

        The cache calls this between its checkpoint cut and its record
        write, so the log is durable before the cache row lands. Skipped
        when the session is no longer live in the store (the detach path:
        the store entry is already gone) or when no write handle exists
        yet (the create-point write runs before the persistence handle
        exists). The restore backstop covers any residual overreach.
        A cache exists only alongside a persistence root, so a missing
        backend here is a loud failure, never a silent skip.
        """
        if self._store.get(session.session_id) is not session:
            return
        persistence = self._resolve_persistence()
        if persistence is None:
            raise RuntimeError(
                "sessions service: no sessionPersistence row is active "
                "for the projection cache flush"
            )
        persistence.drain_live(session.session_id)
        handle = self._handles.get(session.session_id)
        if handle is not None:
            handle.flush()

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
        persistence = self._resolve_persistence()
        if persistence is None:
            session = self._store.create(sid)
            self._active_sessions[sid] = session
            return session
        header = self._new_header(sid)
        session = self._store.create(
            sid, options={"meta": {"createdAt": header["createdAt"]}}
        )
        try:
            handle = persistence.create(header)
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
        persistence = self._resolve_persistence()
        if persistence is None:
            self._active_sessions[session.session_id] = session
            return session
        header = dict(session.header)
        try:
            handle = persistence.create(
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
        persistence = self._resolve_persistence()
        if persistence is None:
            return None
        try:
            handle = persistence.open(session_id, "write")
        except SessionPersistenceNotFoundError:
            return None
        try:
            event_state, events = handle.read()
            session = self._enter_restored(session_id, handle, event_state, events)
            cache = self._resolve_cache()
            # A persistence root installs a cache only when the context
            # carries a registry; without one there are no cells to seed.
            if cache is not None:
                cache.hydrate_prepared(session, list(session.events))
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
        persistence = self._resolve_persistence()
        if persistence is None:
            return None
        try:
            handle = persistence.open(session_id, "read")
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

    def read_projections(self, session_id: str) -> ProjectionSnapshot | None:
        """Cold-read one session's projections from its complete stored log.

        A live session is read from the registry directly. Otherwise the
        stored log is opened for read and the cache cold-reads it (seeding
        from identity-checked rows and writing the refreshed checkpoint
        back, fail-soft). Returns ``None`` when the session is not stored
        anywhere, or when no persistence backend is configured.
        """
        if session_id in self._active_sessions:
            return self._require_projections().snapshot(
                self._active_sessions[session_id]
            )
        live = self._store.get(session_id)
        if live is not None:
            return self._require_projections().snapshot(live)
        persistence = self._resolve_persistence()
        if persistence is None:
            return None
        cache = self._resolve_cache()
        if cache is None:
            raise RuntimeError(
                "session projections are unavailable: no "
                "sessionProjections service on this context"
            )
        try:
            handle = persistence.open(session_id, "read")
        except SessionPersistenceNotFoundError:
            return None
        try:
            _event_state, events = handle.read()
            return cache.cold_snapshot(
                handle.header, handle.inherited_event_count, events
            )
        finally:
            handle.close()

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
        persistence = self._resolve_persistence()
        if persistence is not None:
            for snapshot in persistence.list():
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
        persistence = self._resolve_persistence()
        if persistence is not None:
            persistence.drain_live()
            persistence.flush()

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
