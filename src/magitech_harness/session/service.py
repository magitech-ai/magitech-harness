"""
SessionService (ctx.sessions): Service for session management.
"""

from __future__ import annotations

import uuid
from typing import Any

from anima import Context, Service

from magitech_harness.session.events import (
    SURFACE_EVENT_TYPES,
    EventType,
    SessionEvent,
)
from magitech_harness.session.store import Session, SessionStore


class SessionService(Service):
    """Service providing durable session storage and event broadcasting.

    Live sessions are owned by a :class:`SessionStore` (prepare / enter /
    announce); this service is the shell over it that also owns the
    storage directory and cold loads.
    """

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "sessions")
        self.base_storage_dir = (config or {}).get("base_storage_dir")
        self._store = SessionStore(ctx)
        self._active_sessions: dict[str, Session] = {}

    @property
    def store(self) -> SessionStore:
        """The live-session store this service shells over."""
        return self._store

    def create(self, session_id: str | None = None) -> Session:
        sid = session_id or str(uuid.uuid4())
        storage_path = (
            self.base_storage_dir / f"{sid}.jsonl" if self.base_storage_dir else None
        )
        session = self._store.create(
            sid, options={"storage_path": storage_path} if storage_path else None
        )
        self._active_sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._store.get(session_id) or self._active_sessions.get(session_id)

    def load(self, session_id: str) -> Session | None:
        """Loads a session by ID from memory cache or storage directory."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        if self.base_storage_dir:
            storage_path = self.base_storage_dir / f"{session_id}.jsonl"
            if storage_path.exists():
                session = Session.load(storage_path)
                self._active_sessions[session_id] = session
                return session

        return None

    def list_sessions(self) -> list[str]:
        """Returns a sorted list of all active and persisted session IDs."""
        ids = {session.session_id for session in self._store.list()}
        ids.update(self._active_sessions.keys())
        if self.base_storage_dir and self.base_storage_dir.exists():
            for p in self.base_storage_dir.glob("*.jsonl"):
                ids.add(p.stem)
        return sorted(ids)

    def delete(self, session_id: str) -> bool:
        """Deletes a session from active memory and storage disk."""
        existed = False
        session = self._active_sessions.get(session_id)
        if session is not None:
            self._store.remove(session)
            del self._active_sessions[session_id]
            existed = True
        elif self._store.get(session_id) is not None:
            self._store.remove(self._store.get(session_id))  # type: ignore[arg-type]
            existed = True

        if self.base_storage_dir:
            storage_path = self.base_storage_dir / f"{session_id}.jsonl"
            if storage_path.exists():
                storage_path.unlink()
                existed = True

        return existed

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
