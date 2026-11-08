"""
SessionService (ctx.sessions): Service for session management.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from anima import Context, Service

from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.store import Session


class SessionService(Service):
    """Service providing durable session storage and event broadcasting."""

    def __init__(self, ctx: Context, base_storage_dir: Path | None = None) -> None:
        super().__init__(ctx, "sessions")
        self.base_storage_dir = base_storage_dir
        self._active_sessions: dict[str, Session] = {}

    def create(self, session_id: str | None = None) -> Session:
        sid = session_id or str(uuid.uuid4())
        storage_path = (
            self.base_storage_dir / f"{sid}.jsonl" if self.base_storage_dir else None
        )
        session = Session(session_id=sid, storage_path=storage_path)
        self._active_sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._active_sessions.get(session_id)

    def load(self, session_id: str, *, repair_on_open: bool = True) -> Session | None:
        """Loads a session by ID from memory cache or storage directory."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        if self.base_storage_dir:
            storage_path = self.base_storage_dir / f"{session_id}.jsonl"
            if storage_path.exists():
                session = Session.load(storage_path, repair_on_open=repair_on_open)
                self._active_sessions[session_id] = session
                return session

        return None

    def list_sessions(self) -> list[str]:
        """Returns a sorted list of all active and persisted session IDs."""
        ids = set(self._active_sessions.keys())
        if self.base_storage_dir and self.base_storage_dir.exists():
            for p in self.base_storage_dir.glob("*.jsonl"):
                ids.add(p.stem)
        return sorted(ids)

    def delete(self, session_id: str) -> bool:
        """Deletes a session from active memory and storage disk."""
        existed = False
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            existed = True

        if self.base_storage_dir:
            storage_path = self.base_storage_dir / f"{session_id}.jsonl"
            if storage_path.exists():
                storage_path.unlink()
                existed = True

        return existed

    def log_event(
        self, session: Session, event_type: EventType, payload: dict
    ) -> SessionEvent:
        event = SessionEvent(
            id=str(uuid.uuid4()),
            session_id=session.session_id,
            type=event_type,
            payload=payload,
        )
        session.append(event)
        # Broadcast durable session event
        self.ctx.emit("session/event", session, event)
        return event
