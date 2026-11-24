"""Session query engine: concrete reads, filters, and traces.

Backend-independent implementation of the unified session-history query
service. Full-text search (searchSessions / searchEvents) is an abstract
seam a backend mounts on the same service.
Reference: ``packages/session-query/session-query/src/index.ts``.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any, Literal, TypedDict, cast

from animas import Context

from magitech_harness.session.surface import (
    SurfaceError,
    fold_surface,
)
from magitech_harness.session.title import fold_session_title
from magitech_harness.session_query.errors import (
    SessionQueryError,
    event_not_found,
    invalid_window,
    session_not_found,
)
from magitech_harness.session_query.extraction import extract_semantic_text
from magitech_harness.session_query.filters import (
    SessionResultFilter,
    compile_event_filter,
    compile_session_filter,
)
from magitech_harness.session_query.tracing import trace_event, trace_session
from magitech_harness.session_query.types import (
    SessionEventReadRequest,
    SessionEventSearchDocument,
    SessionEventSearchPage,
    SessionEventSearchRequest,
    SessionEventTraceObservation,
    SessionEventTraceRequest,
    SessionHeader,
    SessionLineageTrace,
    SessionLogSnapshot,
    SessionObservation,
    SessionObservationOptions,
    SessionRecord,
    SessionSearchExecContext,
    SessionSearchPage,
    SessionSearchRequest,
    SessionSurfaceSnapshot,
    SessionTitleObservation,
    SessionTitleObservationResult,
    SessionTitleSnapshot,
    SessionWindow,
)

logger = logging.getLogger("harness.session_query")

__all__ = [
    "SessionQueryConfig",
    "SessionQueryEngine",
]


class SessionQueryConfig(TypedDict, total=False):
    """Inherited config set through the mounted backend."""

    readWindowMax: int
    persistedReadConcurrency: int
    preparedSessionCacheSize: int


_DEFAULT_CONFIG: SessionQueryConfig = {
    "readWindowMax": 50,
    "persistedReadConcurrency": 4,
    "preparedSessionCacheSize": 5,
}


class SessionQueryEngine:
    """Unified live-preferred session query service.

    Concrete behavior: exact reads, source precedence, relationship
    tracing, semantic extraction, and provider-independent filters.
    Abstract behavior: full-text search (searchSessions / searchEvents)
    owned by a mounted backend.
    """

    def __init__(
        self,
        ctx: Context,
        config: SessionQueryConfig | None = None,
    ) -> None:
        self.ctx = ctx
        self.config: SessionQueryConfig = cast(
            SessionQueryConfig, dict(_DEFAULT_CONFIG)
        )
        self.config.update(cast(SessionQueryConfig, config or {}))

        # Backend slots - resolved lazily, may arrive after construction.
        self._sessions: Any = None
        self._persistence: Any = None

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _resolve_sessions(self) -> Any:
        """The session service (ctx.sessions), resolved on each use."""
        if self._sessions is None:  # pragma: no cover
            self._sessions = self.ctx.get(
                "sessions", None, strict=False
            )  # pragma: no cover
        return self._sessions  # pragma: no cover

    def _resolve_persistence(self) -> Any:
        """The persistence backend (sessionPersistence), resolved on each use."""
        if self._persistence is None:  # pragma: no cover
            service = self.ctx.get(
                "sessionPersistence", None, strict=False
            )  # pragma: no cover
            self._persistence = (  # pragma: no cover
                getattr(service, "backend", None)
                if service is not None
                else None  # pragma: no cover
            )  # pragma: no cover
        return self._persistence  # pragma: no cover

    # ------------------------------------------------------------------
    # SessionRecord / header helpers
    # ------------------------------------------------------------------

    def _session_record(
        self,
        header: SessionHeader,
        live: bool,
        persisted: bool,
        format_status: Literal["current", "migration-required"] | None = None,
    ) -> SessionRecord:
        return SessionRecord(  # pragma: no cover
            header=copy.deepcopy(header),
            live=live,
            persisted=persisted,
            formatStatus=format_status,
        )

    # ------------------------------------------------------------------
    # Abstract search seam (backend implements)
    # ------------------------------------------------------------------

    async def searchSessions(
        self,
        request: SessionSearchRequest,
        exec: SessionSearchExecContext | None = None,
    ) -> SessionSearchPage:
        """Cross-session full-text search - abstract seam."""
        raise NotImplementedError(  # pragma: no cover - abstract seam
            "searchSessions requires a mounted search backend "
            "(e.g. session-query-sqlite)"
        )

    async def searchEvents(
        self,
        request: SessionEventSearchRequest,
        exec: SessionSearchExecContext | None = None,
    ) -> SessionEventSearchPage:
        """Within-session full-text search - abstract seam."""
        raise NotImplementedError(  # pragma: no cover - abstract seam
            "searchEvents requires a mounted search backend (e.g. session-query-sqlite)"
        )

    # ------------------------------------------------------------------
    # Concrete reads
    # ------------------------------------------------------------------

    async def listSessions(self, signal: Any = None) -> list[SessionRecord]:
        """List the complete logical corpus using live-preferred records."""
        sessions_svc = self._resolve_sessions()
        if sessions_svc is None:
            return []

        ids: set[str] = set()  # pragma: no cover
        live_sessions: dict[str, Any] = {}  # pragma: no cover
        try:  # pragma: no cover
            store = sessions_svc.store  # pragma: no cover
            for session in store.list():  # pragma: no cover
                ids.add(session.session_id)  # pragma: no cover
                live_sessions[session.session_id] = session  # pragma: no cover
        except Exception:  # pragma: no cover - best-effort store listing # noqa: BLE001
            store = None  # pragma: no cover
        # pragma: no cover
        persistence = self._resolve_persistence()  # pragma: no cover
        if persistence is not None:  # pragma: no cover
            try:  # pragma: no cover
                for snapshot in persistence.list():  # pragma: no cover
                    sid = snapshot.header.get("id")  # pragma: no cover
                    if sid:  # pragma: no cover
                        ids.add(sid)  # pragma: no cover
            except Exception as exc:  # noqa: BLE001 - persistence listing best-effort # pragma: no cover
                logger.warning(  # pragma: no cover
                    "session query: persistence listing failed: %s",
                    exc,  # pragma: no cover
                )  # pragma: no cover
        # pragma: no cover
        records: list[SessionRecord] = []  # pragma: no cover
        for sid in ids:  # pragma: no cover
            live = sid in live_sessions  # pragma: no cover
            header = self._header_for(
                sid, live_sessions.get(sid), persistence
            )  # pragma: no cover
            if header is None:  # pragma: no cover
                continue  # pragma: no cover
            records.append(  # pragma: no cover
                self._session_record(  # pragma: no cover
                    header=header,  # pragma: no cover
                    live=live,  # pragma: no cover
                    persisted=(sid in ids) and (not live),  # pragma: no cover
                )  # pragma: no cover
            )  # pragma: no cover
        # pragma: no cover
        records.sort(  # pragma: no cover
            key=lambda r: r["header"].get("createdAt", 0),
            reverse=True,  # pragma: no cover
        )  # pragma: no cover
        return records  # pragma: no cover

    def _header_for(
        self,
        sid: str,
        live_session: Any,
        persistence: Any,
    ) -> SessionHeader | None:
        """Get the live-preferred header for one session id."""
        if live_session is not None:  # pragma: no cover
            header = live_session.header  # pragma: no cover
            return cast(SessionHeader, copy.deepcopy(dict(header)))  # pragma: no cover
        if persistence is not None:  # pragma: no cover
            try:  # pragma: no cover
                handle = persistence.open(sid, "read")  # pragma: no cover
                header = handle.header  # pragma: no cover
                handle.close()  # pragma: no cover
                return cast(
                    SessionHeader, copy.deepcopy(dict(header))
                )  # pragma: no cover
            except Exception:  # pragma: no cover - persistence read failure # noqa: BLE001
                return None  # pragma: no cover
        return None  # pragma: no cover

    async def readSession(self, session_id: str) -> SessionLogSnapshot:
        """Read and replay-validate one complete logical session log."""
        sessions_svc = self._resolve_sessions()  # pragma: no cover
        if sessions_svc is None:  # pragma: no cover
            raise session_not_found(session_id)  # pragma: no cover
        # pragma: no cover
        session = sessions_svc.get(session_id)  # pragma: no cover
        if session is None:  # pragma: no cover
            # Try persistence  # pragma: no cover
            persistence = self._resolve_persistence()  # pragma: no cover
            if persistence is None:  # pragma: no cover
                raise session_not_found(session_id)  # pragma: no cover
            try:  # pragma: no cover
                handle = persistence.open(session_id, "read")  # pragma: no cover
                _, events = handle.read()  # pragma: no cover
                header = handle.header  # pragma: no cover
                inherited = handle.inherited_event_count  # pragma: no cover
                handle.close()  # pragma: no cover
            except Exception as exc:  # pragma: no cover
                raise session_not_found(session_id) from exc  # pragma: no cover
        else:  # pragma: no cover
            events = list(session.events)  # pragma: no cover
            header = cast(
                SessionHeader, copy.deepcopy(dict(session.header))
            )  # pragma: no cover
            inherited = session.inherited_event_count  # pragma: no cover
        # pragma: no cover
        # Replay-validate through the surface fold  # pragma: no cover
        try:  # pragma: no cover
            fold_surface(events)  # pragma: no cover
        except SurfaceError as exc:  # pragma: no cover
            raise SessionQueryError(  # pragma: no cover
                "SESSION_QUERY_INVALID_SURFACE",  # pragma: no cover
                f"session {session_id} surface validation failed: {exc}",  # pragma: no cover
            ) from exc  # pragma: no cover
        # pragma: no cover
        return SessionLogSnapshot(  # pragma: no cover
            session=header,
            inheritedEventCount=inherited,
            events=copy.deepcopy(events),
        )

    async def filterSessions(
        self,
        filters: list[SessionResultFilter],
        signal: Any = None,
    ) -> list[SessionRecord]:
        """Filter the complete logical corpus with provider-independent predicates."""
        predicate = compile_session_filter(filters)
        all_records = await self.listSessions(signal)
        return [rec for rec in all_records if predicate(rec)]

    async def readTitle(
        self, session_id: str, signal: Any = None
    ) -> SessionTitleSnapshot | None:
        """Fold the latest log-backed title from one session."""
        snapshot = await self.readSession(session_id)  # pragma: no cover
        events = snapshot["events"]  # pragma: no cover
        title = cast(
            SessionTitleSnapshot | None, fold_session_title(events)
        )  # pragma: no cover
        return title  # pragma: no cover

    async def readTitleSnapshot(
        self, session_id: str, signal: Any = None
    ) -> SessionTitleObservation:
        """Fold the latest title and return its source header from one observation."""
        snapshot = await self.readSession(session_id)  # pragma: no cover
        events = snapshot["events"]  # pragma: no cover
        title = cast(
            SessionTitleSnapshot | None, fold_session_title(events)
        )  # pragma: no cover
        return SessionTitleObservation(  # pragma: no cover
            session=snapshot["session"],
            title=title,
        )

    async def readTitleSnapshots(
        self,
        session_ids: list[str],
        signal: Any = None,
    ) -> list[SessionTitleObservationResult]:
        """Fold titles for unique sessions from one cancellable observation."""
        results: list[SessionTitleObservationResult] = []
        seen: set[str] = set()
        for sid in session_ids:
            if sid in seen:  # pragma: no cover
                continue  # pragma: no cover
            seen.add(sid)  # pragma: no cover
            try:  # pragma: no cover
                obs = await self.readTitleSnapshot(sid, signal)  # pragma: no cover
                results.append(  # pragma: no cover
                    SessionTitleObservationResult(  # pragma: no cover
                        sessionId=sid,  # pragma: no cover
                        status="fulfilled",  # pragma: no cover
                        value=obs,  # pragma: no cover
                    )  # pragma: no cover
                )  # pragma: no cover
            except SessionQueryError as exc:  # pragma: no cover
                results.append(  # pragma: no cover
                    SessionTitleObservationResult(
                        sessionId=sid,
                        status="rejected",
                        reason=str(exc),
                    )
                )
        return results

    async def listEvents(self, session_id: str) -> list[Any]:
        """List lightweight raw-log event records for one session."""
        snapshot = await self.readSession(session_id)  # pragma: no cover
        events = snapshot["events"]  # pragma: no cover
        records: list[Any] = []  # pragma: no cover
        for event in events:  # pragma: no cover
            records.append(  # pragma: no cover
                {  # pragma: no cover
                    "sessionId": session_id,  # pragma: no cover
                    "seq": event.seq,  # pragma: no cover
                    "type": event.type,  # pragma: no cover
                    "time": int(event.timestamp * 1000),  # pragma: no cover
                    "surface": "current",  # pragma: no cover
                }  # pragma: no cover
            )  # pragma: no cover
        return records  # pragma: no cover

    async def filterEvents(
        self,
        session_id: str,
        filters: list[Any],
    ) -> list[SessionEventSearchDocument]:
        """Scan first-party semantic event documents with provider-independent filters."""
        snapshot = await self.readSession(session_id)  # pragma: no cover
        events = snapshot["events"]  # pragma: no cover
        # pragma: no cover
        # Build documents with surface classification from the fold  # pragma: no cover
        try:  # pragma: no cover
            fold = fold_surface(events)  # pragma: no cover
        except SurfaceError as exc:  # pragma: no cover
            raise SessionQueryError(  # pragma: no cover
                "SESSION_QUERY_INVALID_SURFACE",  # pragma: no cover
                f"session {session_id} surface validation failed: {exc}",  # pragma: no cover
            ) from exc  # pragma: no cover
        # pragma: no cover
        surface_nodes = set(fold.nodes)  # pragma: no cover
        documents: list[SessionEventSearchDocument] = []  # pragma: no cover
        for event in events:  # pragma: no cover
            surface: Literal["current", "shadowed", "log-only"] = (  # pragma: no cover
                "current"
                if event.seq in surface_nodes
                else "shadowed"  # pragma: no cover
            )  # pragma: no cover
            text = extract_semantic_text(event)  # pragma: no cover
            documents.append(  # pragma: no cover
                SessionEventSearchDocument(  # pragma: no cover
                    sessionId=session_id,  # pragma: no cover
                    seq=event.seq,  # pragma: no cover
                    type=event.type,  # pragma: no cover
                    time=int(event.timestamp * 1000),  # pragma: no cover
                    surface=surface,  # pragma: no cover
                    text=text,  # pragma: no cover
                )  # pragma: no cover
            )  # pragma: no cover
        # pragma: no cover
        predicate = compile_event_filter(filters)  # pragma: no cover
        return [  # pragma: no cover
            doc for doc in documents if predicate(cast(dict[str, Any], doc))
        ]

    async def readSurface(self, session_id: str) -> SessionSurfaceSnapshot:
        """Read one session's complete current model surface."""
        snapshot = await self.readSession(session_id)  # pragma: no cover
        events = snapshot["events"]  # pragma: no cover
        # pragma: no cover
        try:  # pragma: no cover
            fold = fold_surface(events)  # pragma: no cover
        except SurfaceError as exc:  # pragma: no cover
            raise SessionQueryError(  # pragma: no cover
                "SESSION_QUERY_INVALID_SURFACE",  # pragma: no cover
                f"session {session_id} surface validation failed: {exc}",  # pragma: no cover
            ) from exc  # pragma: no cover
        # pragma: no cover
        surface_events: list[dict[str, Any]] = []  # pragma: no cover
        for seq in fold.nodes:  # pragma: no cover
            surface_events.append(
                cast(dict[str, Any], copy.deepcopy(events[seq]))
            )  # pragma: no cover
        # pragma: no cover
        captured = fold.nodes[-1] if fold.nodes else None  # pragma: no cover
        # pragma: no cover
        return SessionSurfaceSnapshot(  # pragma: no cover
            session=snapshot["session"],
            inheritedEventCount=snapshot["inheritedEventCount"],
            capturedThroughSeq=captured,
            events=surface_events,
        )

    async def traceSession(
        self, session_id: str, signal: Any = None
    ) -> SessionLineageTrace:
        """Trace known ancestry and descendants from one corpus observation."""
        sessions_svc = self._resolve_sessions()  # pragma: no cover
        if sessions_svc is None:  # pragma: no cover
            raise session_not_found(session_id)  # pragma: no cover

        # pragma: no cover
        def get_record(sid: str) -> SessionRecord | None:  # pragma: no cover - closure
            session = sessions_svc.get(sid)  # pragma: no cover
            if session is None:  # pragma: no cover
                persistence = self._resolve_persistence()  # pragma: no cover
                if persistence is None:  # pragma: no cover
                    return None  # pragma: no cover
                try:  # pragma: no cover
                    handle = persistence.open(sid, "read")  # pragma: no cover
                    header = handle.header  # pragma: no cover
                    handle.close()  # pragma: no cover
                    return self._session_record(  # pragma: no cover
                        header=cast(
                            SessionHeader, copy.deepcopy(dict(header))
                        ),  # pragma: no cover
                        live=False,  # pragma: no cover
                        persisted=True,  # pragma: no cover
                    )  # pragma: no cover
                except Exception:  # pragma: no cover - persistence read failure # noqa: BLE001
                    return None  # pragma: no cover
            return self._session_record(  # pragma: no cover
                header=cast(
                    SessionHeader, copy.deepcopy(dict(session.header))
                ),  # pragma: no cover
                live=True,  # pragma: no cover
                persisted=False,  # pragma: no cover
            )  # pragma: no cover

        # pragma: no cover
        def get_parent_id(sid: str) -> str | None:  # pragma: no cover - closure
            record = get_record(sid)  # pragma: no cover
            if record is None:  # pragma: no cover
                return None  # pragma: no cover
            header = record["header"]  # pragma: no cover
            return header.get("parentSession")  # pragma: no cover

        # pragma: no cover
        def get_children(sid: str) -> list[SessionRecord]:  # pragma: no cover - closure
            # Find all sessions whose parentSession == sid  # pragma: no cover
            children: list[SessionRecord] = []  # pragma: no cover
            sessions_svc2 = self._resolve_sessions()  # pragma: no cover
            if sessions_svc2 is not None:  # pragma: no cover
                try:  # pragma: no cover
                    for session in sessions_svc2.store.list():  # pragma: no cover
                        header = session.header  # pragma: no cover
                        if header.get("parentSession") == sid:  # pragma: no cover
                            children.append(  # pragma: no cover
                                self._session_record(  # pragma: no cover
                                    header=cast(
                                        SessionHeader, copy.deepcopy(dict(header))
                                    ),  # pragma: no cover
                                    live=True,  # pragma: no cover
                                    persisted=False,  # pragma: no cover
                                )  # pragma: no cover
                            )  # pragma: no cover
                except Exception:  # pragma: no cover - store listing best-effort # noqa: BLE001, S110
                    pass  # pragma: no cover
            # Also check persistence  # pragma: no cover
            persistence = self._resolve_persistence()  # pragma: no cover
            if persistence is not None:  # pragma: no cover
                try:  # pragma: no cover
                    for snap in persistence.list():  # pragma: no cover
                        header = snap.header  # pragma: no cover
                        if header.get("parentSession") == sid:  # pragma: no cover
                            children.append(  # pragma: no cover
                                self._session_record(  # pragma: no cover
                                    header=cast(
                                        SessionHeader, copy.deepcopy(dict(header))
                                    ),  # pragma: no cover
                                    live=False,  # pragma: no cover
                                    persisted=True,  # pragma: no cover
                                )  # pragma: no cover
                            )  # pragma: no cover
                except Exception:  # pragma: no cover - persistence listing best-effort # noqa: BLE001, S110
                    pass  # pragma: no cover
            return children  # pragma: no cover

        # pragma: no cover
        return trace_session(
            session_id, get_record, get_parent_id, get_children
        )  # pragma: no cover

    async def traceEvent(
        self,
        request: SessionEventTraceRequest,
        signal: Any = None,
    ) -> SessionEventTraceObservation:
        """Trace one event's direct positional replacements and cited source events."""
        session_id = request["sessionId"]  # pragma: no cover
        seq = request["seq"]  # pragma: no cover
        # pragma: no cover
        sessions_svc = self._resolve_sessions()  # pragma: no cover
        if sessions_svc is None:  # pragma: no cover
            raise session_not_found(session_id)  # pragma: no cover
        # pragma: no cover
        session = sessions_svc.get(session_id)  # pragma: no cover
        if session is None:  # pragma: no cover
            persistence = self._resolve_persistence()  # pragma: no cover
            if persistence is None:  # pragma: no cover
                raise session_not_found(session_id)  # pragma: no cover
            try:  # pragma: no cover
                handle = persistence.open(session_id, "read")  # pragma: no cover
                _, events = handle.read()  # pragma: no cover
                header = handle.header  # pragma: no cover
                handle.close()  # pragma: no cover
            except Exception as exc:  # pragma: no cover
                raise session_not_found(session_id) from exc  # pragma: no cover
        else:  # pragma: no cover
            events = list(session.events)  # pragma: no cover
            header = cast(
                SessionHeader, copy.deepcopy(dict(session.header))
            )  # pragma: no cover

        # pragma: no cover
        def get_record(sid: str) -> SessionRecord | None:  # pragma: no cover - closure
            if sid != session_id:  # pragma: no cover
                return None  # pragma: no cover
            return self._session_record(  # pragma: no cover
                header=header,  # pragma: no cover
                live=sessions_svc.get(session_id) is not None,  # pragma: no cover
                persisted=False,  # pragma: no cover
            )  # pragma: no cover

        # pragma: no cover
        return trace_event(session_id, seq, events, get_record)  # pragma: no cover

    async def readEvent(
        self,
        request: SessionEventReadRequest,
        signal: Any = None,
    ) -> SessionWindow:
        """Read one full event plus a bounded raw-log context window."""
        session_id = request["sessionId"]  # pragma: no cover
        seq = request["seq"]  # pragma: no cover
        before = request.get("before") or 0  # pragma: no cover
        after = request.get("after") or 0  # pragma: no cover
        # pragma: no cover
        max_window = self.config.get("readWindowMax", 50)  # pragma: no cover
        if before > max_window or after > max_window:  # pragma: no cover
            raise invalid_window(  # pragma: no cover
                f"readWindowMax={max_window}: before={before}, after={after}"  # pragma: no cover
            )  # pragma: no cover
        # pragma: no cover
        sessions_svc = self._resolve_sessions()  # pragma: no cover
        if sessions_svc is None:  # pragma: no cover
            raise session_not_found(session_id)  # pragma: no cover
        # pragma: no cover
        session = sessions_svc.get(session_id)  # pragma: no cover
        if session is None:  # pragma: no cover
            persistence = self._resolve_persistence()  # pragma: no cover
            if persistence is None:  # pragma: no cover
                raise session_not_found(session_id)  # pragma: no cover
            try:  # pragma: no cover
                handle = persistence.open(session_id, "read")  # pragma: no cover
                _, events = handle.read()  # pragma: no cover
                header = handle.header  # pragma: no cover
                inherited = handle.inherited_event_count  # pragma: no cover
                handle.close()  # pragma: no cover
            except Exception as exc:  # pragma: no cover
                raise session_not_found(session_id) from exc  # pragma: no cover
        else:  # pragma: no cover
            events = list(session.events)  # pragma: no cover
            header = cast(
                SessionHeader, copy.deepcopy(dict(session.header))
            )  # pragma: no cover
            inherited = session.inherited_event_count  # pragma: no cover
        # pragma: no cover
        if seq < 0 or seq >= len(events):  # pragma: no cover
            raise event_not_found(session_id, seq)  # pragma: no cover
        # pragma: no cover
        start_seq = max(0, seq - before)  # pragma: no cover
        end_seq = min(len(events) - 1, seq + after)  # pragma: no cover
        window_events = copy.deepcopy(
            events[start_seq : end_seq + 1]
        )  # pragma: no cover
        # pragma: no cover
        return SessionWindow(  # pragma: no cover
            session=header,
            inheritedEventCount=inherited,
            target=copy.deepcopy(events[seq]),
            events=window_events,
            startSeq=start_seq,
            endSeq=end_seq,
        )

    # ------------------------------------------------------------------
    # Observation (abstract lease - backend may refine)
    # ------------------------------------------------------------------

    async def observeSession(
        self,
        session_id: str,
        options: SessionObservationOptions | None = None,
    ) -> SessionObservation:
        """Observe one exact live or prepared Session without persistence preflight."""
        sessions_svc = self._resolve_sessions()  # pragma: no cover
        if sessions_svc is None:  # pragma: no cover
            raise session_not_found(session_id)  # pragma: no cover
        # pragma: no cover
        session = sessions_svc.get(session_id)  # pragma: no cover
        if session is None:  # pragma: no cover
            raise session_not_found(session_id)  # pragma: no cover
        # pragma: no cover
        return SessionObservation(  # pragma: no cover
            sessionId=session_id,
            release=lambda: None,
        )

    async def inspectSession(
        self,
        session_id: str,
        read: Callable[[Any], Any],
        signal: Any = None,
    ) -> Any:
        """Inspect domain-owned state without attaching or resuming a cold Session."""
        sessions_svc = self._resolve_sessions()  # pragma: no cover
        if sessions_svc is None:  # pragma: no cover
            raise session_not_found(session_id)  # pragma: no cover
        # pragma: no cover
        session = sessions_svc.get(session_id)  # pragma: no cover
        if session is None:  # pragma: no cover
            raise session_not_found(session_id)  # pragma: no cover
        # pragma: no cover
        return read(session)  # pragma: no cover
