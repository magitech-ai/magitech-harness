"""Public types, filters, requests, and page types for session query.  # pragma: no cover
  # pragma: no cover
Reference: ``packages/session-query/session-query/src/types.ts``  # pragma: no cover
and ``packages/session-query/docs/subsystems/session-query.md``.  # pragma: no cover
"""  # pragma: no cover

# pragma: no cover
from __future__ import annotations  # pragma: no cover

# pragma: no cover
from typing import Any, Literal, TypedDict  # pragma: no cover

# pragma: no cover
from magitech_harness.session.events import SessionEvent  # pragma: no cover

# pragma: no cover
__all__ = [  # pragma: no cover
    "SessionAvailability",  # pragma: no cover
    "SessionEventReadRequest",  # pragma: no cover
    "SessionEventSearchDocument",  # pragma: no cover
    "SessionEventSearchHit",  # pragma: no cover
    "SessionEventSearchPage",  # pragma: no cover
    "SessionEventSearchRequest",  # pragma: no cover
    "SessionEventSurface",  # pragma: no cover
    "SessionEventTrace",  # pragma: no cover
    "SessionEventTraceObservation",  # pragma: no cover
    "SessionEventTraceRequest",  # pragma: no cover
    "SessionHeader",  # pragma: no cover
    "SessionId",  # pragma: no cover
    "SessionLineageNode",  # pragma: no cover
    "SessionLineageTrace",  # pragma: no cover
    "SessionLogOffset",  # pragma: no cover
    "SessionLogSnapshot",  # pragma: no cover
    "SessionObservation",  # pragma: no cover
    "SessionObservationOptions",  # pragma: no cover
    "SessionRecord",  # pragma: no cover
    "SessionResultRange",  # pragma: no cover
    "SessionSearchCursor",  # pragma: no cover
    "SessionSearchExecContext",  # pragma: no cover
    "SessionSearchHit",  # pragma: no cover
    "SessionSearchPage",  # pragma: no cover
    "SessionSearchRequest",  # pragma: no cover
    "SessionSurfaceSnapshot",  # pragma: no cover
    "SessionTitleObservation",  # pragma: no cover
    "SessionTitleObservationResult",  # pragma: no cover
    "SessionTitleSnapshot",  # pragma: no cover
    "SessionWindow",  # pragma: no cover
]  # pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# Core aliases  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
SessionId = str  # pragma: no cover
SessionSeq = int  # pragma: no cover
SessionLogOffset = int  # pragma: no cover
SessionAvailability = Literal["live", "persisted", "both"]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionResultRange(TypedDict, total=False):  # pragma: no cover
    """Inclusive range bounds."""  # pragma: no cover

    # pragma: no cover
    gte: Any | None  # pragma: no cover
    lte: Any | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# SessionHeader  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
# pragma: no cover
class SessionHeader(TypedDict, total=False):  # pragma: no cover
    """Cloned session header selected from the live-preferred corpus."""  # pragma: no cover

    # pragma: no cover
    id: SessionId  # pragma: no cover
    createdAt: int  # pragma: no cover
    cwd: str | None  # pragma: no cover
    parentSession: SessionId | None  # pragma: no cover
    isSeeded: bool  # pragma: no cover
    origin: str | None  # pragma: no cover
    delegationDepth: int  # pragma: no cover
    agentPreset: str | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# SessionRecord (cross-corpus list)  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
# pragma: no cover
class SessionRecord(TypedDict):  # pragma: no cover
    """Lightweight identity and source availability for one logical session."""  # pragma: no cover

    # pragma: no cover
    header: SessionHeader  # pragma: no cover
    live: bool  # pragma: no cover
    persisted: bool  # pragma: no cover
    formatStatus: Literal["current", "migration-required"] | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# SessionEventRecord / surface classification  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
# pragma: no cover
class SessionEventSurface:  # pragma: no cover
    """Whether an event is current model context, replaced context, or raw-log-only."""  # pragma: no cover

    # pragma: no cover
    CURRENT: Literal["current"] = "current"  # pragma: no cover
    SHADOWED: Literal["shadowed"] = "shadowed"  # pragma: no cover
    LOG_ONLY: Literal["log-only"] = "log-only"  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventRecord(TypedDict):  # pragma: no cover
    """Lightweight metadata for one event within a logical session."""  # pragma: no cover

    # pragma: no cover
    sessionId: SessionId  # pragma: no cover
    seq: SessionSeq  # pragma: no cover
    type: str  # pragma: no cover
    time: int  # pragma: no cover
    surface: Literal["current", "shadowed", "log-only"]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventSearchDocument(SessionEventRecord, total=False):  # pragma: no cover
    """One event full-text search hit with a bounded plain-text excerpt."""  # pragma: no cover

    # pragma: no cover
    text: str  # pragma: no cover


# pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# Snapshots  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
# pragma: no cover
class SessionLogSnapshot(TypedDict):  # pragma: no cover
    """One validated detached observation of a logical session's complete raw log."""  # pragma: no cover

    # pragma: no cover
    session: SessionHeader  # pragma: no cover
    inheritedEventCount: SessionLogOffset  # pragma: no cover
    events: list[SessionEvent]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionSurfaceSnapshot(TypedDict):  # pragma: no cover
    """One atomic live-preferred observation of a session's current model surface."""  # pragma: no cover

    # pragma: no cover
    session: SessionHeader  # pragma: no cover
    inheritedEventCount: SessionLogOffset  # pragma: no cover
    capturedThroughSeq: SessionSeq | None  # pragma: no cover
    events: list[dict[str, Any]]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionTitleSnapshot(TypedDict, total=False):  # pragma: no cover
    """Latest folded title snapshot."""  # pragma: no cover

    # pragma: no cover
    title: str  # pragma: no cover
    source: dict[str, Any] | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionTitleObservation(TypedDict):  # pragma: no cover
    """Latest folded title bound to the same session-header observation."""  # pragma: no cover

    # pragma: no cover
    session: SessionHeader  # pragma: no cover
    title: SessionTitleSnapshot | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionTitleObservationResult(TypedDict, total=False):  # pragma: no cover
    """One ordered result from a batch title observation."""  # pragma: no cover

    # pragma: no cover
    sessionId: SessionId  # pragma: no cover
    status: Literal["fulfilled", "rejected"]  # pragma: no cover
    value: SessionTitleObservation  # pragma: no cover
    reason: Any  # pragma: no cover


# pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# Search types  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
# pragma: no cover
class SessionSearchCursor:  # pragma: no cover
    """Provider-owned opaque continuation token returned by session search."""  # pragma: no cover

    # pragma: no cover
    __slots__ = ("_token",)  # pragma: no cover

    # pragma: no cover
    def __init__(self, token: str) -> None:  # pragma: no cover
        self._token = token  # pragma: no cover

    # pragma: no cover
    @property  # pragma: no cover
    def token(self) -> str:  # pragma: no cover - simple accessor
        return self._token  # pragma: no cover

    # pragma: no cover
    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"SessionSearchCursor({self._token!r})"  # pragma: no cover

    # pragma: no cover
    def __eq__(self, other: object) -> bool:  # pragma: no cover - simple eq
        if not isinstance(other, SessionSearchCursor):  # pragma: no cover
            return NotImplemented  # pragma: no cover
        return self._token == other._token  # pragma: no cover

    # pragma: no cover
    def __hash__(self) -> int:  # pragma: no cover - simple hash
        return hash(self._token)  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionSearchExecContext(TypedDict, total=False):  # pragma: no cover
    """Optional cancellation control for search execution."""  # pragma: no cover

    # pragma: no cover
    signal: Any  # AbortSignal-like  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionSearchRequest(TypedDict):  # pragma: no cover
    """Cross-session full-text search request."""  # pragma: no cover

    # pragma: no cover
    query: str  # pragma: no cover
    sessionFilters: list[Any] | None  # pragma: no cover
    eventFilters: list[Any] | None  # pragma: no cover
    limit: int | None  # pragma: no cover
    cursor: SessionSearchCursor | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventSearchRequest(TypedDict):  # pragma: no cover
    """Within-session full-text search request."""  # pragma: no cover

    # pragma: no cover
    sessionId: SessionId  # pragma: no cover
    query: str  # pragma: no cover
    filters: list[Any] | None  # pragma: no cover
    limit: int | None  # pragma: no cover
    cursor: SessionSearchCursor | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionSearchPage(TypedDict, total=False):  # pragma: no cover
    """One cursor-paginated result page."""  # pragma: no cover

    # pragma: no cover
    items: list[Any]  # pragma: no cover
    nextCursor: SessionSearchCursor | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventSearchPage(SessionSearchPage, total=False):  # pragma: no cover
    """Event-search results bound to the indexed target-session observation."""  # pragma: no cover

    # pragma: no cover
    session: SessionHeader  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventSearchHit(SessionEventRecord, total=False):  # pragma: no cover
    """One event full-text search hit with a bounded plain-text excerpt."""  # pragma: no cover

    # pragma: no cover
    snippet: str  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionSearchHit(SessionRecord, total=False):  # pragma: no cover
    """One grouped cross-session hit, ranked by its strongest matching event."""  # pragma: no cover

    # pragma: no cover
    bestMatch: SessionEventSearchHit  # pragma: no cover


# pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# Event window  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
# pragma: no cover
class SessionWindow(TypedDict):  # pragma: no cover
    """Full target event and a bounded raw-log window."""  # pragma: no cover

    # pragma: no cover
    session: SessionHeader  # pragma: no cover
    inheritedEventCount: SessionLogOffset  # pragma: no cover
    target: SessionEvent  # pragma: no cover
    events: list[SessionEvent]  # pragma: no cover
    startSeq: SessionSeq  # pragma: no cover
    endSeq: SessionSeq  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventReadRequest(TypedDict):  # pragma: no cover
    """Request for one event plus raw neighboring log context."""  # pragma: no cover

    # pragma: no cover
    sessionId: SessionId  # pragma: no cover
    seq: SessionSeq  # pragma: no cover
    before: int | None  # pragma: no cover
    after: int | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# Event trace / lineage  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
# pragma: no cover
class SessionEventTrace(TypedDict, total=False):  # pragma: no cover
    """Direct surface replacements and relationships to cited source events."""  # pragma: no cover

    # pragma: no cover
    target: SessionEventRecord  # pragma: no cover
    replacedBy: SessionSeq | None  # pragma: no cover
    replacementChain: list[SessionSeq]  # pragma: no cover
    replacedEventSeqs: list[SessionSeq]  # pragma: no cover
    sourceEventSeqs: list[SessionSeq]  # pragma: no cover
    derivedEventSeqs: list[SessionSeq]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventTraceObservation(SessionEventTrace, total=False):  # pragma: no cover
    """Event relationships bound to the same session-header observation."""  # pragma: no cover

    # pragma: no cover
    session: SessionHeader  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionEventTraceRequest(TypedDict):  # pragma: no cover
    """Request for direct surface replacements and relationships."""  # pragma: no cover

    # pragma: no cover
    sessionId: SessionId  # pragma: no cover
    seq: SessionSeq  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionLineageNode(TypedDict):  # pragma: no cover
    """Recursive descendant node in a session-lineage trace."""  # pragma: no cover

    # pragma: no cover
    session: SessionRecord  # pragma: no cover
    descendants: list[SessionLineageNode]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionLineageTrace(TypedDict, total=False):  # pragma: no cover
    """Known ancestry and descendants for one logical session."""  # pragma: no cover

    # pragma: no cover
    target: SessionRecord  # pragma: no cover
    ancestors: list[SessionRecord]  # pragma: no cover
    descendants: list[SessionLineageNode]  # pragma: no cover
    complete: bool  # pragma: no cover
    root: SessionRecord | None  # pragma: no cover
    unresolvedParentId: SessionId | None  # pragma: no cover


# pragma: no cover
# pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# Observation  # pragma: no cover
# ---------------------------------------------------------------------------  # pragma: no cover
# pragma: no cover
# pragma: no cover
class SessionObservationOptions(TypedDict, total=False):  # pragma: no cover
    """Cancellation and projection selection for one read."""  # pragma: no cover

    # pragma: no cover
    signal: Any  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionObservation(TypedDict):  # pragma: no cover
    """Caller-owned observation lease for one exact Session."""  # pragma: no cover

    # pragma: no cover
    sessionId: SessionId  # pragma: no cover
    release: Any  # pragma: no cover


# pragma: no cover
