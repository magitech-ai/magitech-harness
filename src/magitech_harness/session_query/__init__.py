"""Session query service (ctx.sessionQuery): unified session-history query.

Exact reads, filters, and traces are backend-independent concrete behavior.
A backend implements full-text observation on the same ``ctx.sessionQuery``
service. Reference: ``packages/session-query/session-query``.
"""

from __future__ import annotations

from magitech_harness.session_query.engine import (
    SessionQueryConfig,
    SessionQueryEngine,
)
from magitech_harness.session_query.errors import (
    SessionQueryError,
    SessionQueryErrorCode,
)
from magitech_harness.session_query.filters import (
    SessionEventResultFilter,
    SessionResultFilter,
)
from magitech_harness.session_query.types import (
    SessionEventReadRequest,
    SessionEventSearchDocument,
    SessionEventSearchHit,
    SessionEventSearchPage,
    SessionEventSearchRequest,
    SessionEventSurface,
    SessionEventTrace,
    SessionEventTraceObservation,
    SessionEventTraceRequest,
    SessionHeader,
    SessionId,
    SessionLineageNode,
    SessionLineageTrace,
    SessionLogOffset,
    SessionLogSnapshot,
    SessionObservation,
    SessionObservationOptions,
    SessionRecord,
    SessionResultRange,
    SessionSearchCursor,
    SessionSearchExecContext,
    SessionSearchHit,
    SessionSearchPage,
    SessionSearchRequest,
    SessionSurfaceSnapshot,
    SessionTitleObservation,
    SessionTitleObservationResult,
    SessionTitleSnapshot,
    SessionWindow,
)

__all__ = [
    "SessionEventReadRequest",
    "SessionEventResultFilter",
    "SessionEventSearchDocument",
    "SessionEventSearchHit",
    "SessionEventSearchPage",
    "SessionEventSearchRequest",
    "SessionEventSurface",
    "SessionEventTrace",
    "SessionEventTraceObservation",
    "SessionEventTraceRequest",
    "SessionHeader",
    "SessionId",
    "SessionLineageNode",
    "SessionLineageTrace",
    "SessionLogOffset",
    "SessionLogSnapshot",
    "SessionObservation",
    "SessionObservationOptions",
    "SessionQueryConfig",
    "SessionQueryEngine",
    "SessionQueryError",
    "SessionQueryErrorCode",
    "SessionRecord",
    "SessionResultFilter",
    "SessionResultRange",
    "SessionSearchCursor",
    "SessionSearchExecContext",
    "SessionSearchHit",
    "SessionSearchPage",
    "SessionSearchRequest",
    "SessionSurfaceSnapshot",
    "SessionTitleObservation",
    "SessionTitleObservationResult",
    "SessionTitleSnapshot",
    "SessionWindow",
]
