"""Closed SessionQuery error taxonomy.

Stable machine-routable failure codes for session reads, traces, and search.
Reference: ``packages/session-query/session-query/src/config.ts``.
"""

from __future__ import annotations


class SessionQueryErrorCode:
    """Closed union of session query error codes."""

    ABORTED = "SESSION_QUERY_ABORTED"
    CORRUPT_SESSION = "SESSION_QUERY_CORRUPT_SESSION"
    EVENT_NOT_FOUND = "SESSION_QUERY_EVENT_NOT_FOUND"
    INDEX_FAILED = "SESSION_QUERY_INDEX_FAILED"
    INVALID_CONFIG = "SESSION_QUERY_INVALID_CONFIG"
    INVALID_CURSOR = "SESSION_QUERY_INVALID_CURSOR"
    INVALID_FILTER = "SESSION_QUERY_INVALID_FILTER"
    INVALID_LIMIT = "SESSION_QUERY_INVALID_LIMIT"
    INVALID_QUERY = "SESSION_QUERY_INVALID_QUERY"
    INVALID_LINEAGE = "SESSION_QUERY_INVALID_LINEAGE"
    INVALID_SURFACE = "SESSION_QUERY_INVALID_SURFACE"
    INVALID_WINDOW = "SESSION_QUERY_INVALID_WINDOW"
    PERSISTENCE_FAILED = "SESSION_QUERY_PERSISTENCE_FAILED"
    SEARCH_DISABLED = "SESSION_QUERY_SEARCH_DISABLED"
    SESSION_NOT_FOUND = "SESSION_QUERY_SESSION_NOT_FOUND"
    STALE_CURSOR = "SESSION_QUERY_STALE_CURSOR"
    SOURCE_CONFLICT = "SESSION_QUERY_SOURCE_CONFLICT"

    @classmethod
    def all(cls) -> list[str]:
        """All valid error codes."""
        return [
            cls.ABORTED,
            cls.CORRUPT_SESSION,
            cls.EVENT_NOT_FOUND,
            cls.INDEX_FAILED,
            cls.INVALID_CONFIG,
            cls.INVALID_CURSOR,
            cls.INVALID_FILTER,
            cls.INVALID_LIMIT,
            cls.INVALID_QUERY,
            cls.INVALID_LINEAGE,
            cls.INVALID_SURFACE,
            cls.INVALID_WINDOW,
            cls.PERSISTENCE_FAILED,
            cls.SEARCH_DISABLED,
            cls.SESSION_NOT_FOUND,
            cls.STALE_CURSOR,
            cls.SOURCE_CONFLICT,
        ]


class SessionQueryError(Exception):
    """Typed failure for session reads, traces, and search.

    Carries a stable ``code`` from :class:`SessionQueryErrorCode` and an
    optional ``cause`` for projection or backend failures.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause

    def __repr__(self) -> str:  # pragma: no cover
        return f"SessionQueryError(code={self.code!r}, message={super().__str__()!r})"


def invalid_filter(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_INVALID_FILTER error."""
    return SessionQueryError(SessionQueryErrorCode.INVALID_FILTER, message)


def invalid_query(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_INVALID_QUERY error."""
    return SessionQueryError(SessionQueryErrorCode.INVALID_QUERY, message)


def invalid_limit(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_INVALID_LIMIT error."""
    return SessionQueryError(SessionQueryErrorCode.INVALID_LIMIT, message)


def invalid_window(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_INVALID_WINDOW error."""
    return SessionQueryError(SessionQueryErrorCode.INVALID_WINDOW, message)


def invalid_surface(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_INVALID_SURFACE error."""
    return SessionQueryError(SessionQueryErrorCode.INVALID_SURFACE, message)


def invalid_lineage(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_INVALID_LINEAGE error."""
    return SessionQueryError(SessionQueryErrorCode.INVALID_LINEAGE, message)


def session_not_found(session_id: str) -> SessionQueryError:
    """Build a SESSION_QUERY_SESSION_NOT_FOUND error."""
    return SessionQueryError(
        SessionQueryErrorCode.SESSION_NOT_FOUND,
        f"session not found: {session_id}",
    )


def event_not_found(session_id: str, seq: int) -> SessionQueryError:
    """Build a SESSION_QUERY_EVENT_NOT_FOUND error."""
    return SessionQueryError(
        SessionQueryErrorCode.EVENT_NOT_FOUND,
        f"event not found: session {session_id} seq {seq}",
    )


def persistence_failed(
    message: str, cause: Exception | None = None
) -> SessionQueryError:
    """Build a SESSION_QUERY_PERSISTENCE_FAILED error."""
    return SessionQueryError(
        SessionQueryErrorCode.PERSISTENCE_FAILED,
        message,
        cause=cause,
    )


def source_conflict(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_SOURCE_CONFLICT error."""
    return SessionQueryError(SessionQueryErrorCode.SOURCE_CONFLICT, message)


def abort(message: str, cause: Exception | None = None) -> SessionQueryError:
    """Build a SESSION_QUERY_ABORTED error."""
    return SessionQueryError(
        SessionQueryErrorCode.ABORTED,
        message,
        cause=cause,
    )


def invalid_config(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_INVALID_CONFIG error."""
    return SessionQueryError(SessionQueryErrorCode.INVALID_CONFIG, message)


def invalid_cursor(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_INVALID_CURSOR error."""
    return SessionQueryError(SessionQueryErrorCode.INVALID_CURSOR, message)


def search_disabled(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_SEARCH_DISABLED error."""
    return SessionQueryError(SessionQueryErrorCode.SEARCH_DISABLED, message)


def stale_cursor(message: str) -> SessionQueryError:
    """Build a SESSION_QUERY_STALE_CURSOR error."""
    return SessionQueryError(SessionQueryErrorCode.STALE_CURSOR, message)


def corrupt_session(message: str, cause: Exception | None = None) -> SessionQueryError:
    """Build a SESSION_QUERY_CORRUPT_SESSION error."""
    return SessionQueryError(
        SessionQueryErrorCode.CORRUPT_SESSION,
        message,
        cause=cause,
    )


def index_failed(message: str, cause: Exception | None = None) -> SessionQueryError:
    """Build a SESSION_QUERY_INDEX_FAILED error."""
    return SessionQueryError(
        SessionQueryErrorCode.INDEX_FAILED,
        message,
        cause=cause,
    )
