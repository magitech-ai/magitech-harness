"""Unit tests for session query errors."""

from __future__ import annotations

from magitech_harness.session_query.errors import (
    SessionQueryError,
    SessionQueryErrorCode,
    abort,
    corrupt_session,
    event_not_found,
    index_failed,
    invalid_config,
    invalid_cursor,
    invalid_filter,
    invalid_limit,
    invalid_lineage,
    invalid_query,
    invalid_surface,
    invalid_window,
    persistence_failed,
    search_disabled,
    session_not_found,
    source_conflict,
    stale_cursor,
)


def test_error_codes_are_strings() -> None:
    """Every error code is a non-empty string."""
    for code in SessionQueryErrorCode.all():
        assert isinstance(code, str)
        assert code


def test_error_code_values() -> None:
    """Known error codes match the expected taxonomy."""
    assert SessionQueryErrorCode.SESSION_NOT_FOUND == "SESSION_QUERY_SESSION_NOT_FOUND"
    assert SessionQueryErrorCode.INVALID_FILTER == "SESSION_QUERY_INVALID_FILTER"
    assert SessionQueryErrorCode.INVALID_SURFACE == "SESSION_QUERY_INVALID_SURFACE"
    assert (
        SessionQueryErrorCode.PERSISTENCE_FAILED == "SESSION_QUERY_PERSISTENCE_FAILED"
    )
    assert SessionQueryErrorCode.SOURCE_CONFLICT == "SESSION_QUERY_SOURCE_CONFLICT"


def test_session_query_error_construction() -> None:
    """SessionQueryError carries code, message, and optional cause."""
    err = SessionQueryError(SessionQueryErrorCode.SESSION_NOT_FOUND, "no such session")
    assert err.code == SessionQueryErrorCode.SESSION_NOT_FOUND
    assert str(err) == "no such session"
    assert err.cause is None

    cause = ValueError("underlying")
    err2 = SessionQueryError(
        SessionQueryErrorCode.PERSISTENCE_FAILED, "save failed", cause=cause
    )
    assert err2.cause is cause


def test_session_not_found() -> None:
    err = session_not_found("session-123")
    assert err.code == SessionQueryErrorCode.SESSION_NOT_FOUND
    assert "session-123" in str(err)


def test_event_not_found() -> None:
    err = event_not_found("session-1", 5)
    assert err.code == SessionQueryErrorCode.EVENT_NOT_FOUND
    assert "session-1" in str(err)
    assert "5" in str(err)


def test_invalid_filter() -> None:
    err = invalid_filter("bad filter")
    assert err.code == SessionQueryErrorCode.INVALID_FILTER


def test_invalid_query() -> None:
    err = invalid_query("bad query")
    assert err.code == SessionQueryErrorCode.INVALID_QUERY


def test_invalid_limit() -> None:
    err = invalid_limit("limit too big")
    assert err.code == SessionQueryErrorCode.INVALID_LIMIT


def test_invalid_window() -> None:
    err = invalid_window("window too big")
    assert err.code == SessionQueryErrorCode.INVALID_WINDOW


def test_invalid_surface() -> None:
    err = invalid_surface("bad surface")
    assert err.code == SessionQueryErrorCode.INVALID_SURFACE


def test_invalid_lineage() -> None:
    err = invalid_lineage("bad lineage")
    assert err.code == SessionQueryErrorCode.INVALID_LINEAGE


def test_persistence_failed() -> None:
    err = persistence_failed("disk full")
    assert err.code == SessionQueryErrorCode.PERSISTENCE_FAILED
    assert err.cause is None

    cause = OSError("no space")
    err2 = persistence_failed("disk full", cause=cause)
    assert err2.cause is cause


def test_source_conflict() -> None:
    err = source_conflict("headers disagree")
    assert err.code == SessionQueryErrorCode.SOURCE_CONFLICT


def test_abort() -> None:
    err = abort("operation cancelled")
    assert err.code == SessionQueryErrorCode.ABORTED


def test_invalid_config() -> None:
    err = invalid_config("bad config")
    assert err.code == SessionQueryErrorCode.INVALID_CONFIG


def test_invalid_cursor() -> None:
    err = invalid_cursor("stale cursor")
    assert err.code == SessionQueryErrorCode.INVALID_CURSOR


def test_invalid_lineage_code() -> None:
    err = invalid_lineage("cycle")
    assert err.code == SessionQueryErrorCode.INVALID_LINEAGE


def test_search_disabled() -> None:
    err = search_disabled("no backend")
    assert err.code == SessionQueryErrorCode.SEARCH_DISABLED


def test_stale_cursor() -> None:
    err = stale_cursor("cursor expired")
    assert err.code == SessionQueryErrorCode.STALE_CURSOR


def test_corrupt_session() -> None:
    err = corrupt_session("bad log")
    assert err.code == SessionQueryErrorCode.CORRUPT_SESSION


def test_index_failed() -> None:
    err = index_failed("index build failed")
    assert err.code == SessionQueryErrorCode.INDEX_FAILED
