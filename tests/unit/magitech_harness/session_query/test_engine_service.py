"""Unit tests for session query engine and service."""

from __future__ import annotations

import pytest
from anima import Context

from magitech_harness.session_query.engine import SessionQueryEngine
from magitech_harness.session_query.service import SessionQueryService


@pytest.mark.asyncio
async def test_engine_list_sessions_no_service() -> None:
    """listSessions returns empty list when no session service is registered."""
    ctx = Context()
    engine = SessionQueryEngine(ctx)
    result = await engine.listSessions()
    assert result == []


@pytest.mark.asyncio
async def test_engine_filter_sessions_no_service() -> None:
    """filterSessions returns empty list when no session service is registered."""
    ctx = Context()
    engine = SessionQueryEngine(ctx)
    result = await engine.filterSessions([{"kind": "id", "values": ["s1"]}])
    assert result == []


@pytest.mark.asyncio
async def test_engine_read_title_snapshots_empty() -> None:
    """readTitleSnapshots returns empty list for empty input."""
    ctx = Context()
    engine = SessionQueryEngine(ctx)
    result = await engine.readTitleSnapshots([])
    assert result == []


def test_service_registration() -> None:
    """SessionQueryService registers ctx.sessionQuery."""
    ctx = Context()
    SessionQueryService(ctx)
    assert ctx.get("sessionQuery", None, strict=False) is not None


def test_service_registration_duplicate_raises() -> None:
    """Registering SessionQueryService twice raises RuntimeError."""
    ctx = Context()
    SessionQueryService(ctx)
    try:
        SessionQueryService(ctx)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
