"""Integration tests: titles round-trip through real persistence."""

from pathlib import Path
from typing import Any

import pytest

from magitech_harness import create_harness
from magitech_harness.session.store import Session


def _user(ctx: Any, session: Session, text: str) -> None:
    """Append a v4-shaped eligible human message to ``session``."""
    ctx.sessions.log_event(
        session,
        "user/message",
        {
            "content": [{"type": "text", "text": text}],
            "source": {"kind": "user", "channel": "integration"},
        },
    )


@pytest.mark.asyncio
async def test_title_survives_restart_and_rename_pins(tmp_path: Path):
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("s1")
    _user(ctx, session, "How do I reset my password for the dashboard?")
    await ctx.sessionTitle.drain()
    snapshot = ctx.sessionTitle.get(session)
    assert snapshot is not None
    assert snapshot["title"] == "How do I reset my"
    assert snapshot["source"] == {"kind": "fallback"}

    renamed = ctx.sessionTitle.rename(session, "My Renamed Session")
    assert renamed["title"] == "My Renamed Session"

    # Subsequent turns must not move the pinned title.
    _user(ctx, session, "Also, where do I change my billing address?")
    _user(ctx, session, "And can I export my data as CSV?")
    await ctx.sessionTitle.drain()
    assert ctx.sessionTitle.get(session)["title"] == "My Renamed Session"

    # The title lives in the log: flush, release the write handle,
    # restart, reload.
    await ctx.sessions.flush()
    assert ctx.sessions.close("s1") is True
    ctx2 = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx2.flush()
    loaded = ctx2.sessions.load("s1")
    assert ctx2.sessionTitle.get(loaded)["title"] == "My Renamed Session"
    title_events = [event for event in loaded.events if event.type == "session/title"]
    assert [event.payload["title"] for event in title_events] == [
        "How do I reset my",
        "My Renamed Session",
    ]


@pytest.mark.asyncio
async def test_fallback_is_deterministic_across_sessions(tmp_path: Path):
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    first = ctx.sessions.create("first")
    second = ctx.sessions.create("second")
    _user(ctx, first, "How do I reset my password for the dashboard?")
    _user(ctx, second, "How do I reset my password for the dashboard?")
    await ctx.sessionTitle.drain()
    # Only the derived fields are deterministic; eventSeq/updatedAt differ.
    for field in ("title", "messageSeqs", "source"):
        assert (
            ctx.sessionTitle.get(first)[field] == (ctx.sessionTitle.get(second)[field])
        )
    await ctx.sessions.flush()
    assert ctx.sessions.close("first") is True
    assert ctx.sessions.close("second") is True
    ctx2 = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx2.flush()
    for field in ("title", "messageSeqs", "source"):
        assert (
            ctx2.sessionTitle.get(ctx2.sessions.load("first"))[field]
            == ctx2.sessionTitle.get(ctx2.sessions.load("second"))[field]
        )
