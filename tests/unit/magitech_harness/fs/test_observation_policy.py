"""Unit tests for the fs-observation-policy plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from anima import Context

from magitech_harness.fs.observation_policy import apply as apply_observation_policy
from magitech_harness.fs.types import FsError, FsTarget


@dataclass
class _FakeSession:
    name: str


@dataclass
class _FakeAgent:
    session: _FakeSession


@pytest.mark.asyncio
async def test_observation_policy_gates_writes_edits_and_deletes() -> None:
    """Enforces prior observation per session owner and drops state on dispose."""
    ctx = Context()
    ctx.use(apply_observation_policy)
    await ctx.flush()

    target: FsTarget = {"targetKey": "/repo/a.txt", "displayPath": "/repo/a.txt"}
    session = _FakeSession("s1")
    actor: dict[str, Any] = {"agent": _FakeAgent(session)}

    # Unseen target -> write createIfAbsent; edit/delete fail FS_NOT_OBSERVED
    assert (
        await ctx.events.waterfall(
            "fs/write-intent", target, actor, default_handler=lambda *_a: None
        )
    ) == {"kind": "createIfAbsent"}

    with pytest.raises(FsError) as edit_unseen:
        await ctx.events.waterfall(
            "fs/edit-intent", target, actor, default_handler=lambda *_a: None
        )
    assert edit_unseen.value.code == "FS_NOT_OBSERVED"

    # Confirmed absent -> write createIfAbsent; delete fails FS_NOT_FOUND
    ctx.events.emit("fs/observed", target, {"kind": "absent"}, actor)
    assert (
        await ctx.events.waterfall(
            "fs/write-intent", target, actor, default_handler=lambda *_a: None
        )
    ) == {"kind": "createIfAbsent"}

    with pytest.raises(FsError) as del_absent:
        await ctx.events.waterfall(
            "fs/delete-intent", target, actor, default_handler=lambda *_a: None
        )
    assert del_absent.value.code == "FS_NOT_FOUND"

    # Confirmed present -> write replaceIfVersion; edit/delete carry version
    ctx.events.emit("fs/observed", target, {"kind": "present", "version": "v1"}, actor)
    assert (
        await ctx.events.waterfall(
            "fs/write-intent", target, actor, default_handler=lambda *_a: None
        )
    ) == {"kind": "replaceIfVersion", "version": "v1"}
    assert (
        await ctx.events.waterfall(
            "fs/edit-intent", target, actor, default_handler=lambda *_a: None
        )
    ) == {"version": "v1"}
    assert (
        await ctx.events.waterfall(
            "fs/delete-intent", target, actor, default_handler=lambda *_a: None
        )
    ) == {"version": "v1"}

    # No actor owner -> reads freely, write createIfAbsent, edit FS_NOT_OBSERVED
    ctx.events.emit("fs/observed", target, {"kind": "present", "version": "v2"}, None)
    with pytest.raises(FsError) as edit_no_actor:
        await ctx.events.waterfall(
            "fs/edit-intent", target, None, default_handler=lambda *_a: None
        )
    assert edit_no_actor.value.code == "FS_NOT_OBSERVED"

    # Null actor / null agent -> treated as unowned
    assert (
        await ctx.events.waterfall(
            "fs/write-intent", target, {"agent": None}, default_handler=lambda *_a: None
        )
    ) == {"kind": "createIfAbsent"}

    # Missing/stale weakref bucket branch: clear -> re-check via dead weakref id reuse fallback
    class _AttrActor:
        def __init__(self, ag: _FakeAgent) -> None:
            self.agent = ag

    attr_actor = _AttrActor(_FakeAgent(session))
    assert (
        await ctx.events.waterfall(
            "fs/delete-intent", target, attr_actor, default_handler=lambda *_a: None
        )
    ) == {"version": "v1"}

    # Disposal drops state and listeners
    await ctx.fiber.dispose()
    assert (
        await ctx.events.waterfall(
            "fs/write-intent", target, actor, default_handler=lambda *_a: "uncond"
        )
        == "uncond"
    )
