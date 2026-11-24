"""Unit tests for the LocalSpillStore filesystem backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from animas import Context

from magitech_harness.spill.local import LocalSpillStore
from magitech_harness.spill.store import sessionDir


def _request() -> dict[str, object]:
    return {
        "owner": {"sessionId": "sess-1"},
        "source": {
            "kind": "tool",
            "toolName": "web_fetch",
            "callId": "call-1",
            "label": "result",
        },
        "suggestedName": "web_fetch.txt",
        "content": "the full body",
    }


@pytest.mark.asyncio
async def test_local_spill_store_registers_and_saves(tmp_path: Path) -> None:
    """LocalSpillStore registers as ctx.spillStore and saves under root."""
    ctx = Context()
    ctx.use(LocalSpillStore, {"root": str(tmp_path), "cleanupPeriodDays": 0})
    await ctx.flush()

    store = ctx.get("spillStore", None, strict=False)
    assert isinstance(store, LocalSpillStore)
    ref = await store.saveText(_request())  # type: ignore[union-attr]
    assert str(Path(ref["locator"]).parent) == sessionDir(str(tmp_path), "sess-1")
    assert Path(ref["locator"]).read_text(encoding="utf-8") == "the full body"
    assert ref["bytes"] == len(b"the full body")
    assert (
        ref["retrievalHint"]
        == "Use read with offset/limit, or grep this path to search within it."
    )


@pytest.mark.asyncio
async def test_local_spill_store_resolves_relative_root(tmp_path: Path) -> None:
    """A relative configured root is resolved to absolute."""
    ctx = Context()
    ctx.use(LocalSpillStore, {"root": ".", "cleanupPeriodDays": 0})
    await ctx.flush()

    store = ctx.get("spillStore", None, strict=False)
    assert isinstance(store, LocalSpillStore)
    assert os.path.isabs(store.root)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_local_spill_store_falls_back_to_private_root() -> None:
    """With no root configured, LocalSpillStore uses privateRoot()."""
    ctx = Context()
    ctx.use(LocalSpillStore, {"cleanupPeriodDays": 0})
    await ctx.flush()

    store = ctx.get("spillStore", None, strict=False)
    assert isinstance(store, LocalSpillStore)
    from magitech_harness.spill.store import privateRoot

    assert store.root == privateRoot()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_local_spill_store_rejects_unwritable_root(tmp_path: Path) -> None:
    """saveText fails when the root is a file (not a directory)."""
    ctx = Context()

    # A file (not a dir) as the root makes mkdir under it fail.
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x")

    ctx.use(LocalSpillStore, {"root": str(file_path), "cleanupPeriodDays": 0})
    await ctx.flush()

    store = ctx.get("spillStore", None, strict=False)
    assert isinstance(store, LocalSpillStore)
    with pytest.raises(OSError):
        await store.saveText(_request())  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_local_spill_store_rejects_bad_cleanup_days() -> None:
    """Negative or fractional cleanupPeriodDays fails at load."""
    ctx = Context()
    with pytest.raises(ValueError, match="cleanupPeriodDays"):
        LocalSpillStore(ctx, {"root": "/tmp/test", "cleanupPeriodDays": -1})
    ctx2 = Context()
    with pytest.raises(ValueError, match="cleanupPeriodDays"):
        LocalSpillStore(ctx2, {"root": "/tmp/test", "cleanupPeriodDays": 1.5})


@pytest.mark.asyncio
async def test_local_spill_store_defaults_cleanup_days(tmp_path: Path) -> None:
    """cleanupPeriodDays defaults to 30 when omitted."""
    ctx = Context()
    ctx.use(LocalSpillStore, {"root": str(tmp_path)})
    await ctx.flush()

    store = ctx.get("spillStore", None, strict=False)
    assert isinstance(store, LocalSpillStore)
    assert store.config["cleanupPeriodDays"] == 30  # type: ignore[union-attr]
