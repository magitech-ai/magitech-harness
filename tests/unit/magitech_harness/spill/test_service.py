"""Unit tests for the spill storage service seam (ctx.spillStore)."""

from __future__ import annotations

from typing import Any

import pytest
from anima import Context

from magitech_harness.spill.service import (
    SaveTextSpill,
    SpillLocator,
    SpillRef,
    SpillStore,
)


class StubStore(SpillStore):
    """Minimal concrete backend: records the last request, returns a fixed ref."""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(ctx)
        self.last: SaveTextSpill | None = None

    async def saveText(self, input: SaveTextSpill) -> SpillRef:
        self.last = input
        return SpillRef(
            locator=SpillLocator(f"/stub/{input['suggestedName']}"),
            bytes=len(input["content"].encode("utf-8")),
            retrievalHint="Use the stub reader.",
        )


def _request(content: str) -> SaveTextSpill:
    return {
        "owner": {"sessionId": "s1"},
        "source": {
            "kind": "tool",
            "toolName": "web_fetch",
            "callId": "c1",
            "label": "result",
        },
        "suggestedName": "web_fetch.txt",
        "content": content,
    }


@pytest.mark.asyncio
async def test_spill_store_registers_and_saves_text() -> None:
    """The service registers as ctx.spillStore and saves text verbatim."""
    ctx = Context()
    ctx.use(StubStore)
    await ctx.flush()

    store = ctx.get("spillStore", None, strict=False)
    assert isinstance(store, StubStore)
    ref = await store.saveText(_request("hello"))
    assert ref == {
        "locator": "/stub/web_fetch.txt",
        "bytes": 5,
        "retrievalHint": "Use the stub reader.",
    }
    assert store.last is not None
    assert store.last["content"] == "hello"


@pytest.mark.asyncio
async def test_spill_store_rejects_second_implementation() -> None:
    """Loading a second SpillStore implementation fails (one per context)."""
    ctx = Context()
    ctx.use(StubStore)
    await ctx.flush()

    first_store = ctx.get("spillStore", None, strict=False)
    assert isinstance(first_store, StubStore)

    # Second registration attempt must fail (one spillStore per context).
    ctx.use(StubStore)
    await ctx.flush()

    # The original service must still be in place; the failed load is
    # stored on the fiber, not raised.
    assert ctx.get("spillStore", None, strict=False) is first_store


@pytest.mark.asyncio
async def test_spill_store_releases_on_disposal() -> None:
    """Disposing the fiber releases the spillStore service."""
    ctx = Context()
    fiber = ctx.use(StubStore)
    await ctx.flush()

    assert ctx.get("spillStore", None, strict=False) is not None

    await fiber.dispose()
    assert ctx.get("spillStore", None, strict=False) is None


@pytest.mark.asyncio
async def test_spill_store_save_text_raises_not_implemented() -> None:
    """The abstract SpillStore.saveText raises NotImplementedError."""
    ctx = Context()
    store = SpillStore(ctx)
    with pytest.raises(NotImplementedError):
        await store.saveText(_request("hello"))
