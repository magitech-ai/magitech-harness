"""Unit tests for the compaction service."""

from __future__ import annotations

from typing import Any

import pytest
from anima import Context

from magitech_harness.compaction.service import CompactionService


def test_compaction_service_registers_engines_and_pruners() -> None:
    ctx = Context()
    svc = CompactionService(ctx)
    disposer = svc.register_engine("basic", lambda session, **kw: {"compacted": True})
    svc.register_pruner("trim", lambda session, **kw: {"pruned": True})
    assert svc.list_engines() == ["basic"]
    assert svc.list_pruners() == ["trim"]
    disposer()
    assert svc.list_engines() == []


def test_compaction_service_rejects_duplicates_and_empty_names() -> None:
    ctx = Context()
    svc = CompactionService(ctx)
    svc.register_engine("basic", lambda session, **kw: {})
    with pytest.raises(ValueError, match="already registered"):
        svc.register_engine("basic", lambda session, **kw: {})
    with pytest.raises(ValueError, match="non-empty"):
        svc.register_engine("   ", lambda session, **kw: {})
    svc.register_pruner("trim", lambda session, **kw: {})
    with pytest.raises(ValueError, match="already registered"):
        svc.register_pruner("trim", lambda session, **kw: {})
    with pytest.raises(ValueError, match="non-empty"):
        svc.register_pruner("   ", lambda session, **kw: {})


@pytest.mark.asyncio
async def test_compaction_service_runs_engines_and_pruners() -> None:
    ctx = Context()
    svc = CompactionService(ctx)

    async def async_engine(session: Any, **kw: Any) -> dict[str, Any]:
        return {"compacted": True, "async": True}

    def failing_engine(session: Any, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    async def async_pruner(session: Any, **kw: Any) -> dict[str, Any]:
        return {"pruned": True, "async": True}

    def failing_pruner(session: Any, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    svc.register_engine("basic", lambda session, **kw: {"compacted": True})
    svc.register_engine("async", async_engine)
    svc.register_engine("failing", failing_engine)
    svc.register_engine("non-dict", lambda session, **kw: "not a dict")
    svc.register_pruner("trim", lambda session, **kw: {"pruned": True})
    svc.register_pruner("async", async_pruner)
    svc.register_pruner("failing", failing_pruner)
    svc.register_pruner("non-dict", lambda session, **kw: "not a dict")
    result = await svc.compact("sess")
    assert result["compacted"] is True
    assert result["async"] is True
    pruned = await svc.prune("sess")
    assert pruned["pruned"] is True
    assert pruned["async"] is True

    # dispose covers teardown branches
    disposer = svc.register_engine("extra", lambda session, **kw: {})
    disposer()
    assert "extra" not in svc.list_engines()
    pruner_disposer = svc.register_pruner("extra-pruner", lambda session, **kw: {})
    pruner_disposer()
    assert "extra-pruner" not in svc.list_pruners()


def test_compaction_service_mounts_in_default_bundle() -> None:
    from magitech_harness import create_harness

    ctx = create_harness()
    assert ctx.get("compaction", None, strict=False) is not None
