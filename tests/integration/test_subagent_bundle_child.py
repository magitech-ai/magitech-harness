"""End-to-end: a subagent turn runs on a bundle-built child context.

The child context is built by the bundle loader, not by direct service
construction; the turn runs through the real agent loop against a mock
provider, and the child session persists under the subsessions dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from animas import Context

from magitech_harness import check_bundle_entries, create_harness
from magitech_harness.llm.service import MockLlmAdapter, MockLLMProvider
from magitech_harness.subagent.manager import SubagentManager
from magitech_harness.subagent.types import SubagentSpec, SubagentState
from magitech_harness.tools.schema import define_tool


@pytest.mark.asyncio
async def test_subagent_turn_on_bundle_child_persists(tmp_path: Path) -> None:
    parent_ctx = create_harness(storage_dir=tmp_path / "parent")
    await parent_ctx.flush()
    check_bundle_entries(parent_ctx)

    mock_provider = MockLLMProvider()
    parent_ctx.llm.register_provider("mock", mock_provider)
    parent_ctx.llm.set_active_provider("mock")
    parent_ctx.tools.register(
        define_tool(
            name="probe_read", description="", parameters={}, execute=AsyncMock()
        )
    )
    parent_ctx.tools.register(
        define_tool(
            name="probe_write", description="", parameters={}, execute=AsyncMock()
        )
    )
    parent_ctx.tools.register(
        define_tool(
            name="invoke_subagent", description="", parameters={}, execute=AsyncMock()
        )
    )

    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})

    captured: list[Context] = []

    def _recording_create_harness(
        storage_dir: Path | str | None = None, **kwargs: object
    ) -> Context:
        ctx = create_harness(storage_dir=storage_dir)
        captured.append(ctx)

        # Patch the unified driver's agent creation at flush time so the
        # turn is deterministic (the legacy mock provider is copied by the
        # manager for provider-seam assertions).
        class _PatchAgents:
            inject: ClassVar[list[str]] = ["agents", "llm"]

            def __init__(self, c: Any) -> None:
                c.llm.register_adapter(["mock"], MockLlmAdapter())

        ctx.use(_PatchAgents)
        return ctx

    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_recording_create_harness,
    ):
        recs = await manager.invoke(
            [
                SubagentSpec(
                    role="worker",
                    prompt="Say hello",
                    allowed_tools=frozenset({"probe_read"}),
                    wait=True,
                )
            ]
        )

    assert len(recs) == 1
    rec = recs[0]
    assert rec.state == SubagentState.COMPLETED
    assert rec.summary == "Default mock response"
    assert rec.effective_tools == ["probe_read"]

    assert len(captured) == 1
    child_ctx = captured[0]

    # Providers copied through the public seams; active provider preserved.
    assert set(child_ctx.llm.providers) == {"mock"}
    assert child_ctx.llm.active_provider is mock_provider

    # The allowed set filters copied parent tools, and the delegation triad
    # never reaches a child. Base tools belong to the installed harness release,
    # so their presence is not asserted against a specific version.
    assert "probe_read" in rec.effective_tools
    assert "probe_write" not in rec.effective_tools
    assert child_ctx.tools.get("invoke_subagent") is None

    # The bundle-built child carries rows the old child never had.
    assert hasattr(child_ctx, "sessionTitle")
    assert hasattr(child_ctx, "sessionProjections")
    assert hasattr(child_ctx, "sessionProjectionCache")

    # The child session persists under the subsessions dir.
    await child_ctx.sessions.flush()
    assert rec.session_path.exists()
    assert tmp_path / "subsessions" in rec.session_path.parents
