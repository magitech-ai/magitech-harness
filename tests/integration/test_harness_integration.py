from pathlib import Path
from typing import ClassVar

import pytest
from anima import Context, EventMode

from magitech_harness import (
    EventType,
    MockLLMProvider,
    create_harness,
)
from magitech_harness.llm.service import MockLlmAdapter
from magitech_harness.llm.types import ModelChunk
from tests.agent_turn_helper import run_agent_turn


class _MockProviderComponent:
    """Test-only component registering the offline mock provider."""

    inject: ClassVar[list[str]] = ["llm"]

    def __init__(self, ctx: Context) -> None:
        ctx.llm.register_provider("mock", MockLLMProvider())
        ctx.llm.register_adapter(["mock"], MockLlmAdapter())


@pytest.mark.asyncio
async def test_shell_service_execution(tmp_path: Path):
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    code, stdout, _stderr = await ctx.shell.run("echo 'Magitech Rocks'")
    assert code == 0
    assert "Magitech Rocks" in stdout


class _ExplodingStreamProvider:
    """Provider whose stream yields one chunk then raises."""

    async def generate(self, messages, tools=None, system_prompt=None, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("stream exploded")

    async def stream(  # type: ignore[no-untyped-def]
        self,
        messages,
        tools=None,
        system_prompt=None,
        model=None,
        signal=None,
        **kwargs,
    ):
        yield ModelChunk(text="partial output")
        raise RuntimeError("stream exploded")


@pytest.mark.asyncio
async def test_waterfall_guard_pre_step_veto(tmp_path: Path) -> None:
    """A pre-step reject ends the turn ``blocked`` with no step (reference shape)."""
    ctx = create_harness(storage_dir=tmp_path)
    ctx.use(_MockProviderComponent)
    await ctx.flush()

    loader = ctx.get("loader", None, strict=False)
    await loader.disable("session-title-first-prompt")

    async def security_guard(decision: dict, next_fn):
        # The unified driver's pre-step payload carries ``kind`` and
        # per-step data; a reject short-circuits before any step opens.
        if decision.get("step") == 1:
            return {**decision, "kind": "reject"}
        return await next_fn(decision)

    ctx.on("agent/pre-step", security_guard, mode=EventMode.WATERFALL)

    result = await run_agent_turn(
        ctx, "Malicious instruction", session_id="blocked_session"
    )

    assert result.text == ""
    # Reference: blocked, and no step/start or step/end is logged.
    assert result.reason is not None
    assert result.reason.get("kind") == "blocked"
    event_types = [ev.type for ev in result.session.events]
    assert EventType.STEP_START not in event_types
    assert EventType.STEP_END not in event_types
