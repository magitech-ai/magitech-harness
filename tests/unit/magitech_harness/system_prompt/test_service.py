"""Unit tests for SystemPromptService."""

import asyncio
import pytest
from anima import Context
from magitech_harness.system_prompt.service import SystemPromptService


@pytest.mark.asyncio
async def test_system_prompt_render_static_and_callables() -> None:
    """Verifies rendering static strings, sync callables, and async callables."""
    ctx = Context()
    service = SystemPromptService(ctx)

    # Static string
    service.register_section("header", "# System Persona: Magitech AI")

    # Sync callable
    service.register_section("time", lambda: "Time: 2026")

    # Async callable
    async def get_env_info():
        await asyncio.sleep(0.001)
        return "OS: Linux"

    service.register_section("env", get_env_info)

    rendered = await service.render()
    assert "# System Persona: Magitech AI" in rendered
    assert "Time: 2026" in rendered
    assert "OS: Linux" in rendered


@pytest.mark.asyncio
async def test_system_prompt_section_unregister() -> None:
    """Verifies that disposing a section removes it from the rendered prompt."""
    ctx = Context()
    service = SystemPromptService(ctx)

    disposer = service.register_section("temporary", "Temp Instruction")
    assert "Temp Instruction" in await service.render()

    disposer()
    assert "Temp Instruction" not in await service.render()

    # Second call is idempotent (branch 26->exit)
    disposer()
