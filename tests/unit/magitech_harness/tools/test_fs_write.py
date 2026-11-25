"""Unit tests for write_file tool component."""

from __future__ import annotations

from pathlib import Path

import pytest
from animas import Context

from magitech_harness.session.events import ToolCallData
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.fs import apply, handle_write_file
from magitech_harness.tools.service import ToolService


@pytest.mark.asyncio
async def test_handle_write_file(tmp_path: Path) -> None:
    """Verifies handle_write_file creates nested dirs and writes content."""
    # Missing path
    res = await handle_write_file({})
    assert "required" in res

    # Blank (whitespace-only) path is rejected
    res = await handle_write_file({"path": "   ", "content": "x"})
    assert "required" in res

    target = tmp_path / "nested" / "sub" / "output.txt"
    res = await handle_write_file({"path": str(target), "content": "Magitech rules"})
    assert "Successfully wrote" in res
    assert target.read_text(encoding="utf-8") == "Magitech rules"


@pytest.mark.asyncio
async def test_handle_write_file_empty_content(tmp_path: Path) -> None:
    """Verifies empty content is legitimate (writes an empty file)."""
    target = tmp_path / "empty.txt"
    res = await handle_write_file({"path": str(target), "content": ""})
    assert "Successfully wrote" in res
    assert target.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_write_component_integration(tmp_path: Path) -> None:
    """Verifies write_file component mounts and executes via ToolService."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolService)
    ctx.use(apply)
    await ctx.flush()

    assert ctx.tools.get("write_file") is not None

    test_file = tmp_path / "integration_write.txt"
    call_write = ToolCallData(
        call_id="c1",
        name="write_file",
        arguments={"path": str(test_file), "content": "Integration write testing"},
    )
    res_write = await ctx.tools.execute(call_write)
    assert not res_write.is_error
    assert test_file.read_text(encoding="utf-8") == "Integration write testing"


@pytest.mark.asyncio
async def test_write_component_registers_system_prompt_section() -> None:
    """Verifies the component registers tool:write_file prompt guidance."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolService)
    ctx.use(apply)
    await ctx.flush()

    assert ctx.systemPrompt.get_section_order("TOOL_WRITE") == 1200


@pytest.mark.asyncio
async def test_write_component_without_system_prompt_service() -> None:
    """Verifies the component mounts and skips section when systemPrompt absent."""
    fake_tools = type(
        "FakeTools",
        (),
        {"register": lambda self, tool: None},
    )()
    ctx = Context()
    ctx.reflect.provide("tools", fake_tools, None)
    await ctx.flush()

    ctx.use(apply)
    await ctx.flush()

    assert ctx.has("tools")
