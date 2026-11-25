"""Unit tests for edit_file tool component."""

from __future__ import annotations

from pathlib import Path

import pytest
from animas import Context

from magitech_harness.session.events import ToolCallData
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.fs import apply, handle_edit_file
from magitech_harness.tools.service import ToolService


@pytest.mark.asyncio
async def test_handle_edit_file(tmp_path: Path) -> None:
    """Verifies handle_edit_file replaces target text accurately."""
    # Missing args
    res = await handle_edit_file({})
    assert "required" in res

    # Blank path
    res = await handle_edit_file({"path": "   ", "target": "a"})
    assert "required" in res

    # Empty target
    res = await handle_edit_file({"path": str(tmp_path / "x.txt"), "target": ""})
    assert "required" in res

    # Target equals replacement (no-op)
    res = await handle_edit_file(
        {"path": str(tmp_path / "x.txt"), "target": "a", "replacement": "a"}
    )
    assert "must differ" in res

    # Missing file
    res = await handle_edit_file({"path": str(tmp_path / "missing.txt"), "target": "a"})
    assert "File not found" in res

    target = tmp_path / "edit_me.txt"
    target.write_text("Hello World", encoding="utf-8")

    # Target not found
    res = await handle_edit_file(
        {"path": str(target), "target": "Nonexistent", "replacement": "New"}
    )
    assert "Target string not found" in res

    # Successful edit
    res = await handle_edit_file(
        {"path": str(target), "target": "World", "replacement": "Magitech"}
    )
    assert "Successfully updated" in res
    assert target.read_text(encoding="utf-8") == "Hello Magitech"

    # Target matched multiple times
    target.write_text("repeat repeat repeat", encoding="utf-8")
    res_multi = await handle_edit_file(
        {"path": str(target), "target": "repeat", "replacement": "once"}
    )
    assert "matched 3 times" in res_multi
    assert "Must match uniquely" in res_multi


@pytest.mark.asyncio
async def test_edit_component_integration(tmp_path: Path) -> None:
    """Verifies edit_file component mounts and executes via ToolService."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolService)
    ctx.use(apply)
    await ctx.flush()

    assert ctx.tools.get("edit_file") is not None

    test_file = tmp_path / "integration_edit.txt"
    test_file.write_text("before text", encoding="utf-8")

    call_edit = ToolCallData(
        call_id="c1",
        name="edit_file",
        arguments={
            "path": str(test_file),
            "target": "before",
            "replacement": "after",
        },
    )
    res_edit = await ctx.tools.execute(call_edit)
    assert not res_edit.is_error
    assert test_file.read_text(encoding="utf-8") == "after text"


@pytest.mark.asyncio
async def test_edit_component_registers_system_prompt_section() -> None:
    """Verifies the component registers tool:edit_file prompt guidance."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolService)
    ctx.use(apply)
    await ctx.flush()

    assert ctx.systemPrompt.get_section_order("TOOL_EDIT") == 1300


@pytest.mark.asyncio
async def test_edit_component_without_system_prompt_service() -> None:
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


def test_apply_rejects_apply_patch() -> None:
    """The patch tool is not ported; asking for it fails loud at mount."""
    from animas import Context

    from magitech_harness.tools.fs import apply as fs_apply
    from magitech_harness.tools.service import ToolsService

    ctx = Context()
    ToolsService(ctx)
    with pytest.raises(ValueError, match="apply_patch is not ported"):
        fs_apply(ctx, {"apply_patch": True})
