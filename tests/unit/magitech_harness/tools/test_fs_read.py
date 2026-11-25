"""Unit tests for read_file tool component."""

from __future__ import annotations

from pathlib import Path

import pytest
from animas import Context

from magitech_harness.session.events import ToolCallData
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.fs import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    _validate_limit,
    apply,
    handle_read_file,
)
from magitech_harness.tools.service import ToolService


@pytest.mark.asyncio
async def test_handle_read_file_validation(tmp_path: Path) -> None:
    """Verifies argument validation and error cases for handle_read_file."""
    # Missing path
    res = await handle_read_file({})
    assert "required" in res

    # Invalid offset
    res = await handle_read_file({"path": "foo.txt", "offset": 0})
    assert "Invalid offset" in res

    # Limit below 1 raises
    with pytest.raises(ValueError, match="Must be 1 or greater"):
        await handle_read_file({"path": "foo.txt", "limit": 0})

    # Nonexistent file
    res = await handle_read_file({"path": str(tmp_path / "nonexistent.txt")})
    assert "File not found" in res

    # Directory without SKILL.md
    dir_path = tmp_path / "empty_dir"
    dir_path.mkdir()
    res = await handle_read_file({"path": str(dir_path)})
    assert "Cannot read directory" in res


@pytest.mark.asyncio
async def test_handle_read_file_limit_validation() -> None:
    """Verifies limit resolution: default, clamp, and validation."""
    # No limit -> default
    assert _validate_limit(None, DEFAULT_LIMIT, MAX_LIMIT) == DEFAULT_LIMIT
    # Non-int limit -> default
    assert _validate_limit("not-a-number", DEFAULT_LIMIT, MAX_LIMIT) == DEFAULT_LIMIT
    # Limit within bounds passes
    assert _validate_limit(100, DEFAULT_LIMIT, MAX_LIMIT) == 100
    # Limit below 1 raises
    with pytest.raises(ValueError, match="Must be 1 or greater"):
        _validate_limit(0, DEFAULT_LIMIT, MAX_LIMIT)
    # Limit above max raises
    with pytest.raises(ValueError, match=f"Must not exceed {MAX_LIMIT}"):
        _validate_limit(MAX_LIMIT + 1, DEFAULT_LIMIT, MAX_LIMIT)


@pytest.mark.asyncio
async def test_handle_read_file_default_limit_applied(tmp_path: Path) -> None:
    """Verifies default_limit caps reads when no limit is supplied."""
    text_file = tmp_path / "many_lines.txt"
    lines = [f"line {i}" for i in range(DEFAULT_LIMIT + 50)]
    text_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = await handle_read_file({"path": str(text_file)})
    result_lines = res.split("\n")
    assert len(result_lines) == DEFAULT_LIMIT


@pytest.mark.asyncio
async def test_handle_read_file_config_limits(tmp_path: Path) -> None:
    """Verifies config-driven default_limit and max_limit."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    # Custom default_limit smaller than file
    res = await handle_read_file(
        {"path": str(text_file)}, default_limit=2, max_limit=100
    )
    assert res == "line1\nline2"

    # Custom max_limit: requesting above it raises
    with pytest.raises(ValueError, match="Must not exceed 5"):
        await handle_read_file(
            {"path": str(text_file), "limit": 10}, default_limit=100, max_limit=5
        )


@pytest.mark.asyncio
async def test_handle_read_file_skill_dir_and_binary(tmp_path: Path) -> None:
    """Verifies reading skill directory with SKILL.md and binary detection."""
    # Directory with SKILL.md
    dir_path = tmp_path / "skill_dir"
    dir_path.mkdir()
    (dir_path / "SKILL.md").write_text("Skill Content", encoding="utf-8")
    res = await handle_read_file({"path": str(dir_path)})
    assert res == "Skill Content"

    # Binary file with null bytes
    bin_file = tmp_path / "binary.bin"
    bin_file.write_bytes(b"hello\x00world")
    res = await handle_read_file({"path": str(bin_file)})
    assert "Binary file not supported" in res

    # Binary file with invalid UTF-8
    bad_utf8 = tmp_path / "bad.txt"
    bad_utf8.write_bytes(b"\xff\xfe\xfd")
    res = await handle_read_file({"path": str(bad_utf8)})
    assert "Binary file not supported" in res


@pytest.mark.asyncio
async def test_handle_read_file_pagination(tmp_path: Path) -> None:
    """Verifies pagination offsets and limits for handle_read_file."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    # Read full file (under default limit)
    res = await handle_read_file({"path": str(text_file)})
    assert res == "line1\nline2\nline3\nline4"

    # Read with offset
    res = await handle_read_file({"path": str(text_file), "offset": 2})
    assert res == "line2\nline3\nline4"

    # Read with offset and limit
    res = await handle_read_file({"path": str(text_file), "offset": 2, "limit": 2})
    assert res == "line2\nline3"

    # Offset beyond EOF
    res = await handle_read_file({"path": str(text_file), "offset": 10})
    assert "beyond end of file" in res


@pytest.mark.asyncio
async def test_read_component_integration(tmp_path: Path) -> None:
    """Verifies read_file component mounts and executes via ToolService."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolService)
    ctx.use(apply)
    await ctx.flush()

    assert ctx.tools.get("read_file") is not None

    test_file = tmp_path / "integration_read.txt"
    test_file.write_text("Read me successfully", encoding="utf-8")

    call_read = ToolCallData(
        call_id="c1",
        name="read_file",
        arguments={"path": str(test_file)},
    )
    res_read = await ctx.tools.execute(call_read)
    assert not res_read.is_error
    assert res_read.content == "Read me successfully"


@pytest.mark.asyncio
async def test_read_component_registers_system_prompt_section(tmp_path: Path) -> None:
    """Verifies the component registers tool:read_file prompt guidance."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolService)
    ctx.use(apply)
    await ctx.flush()

    # Section order should resolve to TOOL_READ (1100)
    assert ctx.systemPrompt.get_section_order("TOOL_READ") == 1100


@pytest.mark.asyncio
async def test_read_component_without_system_prompt_service(tmp_path: Path) -> None:
    """Verifies the component mounts and skips section when systemPrompt absent."""
    ctx = Context()
    # Register a minimal tools service directly so apply's inject=["tools"] is
    # satisfied without mounting ToolService (which requires systemPrompt).
    fake_tools = type(
        "FakeTools",
        (),
        {"register": lambda self, tool: None},
    )()
    ctx.reflect.provide("tools", fake_tools, None)
    await ctx.flush()

    ctx.use(apply)
    await ctx.flush()

    # Component ran without crashing; section registration was skipped.
    assert ctx.has("tools")
