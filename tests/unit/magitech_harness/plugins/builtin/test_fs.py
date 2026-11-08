"""Unit tests for filesystem tools plugin."""

from __future__ import annotations

from pathlib import Path

import pytest
from anima import Context

from magitech_harness.plugins.builtin.fs import (
    apply,
    handle_edit_file,
    handle_read_file,
    handle_write_file,
)
from magitech_harness.session.events import ToolCallData
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

    # Invalid limit
    res = await handle_read_file({"path": "foo.txt", "limit": 0})
    assert "Invalid limit" in res

    # Nonexistent file
    res = await handle_read_file({"path": str(tmp_path / "nonexistent.txt")})
    assert "File not found" in res

    # Directory without SKILL.md
    dir_path = tmp_path / "empty_dir"
    dir_path.mkdir()
    res = await handle_read_file({"path": str(dir_path)})
    assert "Cannot read directory" in res


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

    # Read full file
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
async def test_handle_write_file(tmp_path: Path) -> None:
    """Verifies handle_write_file creates nested dirs and writes content."""
    # Missing path
    res = await handle_write_file({})
    assert "required" in res

    target = tmp_path / "nested" / "sub" / "output.txt"
    res = await handle_write_file({"path": str(target), "content": "Magitech rules"})
    assert "Successfully wrote" in res
    assert target.read_text(encoding="utf-8") == "Magitech rules"


@pytest.mark.asyncio
async def test_handle_edit_file(tmp_path: Path) -> None:
    """Verifies handle_edit_file replaces target text accurately."""
    # Missing args
    res = await handle_edit_file({})
    assert "required" in res

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


@pytest.mark.asyncio
async def test_fs_plugin_integration(tmp_path: Path) -> None:
    """Verifies fs plugin mounts tools and executes them via ToolService."""
    ctx = Context()
    ctx.plugin(ToolService)
    ctx.plugin(apply)
    await ctx.flush()

    assert ctx.tools.get("read_file") is not None
    assert ctx.tools.get("write_file") is not None
    assert ctx.tools.get("edit_file") is not None

    test_file = tmp_path / "integration_test.txt"

    # Call write_file
    call_write = ToolCallData(
        call_id="c1",
        name="write_file",
        arguments={"path": str(test_file), "content": "Integration testing"},
    )
    res_write = await ctx.tools.execute(call_write)
    assert not res_write.is_error

    # Call read_file
    call_read = ToolCallData(
        call_id="c2",
        name="read_file",
        arguments={"path": str(test_file)},
    )
    res_read = await ctx.tools.execute(call_read)
    assert not res_read.is_error
    assert res_read.content == "Integration testing"
