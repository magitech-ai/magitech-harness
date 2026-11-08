"""Integration test for portable plugin seam in Magitech Harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from anima import FiberState

from magitech_harness import create_harness
from magitech_harness.plugins.builtin import fs_tools, truncation_guard
from magitech_harness.session.events import ToolCallData


@pytest.mark.asyncio
async def test_plugin_seam_full_lifecycle(tmp_path: Path) -> None:
    """Verifies end-to-end discovery, mounting, tool execution, and guardrails."""
    # 1. Create harness with mock provider
    ctx = create_harness(mock_provider=True)
    await ctx.flush()

    assert ctx.has("plugins")
    assert ctx.has("tools")
    assert ctx.has("llm")

    # 2. Mount built-in fs_tools and truncation_guard plugins
    ctx.plugin(fs_tools)
    ctx.plugin(truncation_guard, max_lines=3, max_bytes=100)
    await ctx.flush()

    assert ctx.tools.get("read_file") is not None
    assert ctx.tools.get("write_file") is not None
    assert ctx.tools.get("edit_file") is not None

    # 3. Create a third-party plugin directory on disk with manifest.json
    custom_plugin_dir = tmp_path / "custom_math"
    custom_plugin_dir.mkdir()

    plugin_code = (
        "from magitech_harness.tools.definition import define_tool\n"
        "def apply(ctx):\n"
        "    tool = define_tool(\n"
        "        name='add',\n"
        "        description='Adds two numbers',\n"
        "        parameters={'a': {'type': 'number'}, 'b': {'type': 'number'}},\n"
        "        execute=lambda args: str(args.get('a', 0) + args.get('b', 0)),\n"
        "    )\n"
        "    ctx.tools.register(tool)\n"
        "apply.inject = ['tools']\n"
    )
    (custom_plugin_dir / "plugin.py").write_text(plugin_code, encoding="utf-8")

    manifest_data = {
        "name": "custom-math",
        "version": "1.0.0",
        "description": "Math operations plugin",
        "entry_point": "plugin.py:apply",
        "types": ["Tool"],
        "enabled": True,
    }
    (custom_plugin_dir / "manifest.json").write_text(
        json.dumps(manifest_data), encoding="utf-8"
    )

    # 4. Discover and mount the third-party plugin via PluginService
    manifests = ctx.plugins.discover(tmp_path)
    assert len(manifests) == 1
    assert manifests[0].name == "custom-math"

    fiber = ctx.plugins.mount_plugin(manifests[0])
    await ctx.flush()

    assert fiber.state == FiberState.ACTIVE
    assert ctx.tools.get("add") is not None

    # 5. Execute custom tool through ToolService pipeline
    add_call = ToolCallData(
        call_id="call-math-1",
        name="add",
        arguments={"a": 10, "b": 32},
    )
    add_res = await ctx.tools.execute(add_call)
    assert not add_res.is_error
    assert add_res.content == "42"

    # 6. Execute filesystem write and read with truncation middleware active
    test_file = tmp_path / "data.txt"
    multi_line_text = "line 1\nline 2\nline 3\nline 4\nline 5\n"

    write_call = ToolCallData(
        call_id="call-write-1",
        name="write_file",
        arguments={"path": str(test_file), "content": multi_line_text},
    )
    write_res = await ctx.tools.execute(write_call)
    assert not write_res.is_error

    read_call = ToolCallData(
        call_id="call-read-1",
        name="read_file",
        arguments={"path": str(test_file)},
    )
    read_res = await ctx.tools.execute(read_call)
    assert not read_res.is_error
    # Truncation guard capped at max_lines=3
    assert "[Output truncated: showing 3 of 5 lines" in read_res.content

    # 7. Unmount custom plugin and verify tool is removed
    await ctx.plugins.unmount_plugin("custom-math")
    assert ctx.plugins.get_plugin("custom-math") is None
    assert ctx.tools.get("add") is None
