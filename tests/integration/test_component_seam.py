"""Integration test for portable component seam in Magitech Harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from anima import FiberState

from magitech_harness import create_harness
from magitech_harness.session.events import ToolCallData


@pytest.mark.asyncio
async def test_component_seam_full_lifecycle(tmp_path: Path) -> None:
    """Verifies end-to-end discovery, mounting, tool execution, and waterfall middleware."""
    # 1. Create harness with offline mock provider
    ctx = create_harness(mock_provider=True)
    await ctx.flush()

    assert ctx.has("components")
    assert ctx.has("tools")
    assert ctx.has("llm")

    # 2. Create tool component directory on disk with manifest.json
    tool_component_dir = tmp_path / "custom_math"
    tool_component_dir.mkdir()

    tool_code = (
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
    (tool_component_dir / "component.py").write_text(tool_code, encoding="utf-8")

    tool_manifest_data = {
        "name": "custom-math",
        "version": "1.0.0",
        "description": "Math operations component",
        "entry_point": "component.py:apply",
        "types": ["Tool"],
        "enabled": True,
    }
    (tool_component_dir / "manifest.json").write_text(
        json.dumps(tool_manifest_data), encoding="utf-8"
    )

    # 3. Create guardrail / waterfall component on disk
    guard_component_dir = tmp_path / "custom_guard"
    guard_component_dir.mkdir()

    guard_code = (
        "def apply(ctx):\n"
        "    async def _post_exec(output, call_data, next_fn):\n"
        "        res = await next_fn(output, call_data)\n"
        "        return f'[verified: {res}]'\n"
        "    ctx.on('tools/post-execute', _post_exec)\n"
        "apply.inject = ['tools']\n"
    )
    (guard_component_dir / "component.py").write_text(guard_code, encoding="utf-8")

    guard_manifest_data = {
        "name": "custom-guard",
        "version": "1.0.0",
        "description": "Output verification middleware",
        "entry_point": "component.py:apply",
        "types": ["Guard"],
        "enabled": True,
    }
    (guard_component_dir / "manifest.json").write_text(
        json.dumps(guard_manifest_data), encoding="utf-8"
    )

    # 4. Discover components via ComponentService
    manifests = ctx.components.discover(tmp_path)
    assert len(manifests) == 2
    manifest_map = {m.name: m for m in manifests}
    assert "custom-math" in manifest_map
    assert "custom-guard" in manifest_map

    # 5. Mount components into context
    fiber_math = ctx.components.mount_component(manifest_map["custom-math"])
    fiber_guard = ctx.components.mount_component(manifest_map["custom-guard"])
    await ctx.flush()

    assert fiber_math.state == FiberState.ACTIVE
    assert fiber_guard.state == FiberState.ACTIVE
    assert ctx.tools.get("add") is not None

    # 6. Execute custom tool through ToolService pipeline
    add_call = ToolCallData(
        call_id="call-math-1",
        name="add",
        arguments={"a": 10, "b": 32},
    )
    add_res = await ctx.tools.execute(add_call)
    assert not add_res.is_error
    # Post-execute waterfall from custom-guard modified output
    assert add_res.content == "[verified: 42]"

    # 7. Unmount custom components and verify clean removal
    await ctx.components.unmount_component("custom-math")
    assert ctx.components.get_component("custom-math") is None
    assert ctx.tools.get("add") is None

    await ctx.components.unmount_component("custom-guard")
    assert ctx.components.get_component("custom-guard") is None
