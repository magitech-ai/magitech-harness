"""Unit tests for NativeToolExecutor and swapped-row loader injection probes."""

from __future__ import annotations

from pathlib import Path

import pytest
from animas import Context, FiberState
from animas.jsexpr import dump_yaml
from animas.loader import load_profile

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.contracts import ToolDefinition
from magitech_harness.tools.native_executor import (
    NativeToolExecutor,
)
from magitech_harness.tools.native_executor import (
    apply as apply_native_executor,
)
from magitech_harness.tools.schema import define_tool
from magitech_harness.tools.service import ToolsService


def _tool(name: str, *, defer: bool = False) -> ToolDefinition:
    spec: dict[str, object] = {
        "name": name,
        "description": f"tool {name}",
        "parameters": {},
        "output": {
            "schema": {"type": "string"},
            "render": lambda _a, val: [{"type": "text", "text": val}],
        },
        "execute": lambda _a, _c: f"ran:{name}",
    }
    if defer:
        spec["deferLoading"] = True
    return define_tool(spec)


@pytest.mark.asyncio
async def test_native_executor_describe_and_execute() -> None:
    """NativeToolExecutor preserves deferLoading in describe(), validates limits, and executes batches."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolsService)
    ctx.use(apply_native_executor)
    await ctx.flush()

    t = _tool("deferred", defer=True)
    ctx.tools.register(t)
    executor = ctx.tools.select_executor()
    assert isinstance(executor, NativeToolExecutor)
    described = executor.describe([t])
    assert described == {
        "schemas": [
            {
                "name": "deferred",
                "description": "tool deferred",
                "parameters": {"type": "object", "properties": {}},
                "deferLoading": True,
            }
        ]
    }
    assert described["schemas"][0]["parameters"] is t["parameters"]

    # Construction-time validation on maxParallelToolCalls / maxParallel
    for bad in (0, -1, True, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            NativeToolExecutor(ctx, {"maxParallelToolCalls": bad})
    bounded = NativeToolExecutor(ctx, {"maxParallel": 2})
    assert callable(bounded._max_parallel) and bounded._max_parallel() == 2
    default_exec = NativeToolExecutor(ctx)
    assert default_exec._max_parallel() == 10

    scalar_bounded = NativeToolExecutor(ctx, 4)
    assert scalar_bounded._max_parallel() == 4
    dyn = NativeToolExecutor(ctx, max_parallel=lambda: 3)
    assert dyn._max_parallel() == 3

    batch = await ctx.tools.execute(
        {
            "signal": AbortSignal(),
            "calls": [
                {"type": "tool-call", "id": "n1", "name": "deferred", "arguments": "{}"}
            ],
        }
    )
    assert batch["results"][0]["result"] == {
        "isError": False,
        "value": "ran:deferred",
        "content": [{"type": "text", "text": "ran:deferred"}],
    }


@pytest.mark.asyncio
async def test_swapped_row_load_profile_probes(tmp_path: Path) -> None:
    """Declared inject keeps rows pending until dependencies arrive and never bypasses order."""
    # 1. Reversed row order: tool-executor-native -> tools -> system-prompt
    reversed_bundle = tmp_path / "reversed.yaml"
    with reversed_bundle.open("w", encoding="utf-8") as handle:
        dump_yaml(
            [
                {
                    "id": "tool-executor-native",
                    "name": "magitech_harness.tools.native_executor:apply",
                    "config": {},
                },
                {
                    "id": "tools",
                    "name": "magitech_harness.tools.service:ToolService",
                    "config": {},
                },
                {
                    "id": "system-prompt",
                    "name": "magitech_harness.system_prompt.service:SystemPromptService",
                    "config": {},
                },
            ],
            handle,
        )

    ctx = Context()
    loader = load_profile(ctx, reversed_bundle)
    await ctx.flush()
    assert [e.fiber.state for e in loader.entries() if e.fiber is not None] == [
        FiberState.ACTIVE,
        FiberState.ACTIVE,
        FiberState.ACTIVE,
    ]
    assert isinstance(ctx.tools, ToolsService)
    ctx.tools.register(_tool("ok"))
    res = await ctx.tools.execute(
        {
            "signal": AbortSignal(),
            "calls": [
                {"type": "tool-call", "id": "x1", "name": "ok", "arguments": "{}"}
            ],
        }
    )
    assert res["results"][0]["result"]["value"] == "ran:ok"

    # 2. Missing system-prompt: tools stays PENDING (never constructs without its injection)
    no_sp_bundle = tmp_path / "no_sp.yaml"
    with no_sp_bundle.open("w", encoding="utf-8") as handle:
        dump_yaml(
            [
                {
                    "id": "tools",
                    "name": "magitech_harness.tools.service:ToolService",
                    "config": {},
                },
                {
                    "id": "tool-executor-native",
                    "name": "magitech_harness.tools.native_executor:apply",
                    "config": {},
                },
            ],
            handle,
        )
    ctx2 = Context()
    loader2 = load_profile(ctx2, no_sp_bundle)
    await ctx2.flush()
    assert not ctx2.has("tools")
    assert all(
        e.fiber is not None and e.fiber.state is FiberState.PENDING
        for e in loader2.entries()
    )
