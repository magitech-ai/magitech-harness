"""Unit tests for ToolService."""

import pytest
from anima import Context, EventMode

from magitech_harness.session.events import ToolCallData
from magitech_harness.tools.definition import define_tool
from magitech_harness.tools.service import ToolService


@pytest.mark.asyncio
async def test_tool_service_register_unregister_and_get() -> None:
    """Verifies tool registration, retrieval, schema export, and unregistration."""
    ctx = Context()
    service = ToolService(ctx)

    t1 = define_tool(name="calc", description="calculator", execute=lambda a: 1)
    disposer = service.register(t1)

    assert service.get("calc") is t1
    assert len(service.get_schemas()) == 1
    assert service.get_schemas()[0]["function"]["name"] == "calc"

    disposer()
    assert service.get("calc") is None
    assert service.get_schemas() == []

    # Second call is idempotent (branch 28->exit)
    disposer()


@pytest.mark.asyncio
async def test_tool_service_execute_tool_not_found() -> None:
    """Verifies execution of nonexistent tool returns error result and emits event."""
    ctx = Context()
    service = ToolService(ctx)
    results_emitted = []
    ctx.on(
        "tools/result",
        lambda call, res: results_emitted.append((call.name, res.is_error)),
    )

    call_data = ToolCallData(call_id="call_x", name="nonexistent", arguments={})
    res = await service.execute(call_data)

    assert res.is_error
    assert "not found" in res.content
    assert results_emitted == [("nonexistent", True)]


@pytest.mark.asyncio
async def test_tool_service_execute_exception_handling() -> None:
    """Verifies exceptions during tool execution are trapped into ToolResultData error."""
    ctx = Context()
    service = ToolService(ctx)

    def crashing_tool(args):
        raise ValueError("Division by zero in tool")

    service.register(
        define_tool(name="crash", description="Crashes", execute=crashing_tool)
    )

    call_data = ToolCallData(call_id="c_err", name="crash", arguments={})
    res = await service.execute(call_data)

    assert res.is_error
    assert "Division by zero" in res.content


@pytest.mark.asyncio
async def test_tool_service_waterfall_pre_and_post_execute() -> None:
    """Verifies pre-execute and post-execute waterfall hooks."""
    ctx = Context()
    service = ToolService(ctx)

    service.register(
        define_tool(
            name="echo",
            description="echoes input",
            execute=lambda a: a.get("msg", ""),
        )
    )

    # Pre-execute: rewrite argument
    async def pre_hook(call: ToolCallData, next_fn):
        rewritten = ToolCallData(
            call_id=call.call_id,
            name=call.name,
            arguments={"msg": f"pre_{call.arguments.get('msg')}"},
        )
        return await next_fn(rewritten)

    # Post-execute: sanitize output
    async def post_hook(raw_result: str, call: ToolCallData, next_fn):
        return await next_fn(f"post_{raw_result}")

    ctx.on("tools/pre-execute", pre_hook, mode=EventMode.WATERFALL)
    ctx.on("tools/post-execute", post_hook, mode=EventMode.WATERFALL)

    res = await service.execute(
        ToolCallData(call_id="c1", name="echo", arguments={"msg": "hello"})
    )
    assert not res.is_error
    assert res.content == "post_pre_hello"
