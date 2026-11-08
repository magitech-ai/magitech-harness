"""Unit tests for Tool and define_tool."""

import pytest

from magitech_harness.tools.definition import Tool, define_tool


@pytest.mark.asyncio
async def test_tool_execute_sync_and_async() -> None:
    """Verifies execution of sync and async handlers in Tool."""
    # Sync tool
    sync_tool = define_tool(
        name="sync_fn",
        description="Sync test",
        parameters={"x": {"type": "integer"}},
        execute=lambda args: args.get("x", 0) * 2,
    )
    assert await sync_tool.execute({"x": 5}) == 10

    # Async tool
    async def async_handler(args):
        return f"async_{args.get('val')}"

    async_tool = define_tool(
        name="async_fn",
        description="Async test",
        execute=async_handler,
    )
    assert await async_tool.execute({"val": "ok"}) == "async_ok"


@pytest.mark.asyncio
async def test_tool_no_handler_raises() -> None:
    """Verifies that executing a Tool with no handler raises NotImplementedError."""
    tool = Tool(name="empty", description="no handler")
    with pytest.raises(NotImplementedError, match="has no handler"):
        await tool.execute({})


def test_tool_to_schema_variations() -> None:
    """Verifies schema generation with dict parameters and primitive spec."""
    tool = define_tool(
        name="my_tool",
        description="Complex tool",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            "limit": {"type": "integer", "description": "Max items"},
            "simple_flag": "A boolean flag description",
        },
        execute=lambda args: None,
    )

    schema = tool.to_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "my_tool"
    assert fn["description"] == "Complex tool"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "query" in params["required"]
    assert "limit" not in params["required"]
    assert (
        params["properties"]["simple_flag"]["description"]
        == "A boolean flag description"
    )
