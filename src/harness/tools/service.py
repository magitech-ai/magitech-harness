"""
ToolService (ctx.tools): Scoped tool registry with waterfall execution pipeline.
"""

from __future__ import annotations

import logging
from typing import Any
from kernel import Context, Disposer, EventMode, Service
from harness.session.events import ToolCallData, ToolResultData
from harness.tools.definition import Tool

logger = logging.getLogger("harness.tools")


class ToolService(Service):
    """Service providing tool registration and pipeline execution."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "tools")
        self._registry: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Disposer:
        """Register a tool. Attaches disposer as an effect to active plugin."""
        self._registry[tool.name] = tool

        def _unregister():
            if tool.name in self._registry:
                del self._registry[tool.name]

        self.ctx.effect(lambda: _unregister)
        return _unregister

    def get(self, name: str) -> Tool | None:
        return self._registry.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return all registered tool schemas formatted for the model."""
        return [tool.to_schema() for tool in self._registry.values()]

    async def execute(self, tool_call: ToolCallData) -> ToolResultData:
        """Run tool call through pre-execute, execute, and post-execute pipeline."""
        tool = self.get(tool_call.name)
        if not tool:
            err_res = ToolResultData(
                call_id=tool_call.call_id,
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' not found.",
                is_error=True,
            )
            self.ctx.emit("tools/result", tool_call, err_res)
            return err_res

        # 1. tools/pre-execute waterfall (can rewrite arguments or abort)
        async def _exec_tool(call_data: ToolCallData) -> Any:
            return await tool.execute(call_data.arguments)

        try:
            # Pre-execute around-middleware
            modified_call = await self.ctx.waterfall(
                "tools/pre-execute",
                tool_call,
                default_handler=lambda tc: tc,
            )

            # 2. Execution
            raw_result = await _exec_tool(modified_call)

            # 3. tools/post-execute waterfall (can format or sanitize result)
            formatted_result = await self.ctx.waterfall(
                "tools/post-execute",
                raw_result,
                modified_call,
                default_handler=lambda res, tc: res,
            )

            result = ToolResultData(
                call_id=tool_call.call_id,
                name=tool_call.name,
                content=formatted_result,
                is_error=False,
            )
        except Exception as e:
            logger.exception("Error executing tool %s: %s", tool_call.name, e)
            result = ToolResultData(
                call_id=tool_call.call_id,
                name=tool_call.name,
                content=f"Execution error: {str(e)}",
                is_error=True,
            )

        # 4. tools/result event
        self.ctx.emit("tools/result", tool_call, result)
        return result
