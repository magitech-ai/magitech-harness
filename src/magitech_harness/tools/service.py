"""
ToolService (ctx.tools): Scoped tool registry with waterfall execution pipeline.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from anima import Context, Disposer, Service

from magitech_harness.session.events import ToolCallData, ToolResultData
from magitech_harness.tools.definition import Tool

if TYPE_CHECKING:
    from magitech_harness.agent.cancellation import AbortSignal
    from magitech_harness.session.store import Session

logger = logging.getLogger("harness.tools")

# Depth of ``ToolService.execute`` calls in flight on the current task.
# A call entered while another is in flight is a nested dispatch and
# reuses the outer call's checkpoint instead of flushing again.
_EXECUTION_DEPTH: ContextVar[int] = ContextVar(
    "magitech_tools_execute_depth", default=0
)

# Marker naming a nested dispatch on the ``tools/execute`` chain.
_NESTED_CALL = object()


class ToolService(Service):
    """Service providing tool registration and pipeline execution."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "tools")
        self._registry: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Disposer:
        """Register a tool. Attaches disposer as an effect to active component."""
        self._registry[tool.name] = tool

        def _unregister() -> None:
            if tool.name in self._registry:
                del self._registry[tool.name]

        self.ctx.effect(lambda: _unregister)
        return _unregister

    def get(self, name: str) -> Tool | None:
        return self._registry.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return all registered tool schemas formatted for the model."""
        return [tool.to_schema() for tool in self._registry.values()]

    async def execute(
        self,
        tool_call: ToolCallData,
        *,
        session: Session | None = None,
        signal: AbortSignal | None = None,
        parent: Any = None,
    ) -> ToolResultData:
        """Run tool call through pre-execute, execute, and post-execute pipeline.

        The tool body itself runs through the ``tools/execute`` waterfall
        with one execution dict carrying the call, the live session, the
        cancellation signal, and the nesting marker. A listener may flush
        the session before the body runs, or short-circuit the chain by
        returning a finished ``ToolResultData`` (used for a call cancelled
        before dispatch); a finished result bypasses post-execute
        formatting. A call with no session, or a nested call (an explicit
        ``parent`` or a re-entrant ``execute``), passes through
        uncheckpointed.
        """
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

        depth = _EXECUTION_DEPTH.get()
        parent_marker = (
            parent if parent is not None else (_NESTED_CALL if depth > 0 else None)
        )
        token = _EXECUTION_DEPTH.set(depth + 1)
        try:
            # 1. tools/pre-execute waterfall (can rewrite arguments or abort)
            async def _exec_tool(call_data: ToolCallData) -> Any:
                return await tool.execute(call_data.arguments)

            async def _dispatch(execution: dict[str, Any]) -> Any:
                return await _exec_tool(execution["call"])

            try:
                # Pre-execute around-middleware
                modified_call = await self.ctx.waterfall(
                    "tools/pre-execute",
                    tool_call,
                    default_handler=lambda tc: tc,
                )

                # 2. Execution through the tools/execute chain
                execution = {
                    "call": modified_call,
                    "session": session,
                    "signal": signal,
                    "parent": parent_marker,
                }
                raw_result = await self.ctx.waterfall(
                    "tools/execute", execution, default_handler=_dispatch
                )

                if isinstance(raw_result, ToolResultData):
                    # A listener short-circuited the chain with a finished
                    # result; it bypasses post-execute formatting.
                    result = raw_result
                else:
                    # 3. tools/post-execute waterfall (can format or sanitize result)
                    formatted_result = await self.ctx.waterfall(
                        "tools/post-execute",
                        raw_result,
                        modified_call,
                        default_handler=lambda res, *args, **kwargs: res,
                    )

                    result = ToolResultData(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        content=formatted_result,
                        is_error=False,
                    )
            except Exception as e:
                logger.exception("Error executing tool %s", tool_call.name)
                result = ToolResultData(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    content=f"Execution error: {e!s}",
                    is_error=True,
                )

            # 4. tools/result event
            self.ctx.emit("tools/result", tool_call, result)
            return result
        finally:
            _EXECUTION_DEPTH.reset(token)
