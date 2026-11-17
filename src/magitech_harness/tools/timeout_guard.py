"""Tool-call timeout guard: cooperative, scoped-deadline TOOL_TIMEOUT policy.

A Tier 1 profile-bundle plugin. Tools declare ``timeoutMs`` on their
definition. For a limited tool, the guard swaps a deadline signal onto the
dispatch, restores the caller's signal in ``finally``, and — only if its own
scoped deadline fired — replaces the settled result with a structured
``TOOL_TIMEOUT`` error. Never hard-cancels a tool that ignores cancellation.
"""

from __future__ import annotations

from typing import Any

from anima import Context, EventMode

from magitech_harness.agent.cancellation import deadline, timeout_of

TOOL_TIMEOUT = "TOOL_TIMEOUT"


def apply(ctx: Context, **kwargs: Any) -> None:
    """Mount the timeout policy on tools/execute."""

    async def _on_execute(exec_view: dict[str, Any], next_fn: Any) -> Any:
        name = exec_view.get("name")
        tools = getattr(ctx, "tools", None)
        tool = tools.get(name) if tools is not None and name is not None else None
        timeout_ms = tool.get("timeoutMs") if isinstance(tool, dict) else None
        if timeout_ms is None:
            return await next_fn(exec_view)

        wrapper_signal = exec_view["signal"]
        dl = deadline(wrapper_signal, timeout_ms, TOOL_TIMEOUT)
        exec_view["signal"] = dl.signal
        try:
            result = await next_fn(exec_view)
        finally:
            exec_view["signal"] = wrapper_signal
            dl.dispose()

        if timeout_of(dl.signal, TOOL_TIMEOUT) is None:
            return result
        timeout_display = (
            int(timeout_ms) if float(timeout_ms).is_integer() else timeout_ms
        )
        message = f"tool call timed out after {timeout_display}ms"
        # ToolExecutionFailure shape: error.message + error.info.{name,code}.
        result["isError"] = True
        result["error"] = {
            "message": message,
            "info": {"name": "ToolTimeoutError", "code": TOOL_TIMEOUT},
        }
        result["content"] = [{"type": "text", "text": f"Error: {message}"}]
        return result

    ctx.on("tools/execute", _on_execute, mode=EventMode.WATERFALL)


apply.inject = ["tools"]  # type: ignore[attr-defined]

__all__ = ["TOOL_TIMEOUT", "apply"]
