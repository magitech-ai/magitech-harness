"""Native tool executor component driving ToolExecutionPipeline batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from anima import Context, Disposer

from magitech_harness.tools.contracts import (
    ToolBatchInput,
    ToolBatchResult,
    ToolDefinition,
    ToolModelInput,
)
from magitech_harness.tools.pipeline import ToolExecutionPipeline, execute_tool_batch
from magitech_harness.tools.schema import project_tool_schema


def _validate_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


class NativeToolExecutor:
    """Default batch executor translating tool calls directly to ToolExecutionPipeline."""

    def __init__(
        self,
        ctx: Context,
        config: Mapping[str, Any] | Callable[[], int] | int | None = None,
        *,
        max_parallel: Callable[[], int] | int | None = None,
    ) -> None:
        self.ctx = ctx
        self._pipeline = ToolExecutionPipeline(ctx)

        raw_limit: Any = max_parallel
        if raw_limit is None:
            if isinstance(config, Mapping):
                if "maxParallelToolCalls" in config:
                    raw_limit = config["maxParallelToolCalls"]
                elif "maxParallel" in config:
                    raw_limit = config["maxParallel"]
            elif config is not None:
                raw_limit = config

        if raw_limit is None:
            self._max_parallel: Callable[[], int] = lambda: 10
        elif callable(raw_limit):
            self._max_parallel = raw_limit
        else:
            fixed = _validate_positive_int(raw_limit, "maxParallelToolCalls")

            def _fixed_limit() -> int:
                return fixed

            self._max_parallel = _fixed_limit

    def describe(self, definitions: Sequence[ToolDefinition]) -> ToolModelInput:
        """Project tool definitions without copying parameters or stripping deferLoading."""
        return {"schemas": [project_tool_schema(defn, False) for defn in definitions]}

    async def execute(self, batch: ToolBatchInput) -> ToolBatchResult:
        """Execute a tool batch through the shared pipeline with optional concurrency limit."""
        return await execute_tool_batch(
            batch,
            self._pipeline,
            max_parallel=self._max_parallel,
        )


def apply(ctx: Context, config: dict[str, Any] | None = None) -> Disposer:
    """Mount NativeToolExecutor onto ctx.tools."""
    return cast(Disposer, ctx.tools.set_executor(NativeToolExecutor(ctx, config)))


apply.inject = ["tools"]  # type: ignore[attr-defined]

__all__ = ["NativeToolExecutor", "apply"]
