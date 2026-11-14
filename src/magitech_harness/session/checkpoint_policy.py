"""Durability checkpoint policy: flush the session log before dispatch.

The policy joins the three dispatch boundaries as waterfall listeners on
the context event bus. Before a model stream is produced, before a
top-level tool body runs, and before each step begins, the live session
log is flushed through the store — the single durability entry point —
so the logged prefix is durable before any side effect can happen. A
flush rejection blocks dispatch: the downstream adapter, tool body, or
step is never invoked.

Requests that name no session, or name a session that is not live, pass
through uncheckpointed. Nested tool dispatches reuse the durable outer
call and are not flushed again. The turn boundary performs no flush.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from anima import Context, Disposer, EventMode

from magitech_harness.session.events import ToolCallData, ToolResultData
from magitech_harness.session.store import Session, SessionStore

if TYPE_CHECKING:
    from magitech_harness.llm.types import ModelChunk


def aborted_before_dispatch_result(call: ToolCallData) -> ToolResultData:
    """The result for a call cancelled before its tool body was invoked."""
    return ToolResultData(
        call_id=call.call_id,
        name=call.name,
        content="Error: tool call aborted before dispatch",
        is_error=True,
        error={
            "message": "tool call aborted before dispatch",
            "info": {"name": "AbortError", "code": "ABORTED_BEFORE_DISPATCH"},
        },
    )


async def _checkpointed_stream(
    store: SessionStore,
    session: Session,
    next_fn: Callable[..., Awaitable[AsyncIterator[ModelChunk]]],
) -> AsyncIterator[ModelChunk]:
    """Flush, then produce the downstream stream.

    The flush runs when iteration starts, before ``next_fn`` is invoked,
    so the downstream adapter cannot run and no chunk can be yielded
    until the logged prefix is durable.
    """
    await store.flush(session)
    downstream = await next_fn()
    async for chunk in downstream:
        yield chunk


def apply(ctx: Context, config: dict[str, Any] | None = None) -> Disposer:
    """Install the flush-before-dispatch policy as a pure wiring row.

    Joins the three dispatch boundaries as waterfall listeners on the
    context event bus, flushing through the sessions service's store —
    the single durability entry point. The policy owns no service and
    no state; it never dispatches the raw ``session/flush`` event
    itself, it calls ``store.flush`` only. Returns a disposer removing
    all three listeners; the loader runs it when the row unloads.
    """
    sessions = ctx.get("sessions", None, strict=False)
    if sessions is None:
        raise RuntimeError("session-checkpoint-policy requires the sessions service")
    store: SessionStore = sessions.store

    async def _on_llm_stream(
        options: dict[str, Any],
        next_fn: Callable[..., Awaitable[AsyncIterator[ModelChunk]]],
    ) -> AsyncIterator[ModelChunk]:
        """Checkpoint a session-bound model stream before its first chunk."""
        session_id = options.get("session_id")
        if session_id is None:
            return await next_fn()
        session = store.get(session_id)
        if session is None:
            return await next_fn()
        return _checkpointed_stream(store, session, next_fn)

    async def _on_tools_execute(
        execution: dict[str, Any], next_fn: Callable[..., Awaitable[Any]]
    ) -> Any:
        """Checkpoint a top-level tool call, then re-check for cancellation."""
        session = execution.get("session")
        if session is None or execution.get("parent") is not None:
            return await next_fn()
        await store.flush(session)
        signal = execution.get("signal")
        if signal is not None and signal.is_aborted:
            return aborted_before_dispatch_result(execution["call"])
        return await next_fn()

    async def _on_pre_step(
        decision: dict[str, Any],
        next_fn: Callable[..., Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Flush the step's session before the step begins."""
        session = decision.get("session")
        if session is not None:
            await store.flush(session)
        return await next_fn()

    disposers: list[Disposer] = [
        ctx.on("llm/stream", _on_llm_stream, mode=EventMode.WATERFALL),
        ctx.on("tools/execute", _on_tools_execute, mode=EventMode.WATERFALL),
        ctx.on("agent/pre-step", _on_pre_step, mode=EventMode.WATERFALL),
    ]

    def dispose_all() -> None:
        """Remove all three checkpoint listeners."""
        for dispose in disposers:
            dispose()

    return dispose_all


apply.inject = ["llm", "sessions", "tools"]  # type: ignore[attr-defined]
