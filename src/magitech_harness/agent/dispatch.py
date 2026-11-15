"""Agent-scoped event dispatcher and prompt-assembly context helpers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from anima import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.scope import Scoped, scope_target


async def _await_contained(awaitable: Awaitable[Any], ctx: Context, name: str) -> None:
    try:
        await awaitable
    except Exception as err:  # noqa: BLE001 - non-vetoing event observer containment
        ctx.logger.warning(f'agent event "{name}" listener rejected: {err}')


def agent_carrier(agent: Any) -> Scoped[Any]:
    """Build the fused scope carrier coupling an Agent subject to its scope key."""
    return scope_target(agent, agent)


class AgentEventDispatch:
    """Dispatcher binding an Agent subject and its scope carrier."""

    def __init__(self, ctx: Context, agent: Any, carrier: Scoped[Any]) -> None:
        self._ctx = ctx
        self._agent = agent
        self._carrier = carrier
        self._pending_tasks: set[asyncio.Task[None]] = set()

    def _fuse(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(payload) if payload is not None else {}
        merged["agent"] = self._agent
        return merged

    def emit(self, name: str, payload: Mapping[str, Any] | None = None) -> None:
        """Fire-and-forget notification with per-listener sync/async containment."""
        fused = self._fuse(payload)
        listeners = self._ctx._event_bus._filter_listeners(name, this_arg=self._carrier)
        for listener in listeners:
            try:
                returned = listener.callback(fused)
            except Exception as err:  # noqa: BLE001 - observer errors stay contained
                self._ctx.logger.warning(f'agent event "{name}" listener threw: {err}')
                continue
            if inspect.isawaitable(returned):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(_await_contained(returned, self._ctx, name))
                else:
                    tasks = self._pending_tasks
                    task = loop.create_task(_await_contained(returned, self._ctx, name))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

    async def serial(self, name: str, payload: Mapping[str, Any] | None = None) -> Any:
        """Awaited in-order dispatch (serial) in the agent's scope."""
        return await self._ctx.events.serial(
            name, self._fuse(payload), this_arg=self._carrier
        )

    async def waterfall(
        self,
        name: str,
        payload: Mapping[str, Any] | None = None,
        *,
        default_handler: Callable[..., Any] | None = None,
    ) -> Any:
        """Around-middleware dispatch (waterfall) in the agent's scope."""
        return await self._ctx.events.waterfall(
            name,
            self._fuse(payload),
            this_arg=self._carrier,
            default_handler=default_handler,
        )


def agent_events(
    ctx: Context, agent: Any, carrier: Scoped[Any] | None = None
) -> AgentEventDispatch:
    """Build a fused dispatcher coupling an Agent subject to its scope carrier."""
    return AgentEventDispatch(
        ctx, agent, agent_carrier(agent) if carrier is None else carrier
    )


def emit_agent_event(
    ctx: Context,
    agent: Any,
    name: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    """Emit a single contained agent notification without retaining a dispatcher."""
    agent_events(ctx, agent).emit(name, payload)


def assemble_context_for(
    agent: Any, signal: AbortSignal | None = None
) -> dict[str, Any]:
    """Build a prompt-assembly context with agent and scope coupled."""
    context: dict[str, Any] = {"agent": agent, "scope": agent}
    if signal is not None:
        context["signal"] = signal
    return context
