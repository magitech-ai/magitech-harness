"""AgentRegistry service: live agent table, factory delegation, and initiator scope."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from animas import Context, Fiber, FiberState, Service
from animas.core import _access_context, _BoundService, symbols

from magitech_harness.agent.archive_admission import install_turn_archive_admission
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.agent.contracts import (
    AgentFactory,
    AgentHandle,
    CreateAgentOptions,
    ResumeAgentOptions,
    SessionStartSource,
    TurnBoundaryProjection,
)
from magitech_harness.agent.projection import turn_boundary_projection_definition
from magitech_harness.scope import Scoped, scope_target
from magitech_harness.session.projection_registry import ProjectionHandle
from magitech_harness.session.store import Session

_NO_FACTORY_MESSAGE = "no agent factory registered (load an agent-loop plugin)"
_NO_INITIATOR_MESSAGE = "no initiating agent is active"
_DISPOSED_INITIATOR_MESSAGE = "agent initiator scope is disposed"


@dataclass
class _AgentEntry:
    id: str
    agent: Any
    carrier: Scoped[Any]
    announced: bool = False
    announcing: bool = False
    detach_requested: bool = False


@dataclass
class _InitiatorRun:
    active: bool
    parent: _InitiatorRun | None


async def _await_contained_disposed(
    awaitable: Any, ctx: Context, agent_id: str
) -> None:
    try:
        await awaitable
    except Exception as err:  # noqa: BLE001 - non-vetoing disposal containment
        ctx.logger.warning(
            f'agent "{agent_id}": agent/disposed listener rejected: {err}'
        )


def _canonicalize_factory(factory: AgentFactory) -> AgentFactory:
    """Unwrap Service proxy layers so factory calls bind cleanly per caller."""
    if isinstance(factory, _BoundService):
        return cast(AgentFactory, object.__getattribute__(factory, "_svc"))
    original = getattr(factory, symbols["original"], None)
    if original is not None:
        return cast(AgentFactory, original)
    return factory


class AgentRegistry(Service):
    """Live agent registry, factory seam, and process-local initiator attribution."""

    inject: ClassVar[list[str]] = ["sessionProjections"]

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(ctx, "agents")
        self._fiber: Fiber | None = ctx.fiber
        self._turn_boundary_handle: ProjectionHandle[TurnBoundaryProjection] | None = (
            None
        )
        self._store: dict[str, _AgentEntry] = {}
        self._factory: AgentFactory | None = None
        self._initiators: ContextVar[Any | None] = ContextVar(
            f"agents_initiators_{id(self)}", default=None
        )
        self._initiator_runs: ContextVar[_InitiatorRun | None] = ContextVar(
            f"agents_initiator_runs_{id(self)}", default=None
        )
        self._initiator_state: Literal["active", "closing", "disposed"] = "active"
        self._active_initiator_runs = 0
        self._initiator_drain: asyncio.Future[None] | None = None
        self._initiator_disposal: asyncio.Task[None] | None = None
        self._disposed_tasks: set[asyncio.Task[None]] = set()
        ctx.on("internal/status", self._on_internal_status)
        ctx.fiber.effect(
            lambda: [self._dispose_initiators, self._close_initiators],
            label="agents.initiatorLifecycle()",
        )
        install_turn_archive_admission(ctx, self.get)

    def _caller_ctx(self) -> Context:
        return _access_context.get() or self.ctx

    async def init(self) -> Callable[[], None]:
        """Register the host turn-boundary projection and return its disposer."""
        registry = self.ctx.get("sessionProjections", None, strict=True)
        if registry is None:
            raise RuntimeError("agents requires the sessionProjections service")
        result = registry.register(turn_boundary_projection_definition)
        handle = await result if inspect.isawaitable(result) else result
        self._turn_boundary_handle = handle
        return cast(Callable[[], None], handle.dispose)

    def turn_boundary_of(self, session: Session) -> TurnBoundaryProjection:
        """Read the turn and step positions recorded in a Session."""
        if self._turn_boundary_handle is None:
            raise RuntimeError(
                "agents requires the sessionProjections service to be initialized"
            )
        return self._turn_boundary_handle.state_of(session)

    def current_initiator(self) -> Any | None:
        """Read the Agent initiating the active driver chain, or None."""
        self._assert_initiators_readable()
        return self._initiators.get()

    def require_initiator(self) -> Any:
        """Read the initiating Agent or fail when none is active."""
        agent = self.current_initiator()
        if agent is None:
            raise RuntimeError(_NO_INITIATOR_MESSAGE)
        return agent

    def with_initiator[T](self, agent: Any, operation: Callable[[], T]) -> T:
        """Run an operation with one Agent as process-local initiator."""
        return self._run_with_initiator(agent, operation)

    def without_initiator[T](self, operation: Callable[[], T]) -> T:
        """Run an operation while hiding any inherited initiating Agent."""
        return self._run_with_initiator(None, operation)

    def set_factory(self, factory: AgentFactory) -> Callable[[], Any]:
        """Register the single active AgentFactory on the caller's fiber."""
        caller_ctx = self._caller_ctx()

        def _setup() -> Callable[[], None]:
            if self._factory is not None:
                raise RuntimeError("an agent factory is already registered")
            self._factory = _canonicalize_factory(factory)

            def _clear() -> None:
                self._factory = None

            return _clear

        return cast(
            Callable[[], Any],
            caller_ctx.fiber.effect(_setup, label="agents.setFactory()"),
        )

    def _require_factory(self) -> AgentFactory:
        if self._factory is None:
            raise RuntimeError(_NO_FACTORY_MESSAGE)
        return self._factory

    async def create(self, options: CreateAgentOptions) -> AgentHandle:
        """Create and publish a new agent attached to the calling context."""
        owner_ctx = self._caller_ctx()
        target = self._require_factory()
        receiver: Any = (
            _BoundService(target, owner_ctx) if isinstance(target, Service) else target
        )
        return cast(AgentHandle, await receiver.create_agent(owner_ctx, options))

    async def resume(self, options: ResumeAgentOptions) -> AgentHandle:
        """Resume a persisted session's agent attached to the calling context."""
        owner_ctx = self._caller_ctx()
        target = self._require_factory()
        receiver: Any = (
            _BoundService(target, owner_ctx) if isinstance(target, Service) else target
        )
        return cast(AgentHandle, await receiver.resume(owner_ctx, options))

    async def register(self, agent: Any) -> Callable[[], Any]:
        """Enter and announce a pre-constructed agent on the caller's fiber."""
        caller_ctx = self._caller_ctx()
        fiber = caller_ctx.fiber
        fiber.assertActive()
        detach = self.enter(agent)
        disposer = cast(
            Callable[[], Any],
            fiber.effect(lambda: detach, label="agents.register()"),
        )
        try:
            await self.announce(agent, "startup")
        except BaseException:
            res = disposer()
            if inspect.isawaitable(res):
                await res  # pragma: no cover - detach is synchronous
            raise
        return disposer

    def enter(self, agent: Any) -> Callable[[], None]:
        """Insert a prepared agent without announcing it; returns one-shot detach."""
        agent_id = agent.id
        if agent_id != agent.session.id:
            raise ValueError(
                f'agent id "{agent_id}" does not match session id "{agent.session.id}"'
            )
        if agent_id in self._store:
            raise ValueError(f'agent "{agent_id}" is already registered')
        entry = _AgentEntry(
            id=agent_id,
            agent=agent,
            carrier=scope_target(agent, agent),
        )
        self._store[agent_id] = entry
        entered = True

        def _detach() -> None:
            nonlocal entered
            if not entered:
                return
            entered = False
            if entry.announcing:
                entry.detach_requested = True
                return
            self._detach_entered(entry)

        return _detach

    def _detach_entered(self, entry: _AgentEntry) -> None:
        entry.detach_requested = False
        if (
            self._store.get(entry.id) is not entry
        ):  # pragma: no cover - guarded by enter
            return
        del self._store[entry.id]
        if not entry.announced:
            return
        self._emit_disposed(entry)

    def _emit_disposed(self, entry: _AgentEntry) -> None:
        payload = {"agent": entry.agent}
        listeners = self.ctx._event_bus._filter_listeners(
            "agent/disposed", this_arg=entry.carrier
        )
        for listener in listeners:
            try:
                returned = listener.callback(payload)
            except Exception as err:  # noqa: BLE001 - non-vetoing disposal containment
                self.ctx.logger.warning(
                    f'agent "{entry.id}": agent/disposed listener threw: {err}'
                )
                continue
            if inspect.isawaitable(returned):
                try:
                    loop = asyncio.get_running_loop()
                except (
                    RuntimeError
                ):  # pragma: no cover - no active loop during sync detach
                    asyncio.run(_await_contained_disposed(returned, self.ctx, entry.id))
                else:
                    task = loop.create_task(
                        _await_contained_disposed(returned, self.ctx, entry.id)
                    )
                    self._disposed_tasks.add(task)
                    task.add_done_callback(self._disposed_tasks.discard)

    async def announce(
        self,
        agent: Any,
        source: SessionStartSource,
        signal: AbortSignal | None = None,
    ) -> None:
        """Announce an entered agent through serial ``agent/created`` listeners."""
        entry = self._store.get(agent.id)
        if entry is None or entry.agent is not agent:
            raise ValueError(f'agent "{agent.id}" is not live in this registry')
        if entry.announced or entry.announcing:
            raise ValueError(f'agent "{entry.id}" was already announced')
        entry.announcing = True
        entry.announced = True
        payload: dict[str, Any] = {"agent": entry.agent, "source": source}
        if signal is not None:
            payload["signal"] = signal
        try:
            await self.ctx.events.serial(
                "agent/created", payload, this_arg=entry.carrier
            )
        finally:
            entry.announcing = False
            if entry.detach_requested:
                self._detach_entered(entry)

    def get(self, agent_id: str) -> Any | None:
        """Look up a live registered agent by shared agent/session id."""
        entry = self._store.get(agent_id)
        return entry.agent if entry is not None else None

    def list(self) -> list[Any]:
        """Return all live registered agents in registration order."""
        return [entry.agent for entry in self._store.values()]

    def _on_internal_status(
        self, fiber: Fiber, _old_state: FiberState = FiberState.PENDING
    ) -> None:
        if fiber.state == FiberState.UNLOADING and self._has_lifecycle_ancestor(fiber):
            self._close_initiators()

    def _has_lifecycle_ancestor(self, candidate: Fiber) -> bool:
        fiber = self._fiber
        while fiber is not None:
            if fiber is candidate:
                return True
            parent = fiber.parent_fiber
            if parent is fiber:  # pragma: no cover - defensive cycle guard
                return False
            fiber = parent
        return False

    def _assert_initiators_readable(self) -> None:
        if self._initiator_state == "disposed":
            raise RuntimeError(_DISPOSED_INITIATOR_MESSAGE)

    def _close_initiators(self) -> None:
        if self._initiator_state == "active":
            self._initiator_state = "closing"

    async def _dispose_initiators(self) -> None:
        if (
            self._initiator_disposal is None
        ):  # pragma: no branch - Fiber.dispose invokes effect once

            async def _drain() -> None:
                self._close_initiators()
                self._release_reentrant_initiator_runs()
                if self._active_initiator_runs != 0:
                    if self._initiator_drain is None:  # pragma: no branch
                        self._initiator_drain = (
                            asyncio.get_running_loop().create_future()
                        )
                    await self._initiator_drain
                self._initiator_state = "disposed"

            self._initiator_disposal = asyncio.ensure_future(_drain())
        await self._initiator_disposal

    def _run_with_initiator[T](
        self, agent: Any | None, operation: Callable[[], T]
    ) -> T:
        if self._initiator_state != "active":
            raise RuntimeError(_DISPOSED_INITIATOR_MESSAGE)
        run = _InitiatorRun(active=True, parent=self._initiator_runs.get())
        self._active_initiator_runs += 1
        run_token = self._initiator_runs.set(run)
        init_token = self._initiators.set(agent)
        try:
            result = operation()
        except BaseException:
            self._initiators.reset(init_token)
            self._initiator_runs.reset(run_token)
            self._release_initiator_run(run)
            raise
        self._initiators.reset(init_token)
        self._initiator_runs.reset(run_token)
        if asyncio.isfuture(result):
            result.add_done_callback(lambda _fut: self._release_initiator_run(run))
            return result
        if inspect.isawaitable(result):

            async def _tracked() -> Any:
                inner_run_token = self._initiator_runs.set(run)
                inner_init_token = self._initiators.set(agent)
                try:
                    return await result
                finally:
                    self._initiators.reset(inner_init_token)
                    self._initiator_runs.reset(inner_run_token)
                    self._release_initiator_run(run)

            return cast(T, _tracked())
        self._release_initiator_run(run)
        return result

    def _release_reentrant_initiator_runs(self) -> None:
        run = self._initiator_runs.get()
        while run is not None:
            self._release_initiator_run(run)
            run = run.parent

    def _release_initiator_run(self, run: _InitiatorRun) -> None:
        if not run.active:
            return
        run.active = False
        self._active_initiator_runs -= 1
        if self._active_initiator_runs != 0:
            return
        drain = self._initiator_drain
        self._initiator_drain = None
        if drain is not None and not drain.done():
            drain.set_result(None)
