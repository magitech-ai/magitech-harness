"""Concrete agent-loop service: creates scoped ReactLoopAgents and owns teardown."""

from __future__ import annotations

import asyncio
import copy
import inspect
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from anima import Context, Fiber, FiberState, Service

from magitech_harness.agent.cancellation import AbortController, AbortSignal
from magitech_harness.agent.constants import DEFAULT_MAX_PARALLEL_TOOL_CALLS
from magitech_harness.agent.contracts import (
    Agent,
    AgentHandle,
    AgentOptions,
    AgentSetupCommit,
    CreateAgentOptions,
    ModelSelection,
    ResumeAgentOptions,
    SessionStartSource,
)
from magitech_harness.agent.inbox import InboxState, inbox_projection_definition
from magitech_harness.agent.model_selection import (
    ModelSelectionProjectionState,
    model_selection_projection_definition,
)
from magitech_harness.agent.react_loop import ReactLoopAgent
from magitech_harness.llm.error import error_chain
from magitech_harness.llm.values import is_safe_integer
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.persistence import (
    SessionHandle,
    SessionPersistence,
    SessionPersistenceNotFoundError,
)
from magitech_harness.session.projection_registry import ProjectionHandle
from magitech_harness.session.repair import interrupted_turn_closers
from magitech_harness.session.store import Session, SessionPreparation

CONFIGURED_AGENT_IDENTITIES_KEY = "configuredAgentIdentities"

_INACTIVE_STATES: frozenset[FiberState] = frozenset(
    {FiberState.UNLOADING, FiberState.DISPOSED, FiberState.FAILED}
)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _throw_collected_failures(failures: Sequence[BaseException], message: str) -> None:
    if len(failures) == 1:
        raise failures[0]
    if len(failures) > 1:
        raise BaseExceptionGroup(message, list(failures))


def _signal_abort_error(signal: AbortSignal, agent_id: str) -> BaseException:
    if isinstance(signal.reason, BaseException):
        return signal.reason
    err = RuntimeError(f'agent "{agent_id}" creation aborted')
    return err


def _resolve_persistence_backend(ctx: Context) -> SessionPersistence | None:
    service = ctx.get("sessionPersistence", None, strict=False)
    if service is None:
        return None
    if isinstance(service, SessionPersistence):
        return service
    backend = getattr(service, "backend", None)
    return backend if isinstance(backend, SessionPersistence) else None


class FactoryOwnership:
    """Factory-level ownership: live agent teardowns plus config startup work."""

    def __init__(self, fiber: Fiber) -> None:
        self._fiber = fiber
        self._accepting = True
        self._teardown = AbortController()
        self._inactive_event = asyncio.Event()
        self._live_agents: set[Callable[[], Awaitable[None]]] = set()
        self._startup_tasks: set[asyncio.Task[Any]] = set()

    @property
    def signal(self) -> AbortSignal:
        """Aborts when factory teardown begins."""
        return self._teardown.signal

    def is_active(self) -> bool:
        """Return True while accepting new work on an active fiber."""
        return self._accepting and self._fiber.state not in _INACTIVE_STATES

    def track(self, dispose: Callable[[], Awaitable[None]]) -> Callable[[], None]:
        """Track one live agent teardown closure until removed."""
        self._live_agents.add(dispose)
        return lambda: self._live_agents.discard(dispose)

    def track_startup(self, job: Awaitable[Any]) -> None:
        """Track background startup work and ensure factory disposal awaits it."""
        task: asyncio.Task[Any] = asyncio.ensure_future(job)
        self._startup_tasks.add(task)
        task.add_done_callback(lambda t: self._startup_tasks.discard(t))

    def track_wrapper(self, task: asyncio.Task[Any]) -> None:
        """Join an in-flight create/resume continuation without propagating its fault."""

        async def _settle() -> None:
            try:
                await task
            except BaseException:  # noqa: BLE001 - caller observes wrapper faults directly
                return

        self.track_startup(_settle())

    async def wait_while_active(self, job: Awaitable[None]) -> None:
        """Await *job*, stopping early if factory teardown begins."""
        inactive_task = asyncio.create_task(self._inactive_event.wait())
        job_task = asyncio.ensure_future(job)
        try:
            done, pending = await asyncio.wait(
                {job_task, inactive_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pending:
                p.cancel()
            if job_task in done and not job_task.cancelled():
                job_task.result()
        finally:
            if not inactive_task.done():
                inactive_task.cancel()

    async def dispose(self) -> None:
        """Abort factory work and await settlement of live agents and startup tasks."""
        self._accepting = False
        self._teardown.abort(RuntimeError("agent loop is not active"))
        self._inactive_event.set()
        items: list[Awaitable[Any]] = [fn() for fn in list(self._live_agents)]
        items.extend(list(self._startup_tasks))
        if not items:
            return
        results = await asyncio.gather(*items, return_exceptions=True)
        failures = [r for r in results if isinstance(r, BaseException)]
        _throw_collected_failures(failures, "agent loop disposal failed")


async def race_abort[T](
    operation: Awaitable[T] | T, signal: AbortSignal, agent_id: str
) -> T:
    """Await *operation*, raising immediately if *signal* aborts."""
    if signal.aborted:
        raise _signal_abort_error(signal, agent_id)
    if not inspect.isawaitable(operation):
        return operation
    loop = asyncio.get_running_loop()
    aborted_future: asyncio.Future[None] = loop.create_future()

    def _on_abort() -> None:
        aborted_future.set_exception(_signal_abort_error(signal, agent_id))

    unsub = signal.on_abort(_on_abort)
    op_task = asyncio.ensure_future(operation)
    try:
        done, _pending = await asyncio.wait(
            {op_task, aborted_future}, return_when=asyncio.FIRST_COMPLETED
        )
        if aborted_future in done:
            op_task.cancel()
            await asyncio.gather(op_task, return_exceptions=True)
            aborted_future.result()
        return cast(T, op_task.result())
    finally:
        unsub()


async def race_abort_call[T](
    operation: Callable[[], Awaitable[T] | T],
    signal: AbortSignal,
    agent_id: str,
    release_abandoned: Callable[[T], Any] | None = None,
) -> T:
    """Start an abortable operation and release a value that arrives post-abort."""
    if signal.aborted:
        raise _signal_abort_error(signal, agent_id)
    raw = operation()
    if not inspect.isawaitable(raw):
        return raw
    loop = asyncio.get_running_loop()
    aborted_future: asyncio.Future[None] = loop.create_future()

    def _on_abort() -> None:
        aborted_future.set_exception(_signal_abort_error(signal, agent_id))

    unsub = signal.on_abort(_on_abort)
    op_task = asyncio.ensure_future(raw)
    try:
        done, _pending = await asyncio.wait(
            {op_task, aborted_future}, return_when=asyncio.FIRST_COMPLETED
        )
        if aborted_future in done:
            if release_abandoned is not None:

                def _release_later(t: asyncio.Task[T]) -> None:
                    if not t.cancelled() and t.exception() is None:
                        res = release_abandoned(t.result())
                        if inspect.isawaitable(res):
                            asyncio.ensure_future(res)

                op_task.add_done_callback(_release_later)
            aborted_future.result()
        return cast(T, op_task.result())
    finally:
        unsub()


def _assert_agent_options(options: Mapping[str, Any]) -> None:
    max_tokens = options.get("maxTokens")
    if max_tokens is not None and (not is_safe_integer(max_tokens) or max_tokens <= 0):
        raise TypeError("agent maxTokens must be a positive safe integer")


def apply_launcher_identities(
    agents: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Apply launcher-owned session identities over configured agent entries."""
    result: list[dict[str, Any]] = []
    for entry in agents:
        copied = dict(entry)
        if identities is not None and copied.get("id") in identities:
            identity = identities[copied["id"]]
            copied.pop("sessionId", None)
            copied.pop("resumeSessionId", None)
            if identity.get("resume"):
                copied["resumeSessionId"] = identity["id"]
            else:
                copied["sessionId"] = identity["id"]
        result.append(copied)
    return result


def validate_configured_agents(agents: Sequence[Mapping[str, Any]]) -> None:
    """Reject mutually exclusive or duplicate session identities at boot."""
    exact_identities: dict[str, str] = {}
    for entry in agents:
        cfg_id = str(entry["id"])
        session_id = entry.get("sessionId")
        resume_id = entry.get("resumeSessionId")
        has_resume = isinstance(resume_id, str) and resume_id != ""
        if session_id is not None and has_resume:
            raise ValueError(
                f'agent "{cfg_id}": sessionId and resumeSessionId are mutually '
                "exclusive"
            )
        exact = resume_id if has_resume else session_id
        if exact is None:
            continue
        first_id = exact_identities.get(str(exact))
        if first_id is not None:
            raise ValueError(
                f'agents "{first_id}" and "{cfg_id}" use duplicate exact session '
                f'identity "{exact}"'
            )
        exact_identities[str(exact)] = cfg_id


@dataclass
class _StoredSession:
    handle: SessionHandle
    stored_count: int = 0


def _parse_max_parallel(config: Mapping[str, Any]) -> int:
    raw = config.get(
        "maxParallelToolCalls",
        config.get("max_parallel_tool_calls", DEFAULT_MAX_PARALLEL_TOOL_CALLS),
    )
    if not is_safe_integer(raw) or raw < 1:
        raise TypeError("agentLoop maxParallelToolCalls must be a positive integer")
    return int(raw)


class AgentLoop(Service):
    """Concrete agent factory and turn-driver service."""

    inject: ClassVar[list[str]] = [
        "agents",
        "sessions",
        "llm",
        "tools",
        "system_prompt",
        "sessionProjections",
    ]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "agentLoop")
        cfg = dict(config) if config is not None else {}
        raw_agents = cfg.get("agents") or []
        identities = ctx.get(CONFIGURED_AGENT_IDENTITIES_KEY, None, strict=False)
        self.max_parallel_tool_calls = _parse_max_parallel(cfg)
        self.configured_agents = apply_launcher_identities(raw_agents, identities)
        validate_configured_agents(self.configured_agents)
        self._ownership = FactoryOwnership(ctx.fiber)
        self._runtime_ctx = ctx
        self._model_projection_handle: (
            ProjectionHandle[ModelSelectionProjectionState] | None
        ) = None
        self._inbox_projection_handle: ProjectionHandle[InboxState] | None = None

    async def init(self) -> list[Callable[[], Any]]:
        """Register agent projections, factory binding, prompt variables, and agents."""
        ctx = self._runtime_ctx
        disposers: list[Callable[[], Any]] = [self._ownership.dispose]

        model_handle = await _maybe_await(
            ctx.sessionProjections.register(model_selection_projection_definition)
        )
        inbox_handle = await _maybe_await(
            ctx.sessionProjections.register(inbox_projection_definition)
        )
        self._model_projection_handle = model_handle
        self._inbox_projection_handle = inbox_handle
        disposers.append(model_handle.dispose)
        disposers.append(inbox_handle.dispose)

        factory_disposer = ctx.agents.set_factory(self)
        disposers.append(factory_disposer)

        var_prov = ctx.system_prompt.variable(
            "provider",
            lambda c: (
                (c.get("modelSelection") or {}).get("provider")
                if c.get("modelSelection") is not None
                else (
                    c["agent"].options.get("provider")
                    if c.get("agent") is not None
                    else None
                )
            ),
        )
        var_model = ctx.system_prompt.variable(
            "model",
            lambda c: (
                (c.get("modelSelection") or {}).get("model")
                if c.get("modelSelection") is not None
                else (
                    c["agent"].options.get("model")
                    if c.get("agent") is not None
                    else None
                )
            ),
        )
        disposers.extend([var_prov, var_model])

        for entry in self.configured_agents:
            cfg = dict(entry)
            cfg_id = str(cfg.pop("id"))
            session_id = cfg.pop("sessionId", None)
            cwd = cfg.pop("cwd", None)
            resume_id = cfg.pop("resumeSessionId", None)
            options = cast(AgentOptions, cfg)
            meta = {"cwd": cwd} if cwd is not None else {}

            if not resume_id:
                configured_id = session_id or f"{cfg_id}-session-{uuid.uuid4()}"
                persistence = (
                    None if session_id is None else _resolve_persistence_backend(ctx)
                )
                if persistence is None:

                    async def _create_startup(
                        cid: str = configured_id,
                        opts: AgentOptions = options,
                        m: dict[str, Any] = meta,
                        label: str = cfg_id,
                    ) -> None:
                        try:
                            await self.create(cid, opts, m)
                        except BaseException as err:  # noqa: BLE001 - report startup fault
                            self._report_configured_startup_failure(
                                label, "restore", cid, err
                            )

                    self._ownership.track_startup(_create_startup())
                else:

                    async def _restore_startup(
                        pers: SessionPersistence = persistence,
                        cid: str = configured_id,
                        opts: AgentOptions = options,
                        m: dict[str, Any] = meta,
                        label: str = cfg_id,
                    ) -> None:
                        try:
                            await self._restore_or_create_configured(
                                ctx, pers, cid, opts, m
                            )
                        except BaseException as err:  # noqa: BLE001 - report startup fault
                            self._report_configured_startup_failure(
                                label, "restore", cid, err
                            )

                    self._ownership.track_startup(_restore_startup())
                continue

            # Explicit resumeSessionId waits for sessionPersistence to resolve.
            def _on_persistence_ready(
                child_ctx: Context,
                rid: str = str(resume_id),
                opts: AgentOptions = options,
                label: str = cfg_id,
            ) -> None:
                persistence = _resolve_persistence_backend(child_ctx)
                if persistence is None:
                    self._report_configured_startup_failure(
                        label,
                        "resume",
                        rid,
                        RuntimeError("session persistence backend is unavailable"),
                    )
                    return

                async def _do_resume() -> None:
                    try:
                        await self._resume_with(
                            ctx,
                            persistence,
                            {
                                "resumeSessionId": rid,
                                "agentOptions": opts,
                                "setup": lambda _actx, ag: self._apply_configured_route(
                                    ag, opts
                                ),
                            },
                        )
                    except BaseException as err:  # noqa: BLE001 - report startup fault
                        self._report_configured_startup_failure(
                            label, "resume", rid, err
                        )

                self._ownership.track_startup(_do_resume())

            fiber = ctx.inject(["sessionPersistence"], _on_persistence_ready)
            disposers.append(fiber.dispose)

        return disposers

    def _apply_configured_route(self, agent: Agent, options: Mapping[str, Any]) -> None:
        provider = options.get("provider")
        model = options.get("model")
        effort = options.get("reasoningEffort")
        if not provider or not model:
            return
        previous = agent.model.current()
        selection: ModelSelection = {"provider": str(provider), "model": str(model)}
        if effort is not None:
            selection["reasoningEffort"] = str(effort)
        elif (
            previous is not None
            and previous.get("provider") == selection["provider"]
            and previous.get("model") == selection["model"]
            and previous.get("reasoningEffort") is not None
        ):
            selection["reasoningEffort"] = previous["reasoningEffort"]
        agent.model.select(selection)

    def _report_configured_startup_failure(
        self,
        config_id: str,
        action: Literal["restore", "resume"],
        session_id: str,
        error: BaseException,
    ) -> None:
        if not self._ownership.is_active():
            return
        self.ctx.logger.warning(
            f'agent "{config_id}": config-driven {action} of "{session_id}" '
            f"failed: {error_chain(error)}"
        )
        self.ctx.emit(
            "agent-loop/config-start-failed",
            {"sessionId": session_id, "error": error},
        )

    async def _restore_or_create_configured(
        self,
        owner_ctx: Context,
        persistence: SessionPersistence,
        session_id: str,
        agent_options: AgentOptions,
        meta: Mapping[str, Any],
    ) -> None:
        await self._wait_for_draining_configured_identity(owner_ctx, session_id)
        if not self._ownership.is_active():
            return
        try:
            await self._resume_with(
                owner_ctx,
                persistence,
                {
                    "resumeSessionId": session_id,
                    "agentOptions": agent_options,
                    "setup": lambda _actx, ag: self._apply_configured_route(
                        ag, agent_options
                    ),
                },
            )
            return
        except SessionPersistenceNotFoundError:
            if not self._ownership.is_active():
                return
        await self.create(session_id, agent_options, meta)

    async def _wait_for_draining_configured_identity(
        self, owner_ctx: Context, session_id: str
    ) -> None:
        if (
            owner_ctx.agents.get(session_id) is None
            and owner_ctx.sessions.get(session_id) is None
        ):
            return
        loop = asyncio.get_running_loop()
        released: asyncio.Future[None] = loop.create_future()

        def _check_released(*_args: Any) -> None:
            if (
                not released.done()
                and owner_ctx.agents.get(session_id) is None
                and owner_ctx.sessions.get(session_id) is None
            ):
                released.set_result(None)

        disp_agent = owner_ctx.on("agent/disposed", _check_released)
        disp_session = owner_ctx.on("session/disposed", _check_released)
        try:
            _check_released()
            await self._ownership.wait_while_active(released)
        finally:
            disp_agent()
            disp_session()

    async def create(
        self,
        session_id: str,
        options: AgentOptions | Mapping[str, Any] = (),  # type: ignore[assignment]
        meta: Mapping[str, Any] | None = None,
    ) -> ReactLoopAgent:
        """Create and publish one standalone agent and session under *session_id*."""
        opts = dict(options) if isinstance(options, Mapping) else {}
        prep_opts: dict[str, Any] = {"meta": dict(meta)} if meta else {}
        with self._runtime_ctx.sessions.prepare(session_id, prep_opts) as preparation:
            stored = await self._create_stored_session(preparation.session)
            handle = await self._initialize_agent(
                self._runtime_ctx,
                session=preparation.session,
                agent_options=opts,
                source="startup",
                stored=stored,
            )
            return cast(ReactLoopAgent, handle["agent"])

    async def _create_stored_session(self, session: Session) -> _StoredSession | None:
        persistence = _resolve_persistence_backend(self._runtime_ctx)
        if persistence is None:
            return None
        handle = await _maybe_await(
            persistence.create(
                dict(session.header),
                inherited_event_count=session.inherited_event_count,
            )
        )
        return _StoredSession(handle=handle, stored_count=0)

    async def _append_unstored_suffix(
        self, stored: _StoredSession | None, session: Session
    ) -> None:
        if stored is None:
            return
        suffix = session.snapshot_events(stored.stored_count)
        if suffix:
            await _maybe_await(stored.handle.append(suffix))
            stored.stored_count += len(suffix)

    async def create_agent(
        self, owner_ctx: Context, options: CreateAgentOptions
    ) -> AgentHandle:
        """Create an owned agent attached to *owner_ctx*."""
        task = asyncio.ensure_future(self._create_agent_inner(owner_ctx, options))
        self._ownership.track_wrapper(task)
        return await task

    async def _create_agent_inner(
        self, owner_ctx: Context, options: CreateAgentOptions
    ) -> AgentHandle:
        session_id = options.get("sessionId")
        prep_opts: dict[str, Any] = {}
        if "seed" in options:
            prep_opts["seed"] = options["seed"]
        if "meta" in options:
            prep_opts["meta"] = options["meta"]
        if "inheritedEventCount" in options:
            prep_opts["inherited_event_count"] = options["inheritedEventCount"]

        with self._runtime_ctx.sessions.prepare(session_id, prep_opts) as preparation:
            signal = options.get("signal")

            def _create_stored() -> Awaitable[_StoredSession | None]:
                return self._create_stored_session(preparation.session)

            def _release_stored(abandoned: _StoredSession | None) -> Any:
                return abandoned.handle.close() if abandoned is not None else None

            stored = (
                await self._create_stored_session(preparation.session)
                if signal is None
                else await race_abort_call(
                    _create_stored,
                    signal,
                    preparation.session.id,
                    _release_stored,
                )
            )
            return await self._initialize_agent(
                owner_ctx,
                session=preparation.session,
                agent_options=dict(options.get("agentOptions") or {}),
                source="startup",
                stored=stored,
                setup=options.get("setup"),
                signal=signal,
            )

    createAgent = create_agent

    async def resume(
        self, owner_ctx: Context, options: ResumeAgentOptions
    ) -> AgentHandle:
        """Resume an owned agent from the configured persistence service."""
        persistence = _resolve_persistence_backend(self._runtime_ctx)
        if persistence is None:
            raise RuntimeError(
                "cannot resume: session persistence is not configured "
                "(mount a persistence backend)"
            )
        return await self._resume_with(owner_ctx, persistence, options)

    async def _resume_with(
        self,
        owner_ctx: Context,
        persistence: SessionPersistence,
        options: ResumeAgentOptions,
    ) -> AgentHandle:
        task = asyncio.ensure_future(
            self._resume_with_inner(owner_ctx, persistence, options)
        )
        self._ownership.track_wrapper(task)
        return await task

    async def _resume_with_inner(
        self,
        owner_ctx: Context,
        persistence: SessionPersistence,
        options: ResumeAgentOptions,
    ) -> AgentHandle:
        session_id = str(options["resumeSessionId"])
        owner_abort = AbortController()
        unfollow_owner = owner_ctx.fiber.effect(
            lambda: (
                lambda: owner_abort.abort(
                    RuntimeError(
                        f'agent "{session_id}" setup aborted: owner disposed '
                        "during setup"
                    )
                )
            ),
            label=f"agentLoop.resume-load({session_id})",
        )
        fused = AbortController()
        caller_sig = options.get("signal")
        subs: list[Callable[[], None]] = []
        for src_sig in (
            *([] if caller_sig is None else [caller_sig]),
            owner_abort.signal,
            self._ownership.signal,
        ):

            def _forward_abort(s: AbortSignal = src_sig) -> None:
                fused.abort(s.reason)

            subs.append(src_sig.on_abort(_forward_abort))

        handle: SessionHandle | None = None
        preparation: SessionPreparation | None = None
        try:
            try:
                handle = await race_abort_call(
                    lambda: persistence.open(session_id, "write"),
                    fused.signal,
                    session_id,
                    lambda abandoned: abandoned.close(),
                )
                event_state, persisted = cast(
                    tuple[Any, list[SessionEvent]],
                    await race_abort(
                        _maybe_await(handle.read(0, None)), fused.signal, session_id
                    ),
                )
                fused.signal.throw_if_aborted()
                closers = interrupted_turn_closers(persisted)
                if closers:
                    await _maybe_await(handle.append(closers))
                preparation = self._runtime_ctx.sessions.prepare(
                    session_id,
                    {
                        "seed": [*persisted, *closers],
                        "meta": copy.deepcopy(handle.header),
                        "inherited_event_count": handle.inherited_event_count,
                        "event_state": event_state,
                    },
                )
                stored = _StoredSession(
                    handle=handle, stored_count=len(persisted) + len(closers)
                )
                await self._append_unstored_suffix(stored, preparation.session)
            finally:
                for unsub in subs:
                    unsub()
                unfollow_owner()

            owner_ctx.fiber.assertActive()
            if not self._ownership.is_active():
                raise RuntimeError("agent loop is not active")
            handle = None  # initialize_agent owns write-handle rollback and teardown.
            return await self._initialize_agent(
                owner_ctx,
                session=preparation.session,
                agent_options=dict(options.get("agentOptions") or {}),
                source="resume",
                stored=stored,
                setup=options.get("setup"),
                signal=caller_sig,
            )
        finally:
            if preparation is not None:
                preparation.dispose()
            if handle is not None:
                try:
                    await _maybe_await(handle.close())
                except Exception as close_err:  # noqa: BLE001 - best-effort rollback
                    self.ctx.logger.warning(
                        f'agent "{session_id}": failed closing abandoned resume '
                        f"handle: {close_err}"
                    )

    async def _initialize_agent(
        self,
        owner_ctx: Context,
        *,
        session: Session,
        agent_options: dict[str, Any],
        source: SessionStartSource,
        stored: _StoredSession | None = None,
        setup: Callable[..., Any] | None = None,
        signal: AbortSignal | None = None,
    ) -> AgentHandle:
        agent_id = session.id
        loop_ctx = self._runtime_ctx

        abort = AbortController()
        # A distinct lifetime signal: a wake arriving after the agent is fully
        # disposed must not start a new driver. It is aborted only at the very
        # end of teardown, so wakes needed to settle the machine still work.
        lifetime = AbortController()
        unsub_caller = (
            signal.on_abort(
                lambda: abort.abort(
                    signal.reason
                    if isinstance(signal.reason, BaseException)
                    else RuntimeError(f'agent "{agent_id}" creation aborted')
                )
            )
            if signal is not None
            else None
        )
        unsub_factory = self._ownership.signal.on_abort(
            lambda: abort.abort(self._ownership.signal.reason)
        )

        machine: ReactLoopAgent | None = None
        detach_session: Callable[[], None] | None = None
        detach_agent: Callable[[], None] | None = None
        disposing: asyncio.Task[None] | None = None
        publication_done: asyncio.Future[None] | None = None
        ready = False
        unfollow_scope: Callable[[], Any] | None = None
        unfollow_owner: Callable[[], Any] | None = None

        async def _teardown_inner() -> None:
            assert machine is not None
            abort.abort(
                RuntimeError(
                    f'agent "{agent_id}" lifecycle disposed'
                    if self._ownership.is_active()
                    else "agent loop is not active"
                )
            )
            if unsub_caller is not None:
                unsub_caller()
            unsub_factory()
            failures: list[BaseException] = []
            try:
                if publication_done is not None:
                    await publication_done
                machine.cancel({"kind": "disposed"}, {"keepInbox": not ready})
                await machine.when_idle()
            except BaseException as err:  # noqa: BLE001 - aggregate teardown faults
                failures.append(err)
            try:
                await machine.scope.dispose()
            except BaseException as err:  # noqa: BLE001 - aggregate teardown faults
                failures.append(err)
            if stored is not None:
                if detach_session is not None:
                    try:
                        await loop_ctx.sessions.flush(session)
                    except BaseException as err:  # noqa: BLE001 - aggregate teardown faults
                        failures.append(err)
                try:
                    await _maybe_await(stored.handle.close())
                except BaseException as err:  # noqa: BLE001 - aggregate teardown faults
                    failures.append(err)
            try:
                if detach_agent is not None:
                    detach_agent()
                if detach_session is not None:
                    detach_session()
            finally:
                untrack()
            # The lifetime is over: block any late wake from starting a driver.
            lifetime.abort(RuntimeError(f'agent "{agent_id}" lifecycle disposed'))
            _throw_collected_failures(failures, f'agent "{agent_id}" disposal failed')

        async def _teardown() -> None:
            nonlocal disposing
            if disposing is None:
                disposing = asyncio.ensure_future(_teardown_inner())
            await disposing

        async def dispose() -> None:
            try:
                await _teardown()
            finally:
                assert unfollow_owner is not None
                assert unfollow_scope is not None
                await _maybe_await(unfollow_owner())
                await _maybe_await(unfollow_scope())

        untrack = self._ownership.track(_teardown)
        try:
            _assert_agent_options(agent_options)
            owner_ctx.fiber.assertActive()
            if not self._ownership.is_active():
                raise RuntimeError("agent loop is not active")
            if signal is not None and signal.aborted:
                raise _signal_abort_error(signal, agent_id)
            assert self._model_projection_handle is not None
            assert self._inbox_projection_handle is not None
            last_turn = loop_ctx.agents.turn_boundary_of(session)["lastTurn"]
            projections = {
                "modelSelection": self._model_projection_handle.reader_for(session),
                "inbox": self._inbox_projection_handle.reader_for(session),
            }

            def _bind_lifecycle() -> Callable[[], Awaitable[None]]:
                nonlocal machine, unfollow_scope
                created = ReactLoopAgent(
                    loop_ctx,
                    agent_id,
                    agent_options,
                    session,
                    projections,
                    last_turn,
                    lifetime_signal=lifetime.signal,
                )
                machine = created

                def _bind_scope() -> list[Callable[[], Awaitable[None]]]:
                    return [created.scope.raw_dispose, _teardown]

                unfollow_scope = loop_ctx.fiber.effect(
                    _bind_scope, label=f"agentLoop.scope({agent_id})"
                )

                async def _on_owner_dispose() -> None:
                    abort.abort(
                        RuntimeError(
                            f'agent "{agent_id}" setup aborted: owner disposed '
                            "during setup"
                        )
                    )
                    try:
                        await _teardown()
                    finally:
                        assert unfollow_scope is not None
                        await _maybe_await(unfollow_scope())

                return _on_owner_dispose

            unfollow_owner = owner_ctx.fiber.effect(
                _bind_lifecycle, label=f"agentLoop.lifecycle({agent_id})"
            )
        except BaseException:
            untrack()
            if unsub_caller is not None:
                unsub_caller()
            unsub_factory()
            if stored is not None:
                try:
                    await _maybe_await(stored.handle.close())
                except Exception as close_err:  # noqa: BLE001 - preserve primary error
                    self.ctx.logger.warning(
                        f'agent "{agent_id}": handle close after init failure '
                        f"failed: {close_err}"
                    )
            raise

        def _assert_live() -> None:
            if abort.signal.aborted:
                raise _signal_abort_error(abort.signal, agent_id)

        try:
            assert machine is not None
            agent = machine
            _assert_live()

            async def _publish(_sig: AbortSignal) -> AgentHandle:
                nonlocal detach_session, detach_agent, publication_done, ready
                loop = asyncio.get_running_loop()
                publication_done = loop.create_future()
                try:
                    _assert_live()
                    await self._append_unstored_suffix(stored, session)
                    _assert_live()
                    detach_session = agent.ctx.sessions.enter(session)
                    agent.ctx.sessions.announce(session)
                    _assert_live()
                    if setup is not None:
                        setup_res = await race_abort(
                            _maybe_await(setup(agent.ctx, agent)),
                            abort.signal,
                            agent_id,
                        )
                        _assert_live()
                        if setup_res is not None:
                            commit_fn = (
                                setup_res["commit"]
                                if isinstance(setup_res, Mapping)
                                else cast(AgentSetupCommit, setup_res).commit
                            )
                            commit_fn()
                        _assert_live()
                    detach_agent = loop_ctx.agents.enter(agent)
                    await loop_ctx.agents.announce(agent, source, abort.signal)
                    _assert_live()
                    ready = True
                    return {"agent": agent, "dispose": dispose}
                except BaseException:
                    agent.cancel({"kind": "disposed"}, {"keepInbox": True})
                    raise
                finally:
                    assert publication_done is not None
                    publication_done.set_result(None)
                    publication_done = None

            return await agent.run_maintenance(_publish)
        except BaseException:
            try:
                await dispose()
            except Exception as dispose_err:  # noqa: BLE001 - preserve primary error
                self.ctx.logger.warning(
                    f'agent "{agent_id}": rollback dispose failed: {dispose_err}'
                )
            raise


__all__ = [
    "CONFIGURED_AGENT_IDENTITIES_KEY",
    "DEFAULT_MAX_PARALLEL_TOOL_CALLS",
    "AgentLoop",
    "FactoryOwnership",
    "apply_launcher_identities",
    "race_abort",
    "race_abort_call",
    "validate_configured_agents",
]
