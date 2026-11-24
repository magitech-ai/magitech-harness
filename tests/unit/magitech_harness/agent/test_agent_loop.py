"""Unit tests for AgentLoop factory service, lifecycle ownership, and resume."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from animas import Context

from magitech_harness.agent.agent_loop import (
    CONFIGURED_AGENT_IDENTITIES_KEY,
    AgentLoop,
    FactoryOwnership,
    apply_launcher_identities,
    race_abort,
    race_abort_call,
    validate_configured_agents,
)
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.agent.registry import AgentRegistry
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.service import LlmAdapter, LLMService
from magitech_harness.llm.types import GenerateOptions, LlmModelInfo, StreamChunk
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.persistence import SessionHandle, SessionPersistence
from magitech_harness.session.persistence_jsonl import (
    JsonlSessionPersistence,
    JsonlSessionPersistenceService,
)
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.native_executor import apply as apply_native_executor
from magitech_harness.tools.service import ToolService


class _OkAdapter(LlmAdapter):
    def list_models(self, provider: str) -> list[LlmModelInfo]:
        return [{"id": "m", "name": "M", "provider": provider}]

    async def stream(self, _options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        yield {"type": "text-delta", "index": 0, "text": "hello back"}
        yield {"type": "finish", "reason": {"kind": "stop"}}


async def _boot_context(
    *,
    sessions_dir: Path | None = None,
    configured_agents: list[dict[str, Any]] | None = None,
    identities: dict[str, dict[str, Any]] | None = None,
) -> Context:
    ctx = Context()
    if identities is not None:
        ctx.provide(CONFIGURED_AGENT_IDENTITIES_KEY, identities)
    ctx.use(SystemPromptService, {"include_identity": False})
    ctx.use(ToolService)
    ctx.use(apply_native_executor)
    ctx.use(LLMService)
    ctx.use(SessionProjectionRegistry)
    if sessions_dir is not None:
        ctx.use(SessionService, {"base_storage_dir": str(sessions_dir)})
        ctx.use(
            JsonlSessionPersistenceService,
            {"root": str(sessions_dir), "live_batch_delay": 0.01},
        )
    else:
        ctx.use(SessionService)
    ctx.use(AgentRegistry)
    ctx.use(AgentLoop, {"agents": configured_agents or []})
    await ctx.flush()
    ctx.llm.register_adapter(["ok"], _OkAdapter())
    return ctx


def test_configured_agent_validation_and_launcher_identity_helpers() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_configured_agents(
            [{"id": "a", "sessionId": "s1", "resumeSessionId": "s2"}]
        )
    with pytest.raises(ValueError, match="duplicate exact session identity"):
        validate_configured_agents(
            [{"id": "a", "sessionId": "dup"}, {"id": "b", "resumeSessionId": "dup"}]
        )
    applied = apply_launcher_identities(
        [{"id": "main", "sessionId": "cfg-id", "provider": "ok", "model": "m"}],
        {"main": {"id": "launcher-id", "resume": True}},
    )
    assert applied == [
        {"id": "main", "resumeSessionId": "launcher-id", "provider": "ok", "model": "m"}
    ]


@pytest.mark.asyncio
async def test_agent_loop_create_setup_commit_and_prompt_variables() -> None:
    ctx = await _boot_context()
    seen_vars: dict[str, Any] = {}
    handle = await ctx.agents.create(
        {
            "sessionId": "al-create-1",
            "agentOptions": {"provider": "ok", "model": "m"},
            "setup": lambda _agent_ctx, agent: {
                "commit": lambda: seen_vars.setdefault("id", agent.id)
            },
        }
    )
    agent = handle["agent"]
    assert seen_vars["id"] == "al-create-1"
    assembly = await ctx.system_prompt.assemble({"agent": agent})
    assert assembly["variables"]["provider"] == "ok"
    assert assembly["variables"]["model"] == "m"

    agent.followup(
        dict(
            create_user_message(
                content=[{"type": "text", "text": "ping"}], source={"kind": "user"}
            )
        )
    )
    await agent.when_idle()
    assert agent.session.events[-1].payload["reason"] == {"kind": "completed"}

    await handle["dispose"]()
    assert ctx.agents.get("al-create-1") is None
    assert ctx.sessions.get("al-create-1") is None
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_agent_loop_resume_repairs_interrupted_turn_and_validates_persistence(
    tmp_path: Path,
) -> None:
    mem_ctx = await _boot_context()
    with pytest.raises(RuntimeError, match="session persistence is not configured"):
        await mem_ctx.agents.resume({"resumeSessionId": "missing"})
    await mem_ctx.fiber.dispose()

    sessions_dir = tmp_path / "sessions"
    ctx1 = await _boot_context(sessions_dir=sessions_dir)
    h1 = await ctx1.agents.create(
        {
            "sessionId": "al-resume-1",
            "agentOptions": {"provider": "ok", "model": "m"},
        }
    )
    session = h1["agent"].session
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    await h1["dispose"]()
    await ctx1.fiber.dispose()

    ctx2 = await _boot_context(sessions_dir=sessions_dir)
    h2 = await ctx2.agents.resume(
        {
            "resumeSessionId": "al-resume-1",
            "agentOptions": {"provider": "ok", "model": "m"},
        }
    )
    events: list[SessionEvent] = h2["agent"].session.events
    types = [ev.type for ev in events]
    assert types[-2:] == ["step/end", "turn/end"]
    assert events[-1].payload["reason"] == {"kind": "interrupted"}
    await h2["dispose"]()
    await ctx2.fiber.dispose()


@pytest.mark.asyncio
async def test_agent_loop_setup_failure_and_abort_cleanup(tmp_path: Path) -> None:
    ctx = await _boot_context(sessions_dir=tmp_path / "sessions")

    def failing_setup(_actx: Context, _agent: Any) -> None:
        raise RuntimeError("setup boom")

    with pytest.raises(RuntimeError, match="setup boom"):
        await ctx.agents.create(
            {
                "sessionId": "al-fail",
                "agentOptions": {"provider": "ok", "model": "m"},
                "setup": failing_setup,
            }
        )
    assert ctx.agents.get("al-fail") is None
    assert ctx.sessions.get("al-fail") is None

    pre_aborted = AbortSignal()
    pre_aborted.abort(RuntimeError("cancelled before create"))
    with pytest.raises(RuntimeError, match="cancelled before create"):
        await ctx.agents.create(
            {
                "sessionId": "al-aborted",
                "agentOptions": {"provider": "ok", "model": "m"},
                "signal": pre_aborted,
            }
        )
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_agent_loop_configured_startup_restore_or_create(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    ctx1 = await _boot_context(
        sessions_dir=sessions_dir,
        configured_agents=[
            {
                "id": "primary",
                "sessionId": "fixed-cfg-id",
                "provider": "ok",
                "model": "m",
                "cwd": str(tmp_path),
            }
        ],
    )
    # Wait briefly for startup task to finish publishing configured agent
    for _ in range(50):
        if ctx1.agents.get("fixed-cfg-id") is not None:
            break
        await asyncio.sleep(0.01)
    assert ctx1.agents.get("fixed-cfg-id") is not None
    await ctx1.fiber.dispose()

    # Reboot: restore_or_create_configured resumes the existing session cleanly
    ctx2 = await _boot_context(
        sessions_dir=sessions_dir,
        configured_agents=[
            {
                "id": "primary",
                "sessionId": "fixed-cfg-id",
                "provider": "ok",
                "model": "m",
            }
        ],
    )
    for _ in range(50):
        if ctx2.agents.get("fixed-cfg-id") is not None:
            break
        await asyncio.sleep(0.01)
    assert ctx2.agents.get("fixed-cfg-id") is not None
    await ctx2.fiber.dispose()


class _FaultyHandle(SessionHandle):
    """Delegating handle whose close/read/flush/append behavior can be scripted."""

    def __init__(
        self,
        inner: SessionHandle,
        *,
        fail_close: bool = False,
        fail_flush: bool = False,
        on_read: Any = None,
        on_append: Any = None,
    ) -> None:
        self._inner = inner
        self._fail_close = fail_close
        self._fail_flush = fail_flush
        self._on_read = on_read
        self._on_append = on_append

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def access(self) -> Any:
        return self._inner.access

    @property
    def header(self) -> dict[str, Any]:
        return self._inner.header

    @property
    def inherited_event_count(self) -> int:
        return self._inner.inherited_event_count

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def append(self, events: list[SessionEvent]) -> None:
        if self._on_append is not None:
            self._on_append(events)
        self._inner.append(events)

    def read(
        self, offset: int = 0, length: int | None = None
    ) -> tuple[str, list[SessionEvent]]:
        if self._on_read is not None:
            res = self._on_read()
            if res is not None:
                return res  # type: ignore[no-any-return]
        return self._inner.read(offset, length)

    def flush(self) -> None:
        if self._fail_flush:
            raise OSError("flush boom")
        self._inner.flush()

    def close(self) -> None:
        self._inner.close()
        if self._fail_close:
            raise OSError("close boom")


class _DelegatingPersistence(SessionPersistence):
    """Direct SessionPersistence service slot for testing hook/fault branches."""

    def __init__(
        self,
        inner: JsonlSessionPersistence,
        *,
        fail_close_on_open: bool = False,
        fail_close_on_create: bool = False,
        fail_flush: bool = False,
        on_create: Any = None,
        on_open: Any = None,
        on_read: Any = None,
        on_append: Any = None,
    ) -> None:
        self._inner = inner
        self._fail_close_on_open = fail_close_on_open
        self._fail_close_on_create = fail_close_on_create
        self._fail_flush = fail_flush
        self._on_create = on_create
        self._on_open = on_open
        self._on_read = on_read
        self._on_append = on_append

    def create(
        self,
        header: dict[str, Any],
        inherited_event_count: int | None = None,
    ) -> SessionHandle:
        if self._on_create is not None:
            self._on_create(header)
        return _FaultyHandle(
            self._inner.create(header, inherited_event_count),
            fail_close=self._fail_close_on_create or self._fail_close_on_open,
            fail_flush=self._fail_flush,
            on_append=self._on_append,
        )

    def open(self, session_id: str, access: Any) -> SessionHandle:
        if self._on_open is not None:
            self._on_open(session_id, access)
        return _FaultyHandle(
            self._inner.open(session_id, access),
            fail_close=self._fail_close_on_open,
            fail_flush=self._fail_flush,
            on_read=self._on_read,
            on_append=self._on_append,
        )

    def flush(self) -> None:
        self._inner.flush()

    def stat(self, session_id: str) -> Any:
        return self._inner.stat(session_id)

    def list(self) -> list[Any]:
        return self._inner.list()


@pytest.mark.asyncio
async def test_helper_primitives_abort_races_ownership_and_validations() -> None:
    # apply_launcher_identities with resume=False and validate_configured_agents plain entry
    non_resume = apply_launcher_identities(
        [{"id": "a", "resumeSessionId": "old", "provider": "ok", "model": "m"}],
        {"a": {"id": "fresh-id", "resume": False}},
    )
    assert non_resume == [
        {"id": "a", "sessionId": "fresh-id", "provider": "ok", "model": "m"}
    ]
    validate_configured_agents(
        [{"id": "no-session-id", "provider": "ok", "model": "m"}]
    )

    # FactoryOwnership: wait_while_active normal completion & early dispose wakeup + multi errors
    own_ok = FactoryOwnership(Context().fiber)
    done = False

    async def quick_job() -> None:
        nonlocal done
        await asyncio.sleep(0)
        done = True

    await own_ok.wait_while_active(quick_job())
    assert done is True

    own_interrupt = FactoryOwnership(Context().fiber)
    never = asyncio.Event()
    waiter = asyncio.create_task(own_interrupt.wait_while_active(never.wait()))
    await asyncio.sleep(0)
    await own_interrupt.dispose()
    await waiter
    never.set()

    own_one_err = FactoryOwnership(Context().fiber)

    async def dispose_one() -> None:
        raise RuntimeError("single-dispose")

    own_one_err.track(dispose_one)
    with pytest.raises(RuntimeError, match="single-dispose"):
        await own_one_err.dispose()

    own_multi_err = FactoryOwnership(Context().fiber)

    async def dispose_err1() -> None:
        raise RuntimeError("err-1")

    async def dispose_err2() -> None:
        raise ValueError("err-2")

    own_multi_err.track(dispose_err1)
    own_multi_err.track(dispose_err2)
    with pytest.raises(ExceptionGroup) as eg_info:
        await own_multi_err.dispose()
    assert len(eg_info.value.exceptions) == 2

    # race_abort branches
    aborted_str = AbortSignal()
    aborted_str.abort("plain-string-reason")
    with pytest.raises(RuntimeError, match='agent "a1" creation aborted'):
        await race_abort(42, aborted_str, "a1")

    live = AbortSignal()
    assert await race_abort(99, live, "a1") == 99

    async def failing_coro() -> int:
        await asyncio.sleep(0)
        raise ValueError("op failed")

    with pytest.raises(ValueError, match="op failed"):
        await race_abort(failing_coro(), live, "a1")

    abort_mid = AbortSignal()

    async def blocked_coro() -> int:
        await asyncio.Event().wait()
        return 1

    async def abort_soon() -> None:
        await asyncio.sleep(0)
        abort_mid.abort("midway")

    asyncio.create_task(abort_soon())
    with pytest.raises(RuntimeError, match='agent "a2" creation aborted'):
        await race_abort(blocked_coro(), abort_mid, "a2")

    # race_abort_call branches: sync non-awaitable, async normal, async abandoned (sync/async release + err)
    assert await race_abort_call(lambda: "sync-ok", AbortSignal(), "a3") == "sync-ok"

    async def async_ok() -> str:
        await asyncio.sleep(0)
        return "async-val"

    assert await race_abort_call(async_ok, AbortSignal(), "a3") == "async-val"

    sig_pre = AbortSignal()
    sig_pre.abort(RuntimeError("pre-call"))
    with pytest.raises(RuntimeError, match="pre-call"):
        await race_abort_call(async_ok, sig_pre, "a3")

    sig_mid_sync_release = AbortSignal()
    gate_sync = asyncio.Event()
    released_sync: list[str] = []

    async def slow_sync_rel() -> str:
        await gate_sync.wait()
        return "late-sync-handle"

    async def trigger_sync_abort() -> None:
        await asyncio.sleep(0)
        sig_mid_sync_release.abort("too-late-sync")
        await asyncio.sleep(0)
        gate_sync.set()

    asyncio.create_task(trigger_sync_abort())
    with pytest.raises(RuntimeError, match='agent "a3b" creation aborted'):
        await race_abort_call(
            slow_sync_rel,
            sig_mid_sync_release,
            "a3b",
            release_abandoned=released_sync.append,
        )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert released_sync == ["late-sync-handle"]

    sig_mid_release = AbortSignal()
    gate = asyncio.Event()
    released_async: list[str] = []
    released_async_done = asyncio.Event()

    async def slow_resource() -> str:
        await gate.wait()
        return "late-handle"

    async def release_async(val: str) -> None:
        released_async.append(val)
        released_async_done.set()

    async def trigger_abort() -> None:
        await asyncio.sleep(0)
        sig_mid_release.abort("too-late")
        await asyncio.sleep(0)
        gate.set()

    asyncio.create_task(trigger_abort())
    with pytest.raises(RuntimeError, match='agent "a4" creation aborted'):
        await race_abort_call(
            slow_resource,
            sig_mid_release,
            "a4",
            release_abandoned=release_async,
        )
    await asyncio.wait_for(released_async_done.wait(), timeout=2.0)
    assert released_async == ["late-handle"]

    # Aborted async op with release_abandoned failing vs release_abandoned=None resolving cleanly
    sig_mid_err = AbortSignal()
    gate_err = asyncio.Event()

    async def slow_failing_resource() -> str:
        await gate_err.wait()
        raise ValueError("late fail")

    async def trigger_abort_err() -> None:
        await asyncio.sleep(0)
        sig_mid_err.abort("abort-wins")
        await asyncio.sleep(0)
        gate_err.set()

    asyncio.create_task(trigger_abort_err())
    with pytest.raises(RuntimeError, match='agent "a5" creation aborted'):
        await race_abort_call(
            slow_failing_resource,
            sig_mid_err,
            "a5",
            release_abandoned=released_sync.append,
        )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    sig_mid_none = AbortSignal()
    gate_none = asyncio.Event()

    async def slow_ignored_resource() -> str:
        await gate_none.wait()
        return "ignored"

    async def trigger_abort_none() -> None:
        await asyncio.sleep(0)
        sig_mid_none.abort("abort-wins-no-cb")
        await asyncio.sleep(0)
        gate_none.set()

    asyncio.create_task(trigger_abort_none())
    with pytest.raises(RuntimeError, match='agent "a6" creation aborted'):
        await race_abort_call(
            slow_ignored_resource, sig_mid_none, "a6", release_abandoned=None
        )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Validation checks: maxTokens and maxParallelToolCalls
    with pytest.raises((TypeError, ValueError)):
        AgentLoop(Context(), {"maxParallelToolCalls": 0})
    with pytest.raises((TypeError, ValueError)):
        AgentLoop(Context(), {"maxParallelToolCalls": True})

    ctx = await _boot_context()
    with pytest.raises((TypeError, ValueError)):
        await ctx.agents.create(
            {
                "sessionId": "bad-max-tok",
                "agentOptions": {"provider": "ok", "model": "m", "maxTokens": 0},
            }
        )
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_configured_startup_modes_failures_routes_and_draining(
    tmp_path: Path,
) -> None:
    # 1. No persistence backend: create_startup (auto session id + route without model + explicit reasoningEffort)
    ctx_mem = await _boot_context(
        configured_agents=[
            {"id": "minimal-worker"},
            {
                "id": "reasoner",
                "sessionId": "cfg-reasoner",
                "provider": "ok",
                "model": "m",
                "reasoningEffort": "high",
            },
        ]
    )
    for _ in range(50):
        if len(ctx_mem.agents.list()) == 2:
            break
        await asyncio.sleep(0.01)
    reasoner = ctx_mem.agents.get("cfg-reasoner")
    assert reasoner is not None
    assert reasoner.model.current() == {
        "provider": "ok",
        "model": "m",
        "reasoningEffort": "high",
    }
    await ctx_mem.fiber.dispose()

    # 2. _create_startup failure + hollow sessionPersistence for resumeSessionId -> emits agent-loop/config-start-failed
    failures: list[dict[str, Any]] = []
    ctx_fail = Context()
    ctx_fail.on(
        "agent-loop/config-start-failed", lambda payload: failures.append(payload)
    )
    ctx_fail.use(SystemPromptService, {"include_identity": False})
    ctx_fail.use(ToolService)
    ctx_fail.use(apply_native_executor)
    ctx_fail.use(LLMService)
    ctx_fail.use(SessionProjectionRegistry)
    ctx_fail.use(SessionService)
    ctx_fail.provide("sessionPersistence", {})
    ctx_fail.use(AgentRegistry)
    ctx_fail.use(
        AgentLoop,
        {
            "agents": [
                {"id": "bad-create", "sessionId": "s-bad", "maxTokens": 0},
                {"id": "missing-persist", "resumeSessionId": "no-store"},
            ]
        },
    )
    await ctx_fail.flush()
    for _ in range(50):
        if len(failures) >= 2:
            break
        await asyncio.sleep(0.01)
    assert any(f["sessionId"] == "s-bad" for f in failures)
    assert any(f["sessionId"] == "no-store" for f in failures)
    await ctx_fail.fiber.dispose()

    # 3. Seed persisted session via JsonlSessionPersistenceService, then reopen via direct SessionPersistence slot
    #    (tests direct SessionPersistence slot, resumeSessionId success/failure, inherited reasoningEffort, bad-restore)
    sessions_dir = tmp_path / "sessions"
    ctx_resume = await _boot_context(sessions_dir=sessions_dir)
    seed_handle = await ctx_resume.agents.create(
        {"sessionId": "persisted-cfg", "agentOptions": {"provider": "ok", "model": "m"}}
    )
    seed_handle["agent"].model.select(
        {"provider": "ok", "model": "m", "reasoningEffort": "medium"}
    )
    await seed_handle["dispose"]()
    await ctx_resume.fiber.dispose()

    # Reopen via direct SessionPersistence slot: resumeSessionId = "persisted-cfg" (inherits reasoningEffort)
    # and nonexistent resumeSessionId -> reports failure; and restore_startup error (disposed/invalid)
    ctx_reopen = Context()
    reopen_failures: list[dict[str, Any]] = []
    ctx_reopen.on("agent-loop/config-start-failed", lambda p: reopen_failures.append(p))
    ctx_reopen.use(SystemPromptService, {"include_identity": False})
    ctx_reopen.use(ToolService)
    ctx_reopen.use(apply_native_executor)
    ctx_reopen.use(LLMService)
    ctx_reopen.use(SessionProjectionRegistry)
    ctx_reopen.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx_reopen.provide(
        "sessionPersistence",
        _DelegatingPersistence(JsonlSessionPersistence(root=sessions_dir)),
    )
    ctx_reopen.use(AgentRegistry)
    ctx_reopen.use(
        AgentLoop,
        {
            "agents": [
                {
                    "id": "resumed",
                    "resumeSessionId": "persisted-cfg",
                    "provider": "ok",
                    "model": "m",
                },
                {"id": "bad-restore", "sessionId": "bad-restore-id", "maxTokens": 0},
                {"id": "bad-resume", "resumeSessionId": "does-not-exist"},
            ]
        },
    )
    await ctx_reopen.flush()
    ctx_reopen.llm.register_adapter(["ok"], _OkAdapter())
    for _ in range(50):
        if (
            ctx_reopen.agents.get("persisted-cfg") is not None
            and len(reopen_failures) >= 2
        ):
            break
        await asyncio.sleep(0.01)
    resumed = ctx_reopen.agents.get("persisted-cfg")
    assert resumed is not None
    assert resumed.model.current() == {
        "provider": "ok",
        "model": "m",
        "reasoningEffort": "medium",
    }
    assert any(f["sessionId"] == "does-not-exist" for f in reopen_failures)
    assert any(f["sessionId"] == "bad-restore-id" for f in reopen_failures)
    await ctx_reopen.fiber.dispose()

    # 4. Draining active agent + session holding same sessionId before AgentLoop mounts
    ctx_drain = Context()
    ctx_drain.use(SystemPromptService, {"include_identity": False})
    ctx_drain.use(ToolService)
    ctx_drain.use(apply_native_executor)
    ctx_drain.use(LLMService)
    ctx_drain.use(SessionProjectionRegistry)
    ctx_drain.use(SessionService)
    ctx_drain.use(AgentRegistry)
    await ctx_drain.flush()

    pre_prep = ctx_drain.sessions.prepare("drain-id")

    async def release_pre_session() -> None:
        await asyncio.sleep(0.02)
        pre_prep.dispose()

    assert pre_prep.session.id == "drain-id"
    asyncio.create_task(release_pre_session())
    ctx_drain.use(
        AgentLoop,
        {"agents": [{"id": "after-drain", "sessionId": "drain-id"}]},
    )
    await ctx_drain.flush()
    for _ in range(50):
        if ctx_drain.agents.get("drain-id") is not None:
            break
        await asyncio.sleep(0.01)
    assert ctx_drain.agents.get("drain-id") is not None
    await ctx_drain.fiber.dispose()

    # 5. Factory disposed while draining configured identity (exits cleanly without crash)
    ctx_drain_abort = Context()
    ctx_drain_abort.use(SystemPromptService, {"include_identity": False})
    ctx_drain_abort.use(ToolService)
    ctx_drain_abort.use(apply_native_executor)
    ctx_drain_abort.use(LLMService)
    ctx_drain_abort.use(SessionProjectionRegistry)
    ctx_drain_abort.use(SessionService)
    ctx_drain_abort.use(AgentRegistry)
    await ctx_drain_abort.flush()
    ctx_drain_abort.sessions.create("never-freed")
    ctx_drain_abort.use(
        AgentLoop,
        {"agents": [{"id": "wont-boot", "sessionId": "never-freed"}]},
    )
    await ctx_drain_abort.flush()
    await asyncio.sleep(0.01)
    await ctx_drain_abort.fiber.dispose()


@pytest.mark.asyncio
async def test_create_seed_meta_resume_rollback_owner_dispose_and_teardown_errors(
    tmp_path: Path,
) -> None:
    # 1. create_agent with seed, meta, inheritedEventCount, and persisted suffix flush
    sessions_dir = tmp_path / "sessions"
    ctx = await _boot_context(sessions_dir=sessions_dir)
    seed_events = [
        SessionEvent(
            id="seed-ev-0",
            session_id="seeded-1",
            type="user/message",
            seq=0,
            timestamp=1700000000.0,
            surface_op="append",
            payload={
                "message": create_user_message(
                    content=[{"type": "text", "text": "inherited"}]
                )
            },
        ),
        SessionEvent(
            id="seed-ev-1",
            session_id="seeded-1",
            type="session/end-seed",
            seq=1,
            timestamp=1700000001.0,
            payload={"inherited": True},
        ),
    ]
    handle = await ctx.agents.create(
        {
            "sessionId": "seeded-1",
            "meta": {"cwd": str(tmp_path), "isSeeded": True},
            "inheritedEventCount": 1,
            "seed": seed_events,
            "agentOptions": {"provider": "ok", "model": "m"},
        }
    )
    assert len(handle["agent"].session.events) >= 1
    await handle["dispose"]()

    # Dispose while owner context fiber tears down during setup
    setup_entered = asyncio.Event()
    setup_release = asyncio.Event()

    async def slow_setup(_actx: Context, _agent: Any) -> None:
        setup_entered.set()
        await setup_release.wait()

    create_task = asyncio.create_task(
        ctx.agents.create(
            {
                "sessionId": "owner-disposed-during-setup",
                "agentOptions": {"provider": "ok", "model": "m"},
                "setup": slow_setup,
            }
        )
    )
    await setup_entered.wait()
    await ctx.fiber.dispose()
    setup_release.set()
    with pytest.raises(RuntimeError):
        await create_task

    # 2. Resume error paths with _DelegatingPersistence (inactive factory during read,
    #    _initialize_agent error closing owned_handle, and close() raising warning)
    ctx2 = Context()
    ctx2.use(SystemPromptService, {"include_identity": False})
    ctx2.use(ToolService)
    ctx2.use(apply_native_executor)
    ctx2.use(LLMService)
    ctx2.use(SessionProjectionRegistry)
    ctx2.use(SessionService, {"base_storage_dir": str(sessions_dir)})

    wrapper = _DelegatingPersistence(
        JsonlSessionPersistence(root=sessions_dir),
        fail_close_on_open=False,
    )
    ctx2.provide("sessionPersistence", wrapper)
    ctx2.use(AgentRegistry)
    ctx2.use(AgentLoop)
    await ctx2.flush()
    ctx2.llm.register_adapter(["ok"], _OkAdapter())

    # setup fails on resume -> closes owned_handle cleanly and re-raises setup error
    with pytest.raises(RuntimeError, match="resume-setup-boom"):
        await ctx2.agents.resume(
            {
                "resumeSessionId": "seeded-1",
                "agentOptions": {"provider": "ok", "model": "m"},
                "setup": lambda _c, _a: (_ for _ in ()).throw(
                    RuntimeError("resume-setup-boom")
                ),
            }
        )

    # read fails while handle.close() also fails -> logs warning without masking read error
    wrapper._fail_close_on_open = True
    wrapper._on_read = lambda: (_ for _ in ()).throw(RuntimeError("read-boom"))
    with pytest.raises(RuntimeError, match="read-boom"):
        await ctx2.agents.resume(
            {
                "resumeSessionId": "seeded-1",
                "agentOptions": {"provider": "ok", "model": "m"},
            }
        )

    # Factory disposed inside read hook during resume -> returns inactive abort/RuntimeError
    wrapper_dispose = _DelegatingPersistence(
        JsonlSessionPersistence(root=sessions_dir),
    )

    # Replace wrapper on_read to trigger owner dispose during read
    ctx3 = Context()
    ctx3.use(SystemPromptService, {"include_identity": False})
    ctx3.use(ToolService)
    ctx3.use(apply_native_executor)
    ctx3.use(LLMService)
    ctx3.use(SessionProjectionRegistry)
    ctx3.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx3.provide("sessionPersistence", wrapper_dispose)
    ctx3.use(AgentRegistry)
    ctx3.use(AgentLoop)
    await ctx3.flush()

    def dispose_during_read() -> None:
        asyncio.get_running_loop().create_task(ctx3.fiber.dispose())

    wrapper_dispose._on_read = dispose_during_read
    with pytest.raises(RuntimeError):
        await ctx3.agents.resume(
            {
                "resumeSessionId": "seeded-1",
                "agentOptions": {"provider": "ok", "model": "m"},
            }
        )
    await ctx2.fiber.dispose()
    await ctx3.fiber.dispose()

    # 3. Live agent teardown collecting errors from sessions.flush and stored.handle.close
    ctx4 = Context()
    ctx4.use(SystemPromptService, {"include_identity": False})
    ctx4.use(ToolService)
    ctx4.use(apply_native_executor)
    ctx4.use(LLMService)
    ctx4.use(SessionProjectionRegistry)
    ctx4.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx4.provide(
        "sessionPersistence",
        _DelegatingPersistence(
            JsonlSessionPersistence(root=sessions_dir),
            fail_close_on_open=True,
        ),
    )
    ctx4.use(AgentRegistry)
    ctx4.use(AgentLoop)
    await ctx4.flush()
    ctx4.llm.register_adapter(["ok"], _OkAdapter())

    caller_sig = AbortSignal()
    h4 = await ctx4.agents.create(
        {
            "sessionId": "teardown-errs",
            "agentOptions": {"provider": "ok", "model": "m"},
            "signal": caller_sig,
        }
    )
    # Dispose raises OSError("close boom") from stored.handle.close
    with pytest.raises(OSError, match="close boom"):
        await h4["dispose"]()
    await ctx4.fiber.dispose()


@pytest.mark.asyncio
async def test_agent_loop_remaining_route_drain_inactive_and_teardown_paths(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    # 1. Seed two persisted sessions, then restore them via configured_agents WITH persistence:
    #    - "cfg-no-route": hits _apply_configured_route early return (454)
    #    - "cfg-effort": hits _apply_configured_route explicit reasoningEffort (458)
    seed_ctx = await _boot_context(sessions_dir=sessions_dir)
    for sid in ("cfg-no-route", "cfg-effort"):
        h = await seed_ctx.agents.create(
            {"sessionId": sid, "agentOptions": {"provider": "ok", "model": "m"}}
        )
        await h["dispose"]()
    await seed_ctx.fiber.dispose()

    restore_ctx = await _boot_context(
        sessions_dir=sessions_dir,
        configured_agents=[
            {"id": "no-route", "sessionId": "cfg-no-route"},
            {
                "id": "with-effort",
                "sessionId": "cfg-effort",
                "provider": "ok",
                "model": "m",
                "reasoningEffort": "high",
            },
        ],
    )
    for _ in range(50):
        if (
            restore_ctx.agents.get("cfg-no-route") is not None
            and restore_ctx.agents.get("cfg-effort") is not None
        ):
            break
        await asyncio.sleep(0.01)
    ag_effort = restore_ctx.agents.get("cfg-effort")
    assert ag_effort is not None
    assert ag_effort.model.current() == {
        "provider": "ok",
        "model": "m",
        "reasoningEffort": "high",
    }
    await restore_ctx.fiber.dispose()

    # 2. _wait_for_draining_configured_identity WITH persistence (523-541) + double event check (528)
    #    and factory dispose while draining (496)
    ctx_drain = Context()
    ctx_drain.use(SystemPromptService, {"include_identity": False})
    ctx_drain.use(ToolService)
    ctx_drain.use(apply_native_executor)
    ctx_drain.use(LLMService)
    ctx_drain.use(SessionProjectionRegistry)
    ctx_drain.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx_drain.use(
        JsonlSessionPersistenceService,
        {"root": str(sessions_dir), "live_batch_delay": 0.01},
    )
    ctx_drain.use(AgentRegistry)
    await ctx_drain.flush()
    pre_prep = ctx_drain.sessions.prepare("drain-persisted")
    detach_pre = ctx_drain.sessions.enter(pre_prep.session)
    ctx_drain.use(
        AgentLoop,
        {"agents": [{"id": "after-drain-p", "sessionId": "drain-persisted"}]},
    )
    await ctx_drain.flush()
    await asyncio.sleep(0)
    # Startup task is now suspended inside _wait_for_draining_configured_identity (523-541).
    # First emit agent/disposed while session still exists -> False branch of _check_released
    ctx_drain.emit("agent/disposed", {"id": "drain-persisted"})
    detach_pre()
    pre_prep.dispose()
    # Second emit after released.done() is True -> 528->exit branch
    ctx_drain.emit("agent/disposed", {"id": "drain-persisted"})
    for _ in range(50):
        if ctx_drain.agents.get("drain-persisted") is not None:
            break
        await asyncio.sleep(0.01)
    assert ctx_drain.agents.get("drain-persisted") is not None
    await ctx_drain.fiber.dispose()

    # Factory disposed while draining configured identity WITH persistence -> hits line 496
    ctx_drain_abort = Context()
    ctx_drain_abort.use(SystemPromptService, {"include_identity": False})
    ctx_drain_abort.use(ToolService)
    ctx_drain_abort.use(apply_native_executor)
    ctx_drain_abort.use(LLMService)
    ctx_drain_abort.use(SessionProjectionRegistry)
    ctx_drain_abort.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx_drain_abort.use(
        JsonlSessionPersistenceService,
        {"root": str(sessions_dir), "live_batch_delay": 0.01},
    )
    ctx_drain_abort.use(AgentRegistry)
    await ctx_drain_abort.flush()
    prep_nf = ctx_drain_abort.sessions.prepare("never-freed-p")
    detach_nf = ctx_drain_abort.sessions.enter(prep_nf.session)
    ctx_drain_abort.use(
        AgentLoop,
        {"agents": [{"id": "wont-boot-p", "sessionId": "never-freed-p"}]},
    )
    await ctx_drain_abort.flush()
    await asyncio.sleep(0)
    await ctx_drain_abort.fiber.dispose()
    detach_nf()
    prep_nf.dispose()

    # 3. Inactive-window tests via yielding fiber disposer BEFORE AgentLoop teardown:
    #    - _report_configured_startup_failure when inactive (476)
    #    - _restore_or_create_configured NotFound when inactive (512)
    #    - resume after read when inactive (706)
    #    - _initialize_agent when inactive (839)
    from magitech_harness.session.persistence import SessionPersistenceNotFoundError

    disposing_started = asyncio.Event()
    release_dispose = asyncio.Event()

    async def pause_on_dispose() -> None:
        disposing_started.set()
        await release_dispose.wait()

    ctx_win = Context()
    ctx_win.use(SystemPromptService, {"include_identity": False})
    ctx_win.use(ToolService)
    ctx_win.use(apply_native_executor)
    ctx_win.use(LLMService)
    ctx_win.use(SessionProjectionRegistry)
    ctx_win.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    win_persist = _DelegatingPersistence(JsonlSessionPersistence(root=sessions_dir))
    ctx_win.provide("sessionPersistence", win_persist)
    ctx_win.use(AgentRegistry)
    ctx_win.use(AgentLoop)
    await ctx_win.flush()

    al_comp = ctx_win.registry.get(AgentLoop)
    assert al_comp is not None
    al_fiber = al_comp.fibers[0]
    al_fiber.effect(lambda: pause_on_dispose)

    async def wait_win_read() -> tuple[str, list[SessionEvent]]:
        await disposing_started.wait()
        return ("detached", [])

    win_persist._on_read = lambda: wait_win_read()
    resume_task = asyncio.create_task(
        ctx_win.agents.resume(
            {
                "resumeSessionId": "cfg-no-route",
                "agentOptions": {"provider": "ok", "model": "m"},
            }
        )
    )
    await asyncio.sleep(0)
    dispose_task = asyncio.create_task(al_fiber.dispose())
    await disposing_started.wait()

    # While AgentLoop's own fiber is UNLOADING (ownership.is_active() is False, ctx_win.sessions & agents active!),
    # create_agent hits line 839 and resume hits line 706!
    with pytest.raises(RuntimeError, match="agent loop is not active"):
        await ctx_win.agents.create(
            {
                "sessionId": "in-disposing-window",
                "agentOptions": {"provider": "ok", "model": "m"},
            }
        )
    with pytest.raises(RuntimeError, match="agent loop is not active"):
        await resume_task
    release_dispose.set()
    await dispose_task
    await ctx_win.fiber.dispose()

    # Also cover 476 (_report_configured_startup_failure inactive) & 512 (_restore_or_create_configured inactive)
    ctx_win2 = Context()
    ctx_win2.use(SystemPromptService, {"include_identity": False})
    ctx_win2.use(ToolService)
    ctx_win2.use(apply_native_executor)
    ctx_win2.use(LLMService)
    ctx_win2.use(SessionProjectionRegistry)
    ctx_win2.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx_win2.use(AgentRegistry)
    started_cfg_open = asyncio.Event()
    win2_disposing = asyncio.Event()
    win2_release = asyncio.Event()
    win2_failures: list[dict[str, Any]] = []
    ctx_win2.on("agent-loop/config-start-failed", win2_failures.append)

    async def pause_win2() -> None:
        win2_disposing.set()
        await win2_release.wait()

    class _Win2Persist(_DelegatingPersistence):
        def open(self, session_id: str, access: Any) -> SessionHandle:
            if session_id == "missing-inactive":
                raise SessionPersistenceNotFoundError(session_id)
            return super().open(session_id, access)

    p_win2 = _Win2Persist(JsonlSessionPersistence(root=sessions_dir))
    ctx_win2.provide("sessionPersistence", p_win2)
    ctx_win2.use(
        AgentLoop,
        {"agents": [{"id": "fail-inactive", "resumeSessionId": "cfg-no-route"}]},
    )

    async def delayed_boom_read() -> tuple[str, list[SessionEvent]]:
        started_cfg_open.set()
        await win2_disposing.wait()
        raise ValueError("late config fail")

    p_win2._on_read = lambda: delayed_boom_read()
    await ctx_win2.flush()
    loop_win2 = ctx_win2.agentLoop
    al_comp2 = ctx_win2.registry.get(AgentLoop)
    assert al_comp2 is not None
    al_fiber2 = al_comp2.fibers[0]
    al_fiber2.effect(lambda: pause_win2)
    await started_cfg_open.wait()

    # Start a _restore_or_create_configured task that enters _resume_with BEFORE al_fiber2.dispose(),
    # then becomes inactive inside read() before raising SessionPersistenceNotFoundError (hits 512)!
    async def notfound_after_inactive() -> tuple[str, list[SessionEvent]]:
        await win2_disposing.wait()
        raise SessionPersistenceNotFoundError("cfg-effort")

    p_win2._on_read = lambda: notfound_after_inactive()
    restore_512_task = asyncio.create_task(
        loop_win2._restore_or_create_configured(
            ctx_win2,
            p_win2,
            "cfg-effort",
            {},
            {},
        )
    )
    await asyncio.sleep(0)
    d2 = asyncio.create_task(al_fiber2.dispose())
    await win2_disposing.wait()
    await restore_512_task
    win2_release.set()
    await d2
    await ctx_win2.fiber.dispose()
    assert win2_failures == []

    # 4. Abort during interrupted-turn closer append in resume -> hits 841 (signal.aborted inside init),
    #    892 (unsub_caller), and 897-898 (stored.handle.close failure warning on init abort)!
    seed_int_ctx = await _boot_context(sessions_dir=sessions_dir)
    h_int = await seed_int_ctx.agents.create(
        {"sessionId": "int-abort", "agentOptions": {"provider": "ok", "model": "m"}}
    )
    h_int["agent"].session.append("turn/start", {"turn": 1})
    h_int["agent"].session.append("step/start", {"turn": 1, "step": 1})
    await seed_int_ctx.sessions.flush(h_int["agent"].session)
    await h_int["dispose"]()
    await seed_int_ctx.fiber.dispose()

    ctx_sig_init = Context()
    ctx_sig_init.use(SystemPromptService, {"include_identity": False})
    ctx_sig_init.use(ToolService)
    ctx_sig_init.use(apply_native_executor)
    ctx_sig_init.use(LLMService)
    ctx_sig_init.use(SessionProjectionRegistry)
    ctx_sig_init.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    sig_in_append = AbortSignal()

    def abort_on_closer_append(_evs: list[SessionEvent]) -> None:
        sig_in_append.abort("aborted-during-closers")

    ctx_sig_init.provide(
        "sessionPersistence",
        _DelegatingPersistence(
            JsonlSessionPersistence(root=sessions_dir),
            fail_close_on_open=True,
            on_append=abort_on_closer_append,
        ),
    )
    ctx_sig_init.use(AgentRegistry)
    ctx_sig_init.use(AgentLoop)
    await ctx_sig_init.flush()
    # Initialize fails with primary abort error; close() error is logged as warning without masking.
    with pytest.raises(RuntimeError, match='agent "int-abort" creation aborted'):
        await ctx_sig_init.agents.resume(
            {
                "resumeSessionId": "int-abort",
                "signal": sig_in_append,
                "agentOptions": {"provider": "ok", "model": "m"},
            }
        )
    await ctx_sig_init.fiber.dispose()

    # 5. Concurrent dispose while _publish awaits setup (hits 788, 906) + rollback dispose failure (956-957)
    ctx_pub = Context()
    ctx_pub.use(SystemPromptService, {"include_identity": False})
    ctx_pub.use(ToolService)
    ctx_pub.use(apply_native_executor)
    ctx_pub.use(LLMService)
    ctx_pub.use(SessionProjectionRegistry)
    ctx_pub.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx_pub.provide(
        "sessionPersistence",
        _DelegatingPersistence(
            JsonlSessionPersistence(root=sessions_dir),
            fail_close_on_create=True,
        ),
    )
    ctx_pub.use(AgentRegistry)
    ctx_pub.use(AgentLoop)
    await ctx_pub.flush()
    setup_started = asyncio.Event()
    setup_sig = AbortSignal()

    async def slow_pub_setup(_actx: Context, _ag: Any) -> None:
        setup_started.set()
        await asyncio.sleep(0.05)

    create_pub = asyncio.create_task(
        ctx_pub.agents.create(
            {
                "sessionId": "pub-race",
                "signal": setup_sig,
                "agentOptions": {"provider": "ok", "model": "m"},
                "setup": slow_pub_setup,
            }
        )
    )
    await setup_started.wait()
    # Dispose factory while _publish is inside slow_pub_setup -> hits publication_done await (788)
    # AND rollback dispose failure (956-957)! (animas catches & logs effect unload failures)
    await ctx_pub.fiber.dispose()
    with pytest.raises(RuntimeError):
        await create_pub

    # 6. _publish _assert_live failure inside _publish when signal aborts inside commit (906),
    #    plus _append_unstored_suffix failure BEFORE sessions.enter (800->805, 812->815)!
    ctx_live_assert = Context()
    ctx_live_assert.use(SystemPromptService, {"include_identity": False})
    ctx_live_assert.use(ToolService)
    ctx_live_assert.use(apply_native_executor)
    ctx_live_assert.use(LLMService)
    ctx_live_assert.use(SessionProjectionRegistry)
    ctx_live_assert.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    p_suffix_fail = _DelegatingPersistence(JsonlSessionPersistence(root=sessions_dir))
    ctx_live_assert.provide("sessionPersistence", p_suffix_fail)
    ctx_live_assert.use(AgentRegistry)
    ctx_live_assert.use(AgentLoop)
    await ctx_live_assert.flush()

    sig_post_setup = AbortSignal()

    def abort_post_setup(_actx: Context, _ag: Any) -> dict[str, Any]:
        return {"commit": lambda: sig_post_setup.abort("post-setup")}

    with pytest.raises(RuntimeError, match='agent "post-setup-abort" creation aborted'):
        await ctx_live_assert.agents.create(
            {
                "sessionId": "post-setup-abort",
                "signal": sig_post_setup,
                "agentOptions": {"provider": "ok", "model": "m"},
                "setup": abort_post_setup,
            }
        )

    p_suffix_fail._on_append = lambda _evs: (_ for _ in ()).throw(
        OSError("suffix append boom")
    )
    with pytest.raises(OSError, match="suffix append boom"):
        await ctx_live_assert.agents.create(
            {
                "sessionId": "suffix-fail-rollback",
                "meta": {"cwd": str(tmp_path), "isSeeded": True},
                "inheritedEventCount": 1,
                "seed": [
                    SessionEvent(
                        id="sf-ev-0",
                        session_id="suffix-fail-rollback",
                        type="user/message",
                        seq=0,
                        timestamp=1700000000.0,
                        surface_op="append",
                        payload={
                            "message": create_user_message(
                                content=[{"type": "text", "text": "s"}]
                            )
                        },
                    ),
                    SessionEvent(
                        id="sf-ev-1",
                        session_id="suffix-fail-rollback",
                        type="session/end-seed",
                        seq=1,
                        timestamp=1700000001.0,
                        payload={"inherited": True},
                    ),
                ],
                "agentOptions": {"provider": "ok", "model": "m"},
            }
        )
    await ctx_live_assert.fiber.dispose()

    # 7. _teardown_inner collecting MULTIPLE errors (792-793 cancel/when_idle, 797-798 scope.dispose,
    #    803-804 sessions.flush, 807-808 handle.close, 817 ExceptionGroup)
    ctx_agg = Context()
    ctx_agg.use(SystemPromptService, {"include_identity": False})
    ctx_agg.use(ToolService)
    ctx_agg.use(apply_native_executor)
    ctx_agg.use(LLMService)
    ctx_agg.use(SessionProjectionRegistry)
    ctx_agg.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx_agg.provide(
        "sessionPersistence",
        _DelegatingPersistence(
            JsonlSessionPersistence(root=sessions_dir),
            fail_close_on_create=True,
        ),
    )
    ctx_agg.use(AgentRegistry)
    ctx_agg.use(AgentLoop)
    await ctx_agg.flush()
    ctx_agg.llm.register_adapter(["ok"], _OkAdapter())
    # Force sessions.flush to raise during teardown (hits 803-804)
    ctx_agg.on(
        "session/flush",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("flush hook boom")),
    )

    def attach_scope_fail(agent_ctx: Context, _ag: Any) -> None:
        async def bad_scope_dispose() -> None:
            raise RuntimeError("scope dispose boom")

        agent_ctx.fiber.effect(lambda: bad_scope_dispose)

    h_agg = await ctx_agg.agents.create(
        {
            "sessionId": "teardown-multi",
            "agentOptions": {"provider": "ok", "model": "m"},
            "setup": attach_scope_fail,
        }
    )
    inertia_err_fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    inertia_err_fut.set_exception(RuntimeError("scope inertia boom"))
    h_agg["agent"].scope._fiber.inertia = inertia_err_fut
    # Also attach a failing listener on a running turn signal inside run_maintenance to hit 792-793!
    maint_entered = asyncio.Event()
    maint_release = asyncio.Event()

    async def holding_maint(sig: AbortSignal) -> None:
        sig.on_abort(lambda: (_ for _ in ()).throw(RuntimeError("cancel boom")))
        maint_entered.set()
        await maint_release.wait()

    maint_task = asyncio.create_task(h_agg["agent"].run_maintenance(holding_maint))
    await maint_entered.wait()
    with pytest.raises(ExceptionGroup) as eg_teardown:
        await h_agg["dispose"]()
    maint_release.set()
    await maint_task
    assert len(eg_teardown.value.exceptions) == 4
    await ctx_agg.fiber.dispose()

    # 8. Post-abort create persistence resolution releases abandoned handle
    open_entered = asyncio.Event()
    open_release = asyncio.Event()
    closed_abandoned: list[str] = []

    class _SlowCreatePersistence(_DelegatingPersistence):
        async def create(  # type: ignore[override]
            self,
            header: dict[str, Any],
            inherited_event_count: int | None = None,
        ) -> SessionHandle:
            handle = self._inner.create(header, inherited_event_count)
            open_entered.set()
            await open_release.wait()
            wrapped = _FaultyHandle(handle)
            orig_close = wrapped.close

            def track_close() -> None:
                closed_abandoned.append(str(header.get("id", "")))
                orig_close()

            wrapped.close = track_close  # type: ignore[method-assign]
            return wrapped

    ctx_abandon = Context()
    ctx_abandon.use(SystemPromptService, {"include_identity": False})
    ctx_abandon.use(ToolService)
    ctx_abandon.use(apply_native_executor)
    ctx_abandon.use(LLMService)
    ctx_abandon.use(SessionProjectionRegistry)
    ctx_abandon.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx_abandon.provide(
        "sessionPersistence",
        _SlowCreatePersistence(JsonlSessionPersistence(root=sessions_dir)),
    )
    ctx_abandon.use(AgentRegistry)
    ctx_abandon.use(AgentLoop)
    await ctx_abandon.flush()

    abort_sig = AbortSignal()
    create_abandoned = asyncio.create_task(
        ctx_abandon.agents.create(
            {
                "sessionId": "abandoned-create",
                "signal": abort_sig,
                "agentOptions": {"provider": "ok", "model": "m"},
            }
        )
    )
    await open_entered.wait()
    abort_sig.abort("stop-create")
    with pytest.raises(RuntimeError, match="abandoned-create"):
        await create_abandoned
    open_release.set()
    await asyncio.sleep(0.02)
    assert closed_abandoned == ["abandoned-create"]
    await ctx_abandon.fiber.dispose()

    # 9. FactoryOwnership.track_wrapper awaits in-flight failing create without failing dispose
    wrap_entered = asyncio.Event()
    wrap_release = asyncio.Event()

    class _DelayedFailPersistence(_DelegatingPersistence):
        async def create(  # type: ignore[override]
            self,
            header: dict[str, Any],
            inherited_event_count: int | None = None,
        ) -> SessionHandle:
            del header, inherited_event_count
            wrap_entered.set()
            await wrap_release.wait()
            raise OSError("late create boom")

    ctx_wrap = Context()
    ctx_wrap.use(SystemPromptService, {"include_identity": False})
    ctx_wrap.use(ToolService)
    ctx_wrap.use(apply_native_executor)
    ctx_wrap.use(LLMService)
    ctx_wrap.use(SessionProjectionRegistry)
    ctx_wrap.use(SessionService, {"base_storage_dir": str(sessions_dir)})
    ctx_wrap.provide(
        "sessionPersistence",
        _DelayedFailPersistence(JsonlSessionPersistence(root=sessions_dir)),
    )
    ctx_wrap.use(AgentRegistry)
    ctx_wrap.use(AgentLoop)
    await ctx_wrap.flush()

    create_wrap_task = asyncio.create_task(
        ctx_wrap.agents.create(
            {
                "sessionId": "wrap-create-fail",
                "agentOptions": {"provider": "ok", "model": "m"},
            }
        )
    )
    await wrap_entered.wait()
    dispose_wrap_task = asyncio.create_task(ctx_wrap.fiber.dispose())
    await asyncio.sleep(0.02)
    assert not dispose_wrap_task.done()
    wrap_release.set()
    await dispose_wrap_task
    with pytest.raises(OSError, match="late create boom"):
        await create_wrap_task
