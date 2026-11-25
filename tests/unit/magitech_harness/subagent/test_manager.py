"""Unit tests for SubagentManager orchestration, workspace isolation, and lifecycles."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from animas import Context

from magitech_harness import create_harness as real_create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.service import LLMService, MockLLMProvider
from magitech_harness.session.events import EventType
from magitech_harness.session.service import SessionService
from magitech_harness.subagent.manager import SubagentManager
from magitech_harness.subagent.service import SubagentService
from magitech_harness.subagent.types import SubagentSpec, SubagentState
from magitech_harness.tools.schema import define_tool
from magitech_harness.tools.service import ToolService


class MockTurn:
    """Scripted turn outcome for the unified child agent driver."""

    def __init__(self, return_value: str = "Turn result") -> None:
        self.return_value = return_value
        self.raise_exc: Exception | None = None


class _MockChildAgent:
    def __init__(self, ctx: Any, session: Any, turn: MockTurn) -> None:
        self.session = session
        self._turn = turn
        self.model = SimpleNamespace(current=lambda: None, select=lambda _sel: None)

    def followup(self, message: Any) -> None:
        pass

    async def when_idle(self) -> None:
        if self._turn.raise_exc:
            raise self._turn.raise_exc
        # Append turn/start + assistant message so the outcome text is deterministic.
        self.session.append(EventType.TURN_START, {"turn": 1})
        self.session.append(
            EventType.ASSISTANT_MESSAGE,
            {
                "turn": 1,
                "step": 1,
                "stream": [],
                "message": {
                    "id": "mock-assistant",
                    "role": "assistant",
                    "source": {"kind": "model", "provider": "mock", "model": "mock"},
                    "content": [{"type": "text", "text": self._turn.return_value}],
                },
            },
            surface_op="append",
        )


def _bundle_child_factory(
    return_value: str = "Turn result",
    raise_exc: Exception | None = None,
    captured: list[Context] | None = None,
    turns: list[MockTurn] | None = None,
) -> Callable[..., Context]:
    """Replacement for manager.create_harness: real bundle ctx, mock loop.

    The child context is built by the real bundle loader, exactly as in
    production; only the agent loop is swapped for a MockLoop so the
    turn result is deterministic.
    """

    def _factory(storage_dir: Path | str | None = None, **kwargs: Any) -> Context:
        ctx = real_create_harness(storage_dir=storage_dir)
        turn = MockTurn(return_value)
        turn.raise_exc = raise_exc
        if captured is not None:
            captured.append(ctx)
        if turns is not None:
            turns.append(turn)

        # Unique per child context: both invokes share subsessions_dir, so a
        # fixed id lets the second invoke load the first one's persisted
        # session and shift first_seq — a Windows-only persistence-timing race.
        default_session_id = f"mock-child-{uuid.uuid4().hex[:8]}"

        async def _fake_create(options: Any) -> Any:
            session_id = str(options.get("sessionId", default_session_id))
            session = ctx.sessions.get(session_id)
            if session is None:
                # The driver prepares a fresh session after the pre-created
                # one was closed; resume it from persistence if available.
                session = ctx.sessions.load(session_id)
            if session is None:
                session = ctx.sessions.create(session_id)
            agent = _MockChildAgent(ctx, session, turn)
            return {"agent": agent, "dispose": lambda: None}

        # Patch after the bundle settles: ``agents`` only exists post-flush,
        # so mount a tiny component whose apply() runs during flush.
        class _PatchAgents:
            inject: ClassVar[list[str]] = ["agents"]

            def __init__(self, c: Context) -> None:
                c.agents.create = _fake_create  # type: ignore[method-assign]
                c.agents.resume = _fake_create  # type: ignore[method-assign]

        ctx.use(_PatchAgents)
        return ctx

    return _factory


@pytest.fixture
def parent_ctx(tmp_path: Path) -> Context:
    ctx = Context()
    ctx.workspace_dir = tmp_path
    SessionService(ctx, {"base_storage_dir": tmp_path / "sessions"})
    tools = ToolService(ctx)
    tools.register(
        define_tool(
            name="read_file", description="", parameters={}, execute=AsyncMock()
        )
    )
    tools.register(
        define_tool(
            name="write_file", description="", parameters={}, execute=AsyncMock()
        )
    )
    tools.register(
        define_tool(
            name="invoke_subagent", description="", parameters={}, execute=AsyncMock()
        )
    )
    llm = LLMService(ctx)
    llm.default_model = "test-model"
    return ctx


@pytest.mark.asyncio
async def test_subagent_manager_invoke_sync_and_presets(
    parent_ctx: Context, tmp_path: Path
) -> None:
    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})

    # 1. Research preset: automatically restricts to research tools
    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_bundle_child_factory("Research completed"),
    ):
        specs = [
            SubagentSpec(
                role="Codebase Researcher",
                prompt="Audit files",
                wait=True,
            )
        ]
        records = await manager.invoke(specs)
        assert len(records) == 1
        rec = records[0]
        assert rec.role == "Codebase Researcher"
        assert rec.state == SubagentState.COMPLETED
        assert rec.summary == "Research completed"
        assert "read_file" in rec.effective_tools
        assert "write_file" not in rec.effective_tools

    # 2. General role with custom allowed_tools
    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_bundle_child_factory("Custom completed"),
    ):
        specs2 = [
            SubagentSpec(
                role="worker",
                prompt="Do work",
                model="custom-model",
                allowed_tools=frozenset({"write_file"}),
                wait=True,
            )
        ]
        records2 = await manager.invoke(specs2)
        rec2 = records2[0]
        assert rec2.model_id == "custom-model"
        assert rec2.effective_tools == ["write_file"]
        assert rec2.summary == "Custom completed"


@pytest.mark.asyncio
async def test_subagent_manager_invoke_background_and_send_message(
    parent_ctx: Context, tmp_path: Path
) -> None:
    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})

    turn_instances: list[MockTurn] = []

    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_bundle_child_factory("Turn 1 done", turns=turn_instances),
    ):
        # Background execution (wait=False)
        spec = SubagentSpec(role="assistant", prompt="Background work", wait=False)
        records = await manager.invoke([spec])
        rec = records[0]
        assert rec.subagent_id in manager._tasks

        # Await task to finish
        await manager._tasks[rec.subagent_id]
        assert rec.state == SubagentState.COMPLETED

        # Send follow-up message
        turn_instances[0].return_value = "Follow up done"
        rec_after = await manager.send_message(rec.subagent_id, "Next instruction")
        assert rec_after.summary == "Follow up done"


@pytest.mark.asyncio
async def test_subagent_manager_branch_workspace_git(
    parent_ctx: Context, tmp_path: Path
) -> None:
    (tmp_path / ".git").mkdir()
    manager = SubagentManager(
        parent_ctx,
        {
            "subsessions_dir": tmp_path / "subsessions",
            "workspace_root": tmp_path,
        },
    )

    # 1. Successful worktree creation
    mock_proc_ok = MagicMock()
    mock_proc_ok.returncode = 0
    mock_proc_ok.communicate = AsyncMock(return_value=(b"", b""))

    with (
        patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc_ok),
        ),
        patch(
            "magitech_harness.subagent.manager.create_harness",
            side_effect=_bundle_child_factory("Worktree turn done"),
        ),
    ):
        spec = SubagentSpec(
            role="branch-worker",
            prompt="Work on branch",
            workspace="branch",
            wait=True,
        )
        records = await manager.invoke([spec])
        rec = records[0]
        assert "worktrees" in str(rec.workspace_path)
        assert rec.subagent_id in manager._worktrees

        # Test manage kill cleans up worktree
        mock_rm_proc = MagicMock()
        mock_rm_proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_rm_proc),
        ):
            killed = await manager.manage("kill", [rec.subagent_id])
            assert len(killed) == 1
            assert killed[0].state == SubagentState.TERMINATED
            assert rec.subagent_id not in manager._worktrees

    # 2. Branch workspace when not a git repo -> falls back to workspace_root
    non_git_ws = tmp_path / "nongit"
    non_git_ws.mkdir()
    manager_nongit = SubagentManager(
        parent_ctx,
        {
            "subsessions_dir": tmp_path / "subsessions",
            "workspace_root": non_git_ws,
        },
    )
    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_bundle_child_factory("Done"),
    ):
        spec2 = SubagentSpec(role="w", prompt="", workspace="branch", wait=True)
        recs2 = await manager_nongit.invoke([spec2])
        assert recs2[0].workspace_path == non_git_ws

    # 3. Branch workspace when git command raises OSError
    with (
        patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("git not found"),
        ),
        patch(
            "magitech_harness.subagent.manager.create_harness",
            side_effect=_bundle_child_factory("Done"),
        ),
    ):
        recs3 = await manager.invoke(
            [SubagentSpec(role="w", prompt="", workspace="branch", wait=True)]
        )
        assert recs3[0].workspace_path == tmp_path

    # 4. Branch workspace when git returncode != 0
    mock_proc_fail = MagicMock()
    mock_proc_fail.returncode = 1
    mock_proc_fail.communicate = AsyncMock(return_value=(b"", b"fatal: git error"))
    with (
        patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc_fail),
        ),
        patch(
            "magitech_harness.subagent.manager.create_harness",
            side_effect=_bundle_child_factory("Done"),
        ),
    ):
        recs4 = await manager.invoke(
            [SubagentSpec(role="w", prompt="", workspace="branch", wait=True)]
        )
        assert recs4[0].workspace_path == tmp_path


@pytest.mark.asyncio
async def test_subagent_manager_manage_and_errors(
    parent_ctx: Context, tmp_path: Path
) -> None:
    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})

    # 1. Send message to nonexistent subagent
    with pytest.raises(ValueError, match="does not exist"):
        await manager.send_message("nonexistent", "hello")

    # 2. Turn execution failure in child loop
    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_bundle_child_factory(raise_exc=RuntimeError("Execution crashed")),
    ):
        spec = SubagentSpec(role="failing", prompt="crash me", wait=True)
        recs = await manager.invoke([spec])
        assert recs[0].state == SubagentState.ERRORED
        assert "Execution crashed" in (recs[0].error or "")

    # 3. Session missing error in child context. The store is the source
    # of truth now, so removing the session must go through the public
    # delete path: clearing the service cache alone no longer hides it.
    # Delete refuses a live write handle, so the session is closed first.
    sid = recs[0].subagent_id
    child_ctx = manager._child_contexts[sid]
    child_ctx.sessions.close(recs[0].session_id)
    assert child_ctx.sessions.delete(recs[0].session_id) is True
    await manager._run_child(sid, "another prompt")
    assert recs[0].state == SubagentState.ERRORED
    assert "not found in child context" in (recs[0].error or "")

    # 4. Manage list and kill_all
    active = await manager.manage("list")
    assert len(active) >= 1

    # Kill with non-existent sid in list
    res_kill = await manager.manage("kill", ["nonexistent_id", sid])
    assert len(res_kill) == 1

    all_killed = await manager.manage("kill_all")
    assert all(r.state == SubagentState.TERMINATED for r in all_killed)

    # 5. Send message to terminated subagent
    with pytest.raises(ValueError, match="terminated"):
        await manager.send_message(sid, "hello")

    # 6. Unknown manage action
    with pytest.raises(ValueError, match="Unknown manage_subagents action"):
        await manager.manage("invalid_action")

    # 7. Shutdown calls kill_all
    await manager.shutdown()


@pytest.mark.asyncio
async def test_subagent_cancellation_cascade(
    parent_ctx: Context, tmp_path: Path
) -> None:
    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})

    parent_signal = AbortSignal()
    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_bundle_child_factory("Done"),
    ):
        recs = await manager.invoke(
            [SubagentSpec(role="w", prompt="test", wait=True)],
            parent_signal=parent_signal,
        )
        child_signal = manager._signals[recs[0].subagent_id]
        assert not child_signal.is_aborted

        parent_signal.abort("Parent aborted")
        assert child_signal.is_aborted
        assert child_signal.reason == "Parent aborted"


@pytest.mark.asyncio
async def test_subagent_empty_context_and_worktree_cleanup_failure(
    tmp_path: Path,
) -> None:
    empty_ctx = Context()
    manager = SubagentManager(empty_ctx, {"subsessions_dir": tmp_path / "subsessions"})

    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_bundle_child_factory("Done"),
    ):
        recs = await manager.invoke(
            [SubagentSpec(role="no_tools", prompt="", wait=True)]
        )
        assert recs[0].effective_tools == []

    # Test worktree removal failure in _terminate_child
    sid = recs[0].subagent_id
    manager._worktrees[sid] = tmp_path / "fake_wt"
    with patch("asyncio.create_subprocess_exec", side_effect=OSError("rm failed")):
        await manager._terminate_child(sid)
        assert sid not in manager._worktrees


@pytest.mark.asyncio
async def test_subagent_terminate_running_task_and_missing_record(
    parent_ctx: Context, tmp_path: Path
) -> None:
    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})

    # Background task that sleeps
    async def _long_turn(*args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(10)

    task = asyncio.create_task(_long_turn())
    manager._tasks["sub-sleep"] = task

    await manager._terminate_child("sub-sleep")
    assert task.cancelled()

    # Already completed task
    async def _instant() -> None:
        pass

    done_task = asyncio.create_task(_instant())
    await done_task
    manager._tasks["sub-done"] = done_task
    await manager._terminate_child("sub-done")

    # Terminate child where sid has no record and no signal
    await manager._terminate_child("nonexistent")


@pytest.mark.asyncio
async def test_child_ctx_bundle_built_copies_providers_and_persists(
    parent_ctx: Context, tmp_path: Path
) -> None:
    """Child ctx comes from the bundle and inherits providers and tools."""
    mock_provider = MockLLMProvider()
    alt_provider = MockLLMProvider()
    parent_ctx.llm.register_provider("mock", mock_provider)
    parent_ctx.llm.register_provider("alt", alt_provider)
    parent_ctx.llm.set_active_provider("alt")
    # A custom parent tool must be copied where the child bundle does not
    # already mount that name; this covers both the copy and the skip.
    parent_ctx.tools.register(
        define_tool(
            name="probe_custom",
            description="custom",
            parameters={},
            execute=lambda _a: "ok",
        )
    )

    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})
    captured: list[Context] = []
    base_factory = _bundle_child_factory(captured=captured)

    def _factory_with_preloaded(
        storage_dir: Path | str | None = None, **kwargs: Any
    ) -> Context:
        # Pre-load one parent-named tool so the copy loop also covers the
        # "child already mounts this name" path on every harness release.
        ctx = base_factory(storage_dir=storage_dir, **kwargs)

        class _PreloadCustom:
            inject: ClassVar[list[str]] = ["tools"]

            def __init__(self, c: Context) -> None:
                c.tools.register(
                    define_tool(
                        name="probe_custom",
                        description="custom",
                        parameters={},
                        execute=lambda _a: "ok",
                    )
                )

        ctx.use(_PreloadCustom)
        return ctx

    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_factory_with_preloaded,
    ):
        recs = await manager.invoke(
            [
                SubagentSpec(
                    role="worker",
                    prompt="Do work",
                    allowed_tools=frozenset({"read_file", "probe_custom"}),
                    wait=True,
                )
            ]
        )
    assert len(recs) == 1
    rec = recs[0]
    assert rec.state == SubagentState.COMPLETED
    assert rec.summary == "Turn result"
    assert rec.effective_tools == ["read_file", "probe_custom"]

    assert len(captured) == 1
    child_ctx = captured[0]

    # Providers copied through the public seams; active provider preserved.
    assert set(child_ctx.llm.providers) == {"mock", "alt"}
    assert child_ctx.llm.active_provider is alt_provider

    # The bundle mounts the base tool set; the allowed set filters copied
    # parent tools and restricts the child's own scope, and the delegation
    # triad never reaches a child.
    assert child_ctx.tools.get("read_file") is not None
    assert "write_file" not in rec.effective_tools
    assert child_ctx.tools.get("invoke_subagent") is None

    # The bundle-built child carries rows the old child never had.
    assert hasattr(child_ctx, "sessionTitle")
    assert hasattr(child_ctx, "sessionProjections")
    assert hasattr(child_ctx, "sessionProjectionCache")

    # The child session persists under the subsessions dir.
    await child_ctx.sessions.flush()
    assert rec.session_path.exists()
    assert tmp_path / "subsessions" in rec.session_path.parents


@pytest.mark.asyncio
async def test_child_with_parent_llm_without_providers(
    parent_ctx: Context, tmp_path: Path
) -> None:
    """A parent LLM with no providers yields a child LLM with none either."""
    assert parent_ctx.llm.providers == {}

    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})
    captured: list[Context] = []
    with patch(
        "magitech_harness.subagent.manager.create_harness",
        side_effect=_bundle_child_factory(captured=captured),
    ):
        recs = await manager.invoke(
            [SubagentSpec(role="worker", prompt="Do work", wait=True)]
        )
    assert recs[0].state == SubagentState.COMPLETED
    assert len(captured) == 1
    assert captured[0].llm.providers == {}
    assert captured[0].llm.active_provider is None


@pytest.mark.asyncio
async def test_child_agent_defaults_when_no_providers_and_with_signal(
    parent_ctx: Context, tmp_path: Path
) -> None:
    """Covers provider fallback and signal forwarding on the unified driver."""

    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})
    captured: list[Context] = []

    def _factory(storage_dir: Path | str | None = None, **kwargs: Any) -> Context:
        ctx = real_create_harness(storage_dir=storage_dir)
        captured.append(ctx)
        turn = MockTurn("ok")

        async def _fake_create(options: Any) -> Any:
            session = ctx.sessions.create(str(options.get("sessionId", "child")))
            agent = _MockChildAgent(ctx, session, turn)
            return {"agent": agent, "dispose": lambda: None}

        class _NoAdapterPatch:
            inject: ClassVar[list[str]] = ["agents"]

            def __init__(self, c: Context) -> None:
                c.agents.create = _fake_create  # type: ignore[method-assign]
                c.agents.resume = _fake_create  # type: ignore[method-assign]

        ctx.use(_NoAdapterPatch)
        return ctx

    signal = AbortSignal()
    with patch(
        "magitech_harness.subagent.manager.create_harness", side_effect=_factory
    ):
        recs = await manager.invoke(
            [SubagentSpec(role="worker", prompt="Do work", wait=True)],
            parent_signal=signal,
        )
    assert recs[0].state == SubagentState.COMPLETED
    assert recs[0].summary == "ok"


@pytest.mark.asyncio
async def test_child_agent_with_model_slug(parent_ctx: Context, tmp_path: Path) -> None:
    """Covers the model slug (provider/model) branch on the unified driver."""
    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})

    def _factory(storage_dir: Path | str | None = None, **kwargs: Any) -> Context:
        ctx = real_create_harness(storage_dir=storage_dir)
        turn = MockTurn("ok")

        async def _fake_create(options: Any) -> Any:
            session = ctx.sessions.create(str(options.get("sessionId", "child")))
            agent = _MockChildAgent(ctx, session, turn)
            return {"agent": agent, "dispose": lambda: None}

        class _Patch:
            inject: ClassVar[list[str]] = ["agents"]

            def __init__(self, c: Context) -> None:
                c.agents.create = _fake_create  # type: ignore[method-assign]
                c.agents.resume = _fake_create  # type: ignore[method-assign]

        ctx.use(_Patch)
        return ctx

    with patch(
        "magitech_harness.subagent.manager.create_harness", side_effect=_factory
    ):
        recs = await manager.invoke(
            [
                SubagentSpec(
                    role="worker",
                    prompt="Do work",
                    model="deepseek/deepseek-chat",
                    wait=True,
                )
            ]
        )
    assert recs[0].state == SubagentState.COMPLETED


@pytest.mark.asyncio
async def test_allowed_set_with_no_mounted_names_denies_essentially_everything(
    parent_ctx: Context, tmp_path: Path
) -> None:
    """An unsatisfiable allowed set fails closed, not open.

    The child bundle mounts base tools, so skipping the restriction when the
    intersection is empty would hand the child every bundle tool, shell
    included. The empty allow set must deny instead.
    """
    manager = SubagentManager(parent_ctx, {"subsessions_dir": tmp_path / "subsessions"})
    agents: list[Any] = []

    def _factory(storage_dir: Path | str | None = None, **kwargs: Any) -> Context:
        ctx = real_create_harness(storage_dir=storage_dir)
        turn = MockTurn("ok")

        class _PreloadCustom:
            inject: ClassVar[list[str]] = ["tools"]

            def __init__(self, c: Context) -> None:
                c.tools.register(
                    define_tool(
                        name="probe_custom",
                        description="custom",
                        parameters={},
                        execute=lambda _a: "ok",
                    )
                )

        async def _fake_create(options: Any) -> Any:
            session = ctx.sessions.create(str(options.get("sessionId", "child")))
            agent = _MockChildAgent(ctx, session, turn)
            agents.append(agent)
            return {"agent": agent, "dispose": lambda: None}

        class _NoAdapterPatch:
            inject: ClassVar[list[str]] = ["agents"]

            def __init__(self, c: Context) -> None:
                c.agents.create = _fake_create  # type: ignore[method-assign]
                c.agents.resume = _fake_create  # type: ignore[method-assign]

        ctx.use(_PreloadCustom)
        ctx.use(_NoAdapterPatch)
        return ctx

    with patch(
        "magitech_harness.subagent.manager.create_harness", side_effect=_factory
    ):
        recs = await manager.invoke(
            [
                SubagentSpec(
                    role="worker",
                    prompt="Do work",
                    allowed_tools=frozenset({"absent_tool"}),
                    wait=True,
                )
            ]
        )
    assert recs[0].state == SubagentState.COMPLETED
    assert recs[0].effective_tools == []
    assert len(agents) == 1
    agent = agents[0]
    child_ctx = manager._child_contexts[recs[0].subagent_id]
    # Nothing is admitted through the child agent's scope, including a tool the
    # child mounts, and including the base shell when the harness ships it.
    assert child_ctx.tools.get("probe_custom", scope=agent) is None
    if child_ctx.tools.get("bash") is not None:
        assert child_ctx.tools.get("bash", scope=agent) is None


def test_mount_shape_and_tool_names(tmp_path: Any) -> None:
    from magitech_harness import create_harness

    ctx = create_harness(storage_dir=tmp_path / "mount")
    assert isinstance(ctx.subagents, SubagentService)
    assert sorted(provider.name for provider in ctx.subagents._providers.values()) == [
        "fork",
        "spawn",
    ]
    names = {definition["name"] for definition in ctx.tools.definitions()}
    assert {
        "subagent",
        "subagent_fork",
        "send_message",
        "list_agents",
        "interrupt_agent",
        "wait_agent",
    } <= names


@pytest.mark.asyncio
async def test_subagent_tools_registered_and_callable(tmp_path: Any) -> None:
    from magitech_harness import create_harness
    from magitech_harness.agent.cancellation import AbortSignal

    ctx = create_harness(storage_dir=tmp_path / "tools")
    await ctx.flush()
    session = ctx.sessions.create("subagent-tools")
    agent = type("Agent", (), {"id": "root", "session": session})()

    async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        batch = await ctx.tools.execute(
            {
                "calls": [
                    {
                        "type": "tool-call",
                        "id": name,
                        "name": name,
                        "arguments": arguments,
                    }
                ],
                "signal": AbortSignal(),
                "agent": agent,
            }
        )
        return batch["results"][0]["result"]

    result = await call("list_agents", {})
    assert result["isError"] is False
    assert result["value"]["agents"][0]["name"] == "root"
    assert result["value"]["running"] == 0

    result = await call("interrupt_agent", {"agent": "missing"})
    assert result["isError"] is True

    result = await call("send_message", {"agent": "missing", "message": "hi"})
    assert result["isError"] is True

    result = await call("wait_agent", {"timeout_ms": 999})
    assert result["isError"] is True


@pytest.mark.asyncio
async def test_delegate_tool_starts_child(tmp_path: Any) -> None:
    from magitech_harness import create_harness
    from magitech_harness.agent.cancellation import AbortSignal
    from magitech_harness.llm.service import MockLLMProvider

    ctx = create_harness(storage_dir=tmp_path / "delegate")
    await ctx.flush()
    ctx.llm.register_provider("mock", MockLLMProvider())
    ctx.llm.set_active_provider("mock")
    session = ctx.sessions.create("delegate-session")
    agent = type("Agent", (), {"id": "root", "session": session})()
    batch = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "d1",
                    "name": "subagent",
                    "arguments": {
                        "description": "researcher",
                        "prompt": "Do the work",
                        "name": "researcher",
                    },
                }
            ],
            "signal": AbortSignal(),
            "agent": agent,
        }
    )
    result = batch["results"][0]["result"]
    assert result["isError"] is False
    assert result["value"]["kind"] == "activation"
    assert result["value"]["name"] == "researcher"


def test_service_guards(tmp_path: Any) -> None:
    from magitech_harness.subagent.service import (
        ROOT_AGENT_NAME,
        SubagentError,
        SubagentProvider,
    )

    ctx = Context()
    LLMService(ctx)
    service = SubagentService(ctx)
    with pytest.raises(SubagentError, match="no subagent provider"):
        asyncio.run(
            service.startActivation(
                SubagentSpec(role="worker", prompt="go"), provider="missing"
            )
        )
    assert service.resolve_max_depth("provider-managed") is None
    with pytest.raises(SubagentError, match="must be non-negative"):
        service.resolve_max_depth(-1)
    dispose = service.register_provider(
        SubagentProvider(name="x", start=lambda _s, _c: _noop())
    )
    assert service.get_provider("x") is not None
    dispose()
    assert service.get_provider("x") is None
    assert service.capacity_of(None)["limit"] == service.max_active_subagents
    assert service.names_of(None)["self"] == ROOT_AGENT_NAME
    with pytest.raises(SubagentError, match="unknown subagent"):
        service._member("absent")


async def _noop() -> dict[str, Any]:
    return {}
