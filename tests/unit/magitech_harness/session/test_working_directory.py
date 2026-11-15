"""Unit tests for WorkingDirectoryService."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from anima import Context, EventMode, FiberState

from magitech_harness.agent.agent_loop import AgentLoop
from magitech_harness.agent.cancellation import AbortController
from magitech_harness.agent.registry import AgentRegistry
from magitech_harness.fs.local import LocalFileSystem
from magitech_harness.fs.types import FsError
from magitech_harness.llm.service import LLMService
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService
from magitech_harness.session.working_directory import WorkingDirectoryService
from magitech_harness.system_prompt.service import (
    SystemPromptService,
    render_context_snapshot,
)
from magitech_harness.tools.native_executor import apply as apply_native_executor
from magitech_harness.tools.service import ToolService


async def _mount_stack(ctx: Context, default_dir: Path | None = None) -> None:
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(ToolService)
    ctx.use(apply_native_executor)
    ctx.use(SystemPromptService)
    ctx.use(LocalFileSystem)
    ctx.use(AgentRegistry)
    ctx.use(AgentLoop, {"agents": []})
    cfg: dict[str, Any] = (
        {"default_directory": str(default_dir)} if default_dir is not None else {}
    )
    ctx.use(WorkingDirectoryService, cfg)
    await ctx.flush()


@pytest.mark.asyncio
async def test_working_directory_rejects_relative_default() -> None:
    """Fails loud when default_directory is not absolute."""
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SystemPromptService)
    ctx.use(LocalFileSystem)
    fiber = ctx.use(WorkingDirectoryService, {"default_directory": "relative/dir"})
    await ctx.flush()
    assert fiber.state == FiberState.FAILED
    with pytest.raises(ValueError, match="must be absolute"):
        WorkingDirectoryService(ctx, {"default_directory": "relative/dir"})
    for bad_cfg in (
        {"default_directory": None},
        {"defaultDirectory": ""},
        {"defaultDirectory": 123},
    ):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            WorkingDirectoryService(ctx, bad_cfg)


@pytest.mark.asyncio
async def test_working_directory_get_ensure_set_and_recovery(tmp_path: Path) -> None:
    """Manages session cwd, commits notices, recovers missing cwd, and updates prompt."""
    default_dir = tmp_path / "default"
    initial_dir = tmp_path / "project"
    sub_dir = initial_dir / "nested"
    default_dir.mkdir()
    initial_dir.mkdir()
    sub_dir.mkdir()

    ctx = Context()
    await _mount_stack(ctx, default_dir)

    handle = await ctx.agents.create(
        {"sessionId": "wd-1", "meta": {"cwd": str(initial_dir)}}
    )
    agent = handle["agent"]
    wd = ctx.workingDirectory

    assert wd.get(agent.session) == str(initial_dir)
    assert await wd.ensure(agent) == str(initial_dir)

    # Set to relative subdir commits event + notice
    changed = await wd.set(agent, "nested")
    assert changed == str(sub_dir.resolve())
    assert wd.get(agent.session) == changed
    assert any(
        ev.type == "working-directory/change" and ev.payload["cwd"] == changed
        for ev in agent.session.events
    )
    assert any(
        msg["source"] == {"kind": "working-directory"}
        and "changed from" in msg["content"][0]["text"]
        for msg in agent.inbox.next_step
    )

    # Same directory is a no-op
    before_count = len(agent.session.events)
    assert await wd.set(agent, ".") == changed
    assert len(agent.session.events) == before_count

    # Remove sub_dir -> ensure recovers back to initial_dir with notice
    sub_dir.rmdir()
    recovered = await wd.ensure(agent)
    assert recovered == str(initial_dir.resolve())
    assert any(
        msg["source"] == {"kind": "working-directory"}
        and "unavailable" in msg["content"][0]["text"]
        for msg in agent.inbox.next_step
    )

    # Prompt assembly ensures and includes Current working directory
    assembly = await ctx.systemPrompt.assemble({"agent": agent, "scope": agent})
    assert "Current working directory: " in render_context_snapshot(assembly)
    assert str(initial_dir.resolve()) in render_context_snapshot(assembly)

    # Empty or nonexistent path raises
    with pytest.raises(ValueError, match="must not be empty"):
        await wd.set(agent, "")
    with pytest.raises(RuntimeError, match="does not exist"):
        await wd.set(agent, "no-such-dir")

    # Cancellation during serialize
    ctrl = AbortController()
    ctrl.abort(RuntimeError("stop-wd"))
    with pytest.raises(RuntimeError, match="stop-wd"):
        await wd.ensure(agent, signal=ctrl.signal)

    await handle["dispose"]()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_working_directory_initial_unset_header_records_default(
    tmp_path: Path,
) -> None:
    """Commits default_directory on first ensure when header has no cwd."""
    default_dir = tmp_path / "fallback"
    default_dir.mkdir()
    ctx = Context()
    await _mount_stack(ctx, default_dir)

    handle = await ctx.agents.create({"sessionId": "wd-unset"})
    agent = handle["agent"]
    assert ctx.workingDirectory.get(agent.session) == str(default_dir)
    assert await ctx.workingDirectory.ensure(agent) == str(default_dir)
    assert any(
        ev.type == "working-directory/change" and ev.payload["cwd"] == str(default_dir)
        for ev in agent.session.events
    )
    # Prompt without agent returns empty context block
    empty_assembly = await ctx.systemPrompt.assemble({"scope": agent})
    assert "Current working directory" not in render_context_snapshot(empty_assembly)

    # Inject warning branch is non-fatal after committing change
    alt_dir = tmp_path / "alt"
    alt_dir.mkdir()
    orig_inject = agent.inject
    agent.inject = lambda _msg: (_ for _ in ()).throw(RuntimeError("inbox closed"))  # type: ignore[method-assign]
    assert await ctx.workingDirectory.set(agent, str(alt_dir)) == str(alt_dir)
    agent.inject = orig_inject  # type: ignore[method-assign]

    # Mid-flight abort during ensure, _resolve_directory, _commit, and teardown
    ctrl_mid = AbortController()
    orig_stat = ctx.fs.stat

    async def _abort_stat(target: Any, sig: Any = None) -> Any:
        res = await orig_stat(target, sig)
        ctrl_mid.abort(RuntimeError("mid-abort"))
        return res

    ctx.fs.stat = _abort_stat  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="mid-abort"):
        await ctx.workingDirectory.ensure(agent, signal=ctrl_mid.signal)

    ctrl_plain = AbortController()

    async def _abort_plain(target: Any, sig: Any = None) -> Any:
        res = await orig_stat(target, sig)
        ctrl_plain.abort("plain-reason")
        return res

    ctx.fs.stat = _abort_plain  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="working-directory operation aborted"):
        await ctx.workingDirectory.set(
            agent, str(default_dir), signal=ctrl_plain.signal
        )
    ctx.fs.stat = orig_stat  # type: ignore[method-assign]

    # _commit abort check directly
    ctrl_commit = AbortController()
    ctrl_commit.abort("commit-stop")
    with pytest.raises(RuntimeError, match="working-directory operation aborted"):
        ctx.workingDirectory._commit(agent, str(default_dir), False, ctrl_commit.signal)

    # Downstream assemble hook clearing contexts triggers re-insertion branch
    async def _drop_wd(assembly: Any, _context: Any, next_fn: Any) -> Any:
        res = await next_fn()
        res["contexts"] = [
            c for c in res["contexts"] if c["name"] != "working-directory:current"
        ]
        return res

    off_drop = ctx.on("system-prompt/assemble", _drop_wd, mode=EventMode.WATERFALL)
    re_inserted = await ctx.systemPrompt.assemble({"agent": agent, "scope": agent})
    assert "Current working directory" in render_context_snapshot(re_inserted)
    off_drop()

    # Disposing while an operation is active triggers _teardown wait + aborted _lifecycle
    async def _slow_stat(target: Any, sig: Any = None) -> Any:
        await asyncio.sleep(0.05)
        return await orig_stat(target, sig)

    ctx.fs.stat = _slow_stat  # type: ignore[method-assign]
    pending = asyncio.create_task(ctx.workingDirectory.ensure(agent))
    await asyncio.sleep(0.01)
    await ctx.fiber.dispose()
    with pytest.raises((FsError, RuntimeError)):
        await pending
    await handle["dispose"]()

    # Disposing during downstream system-prompt/assemble raises post-next_fn disposed error
    ctx2 = Context()
    await _mount_stack(ctx2, default_dir)
    handle2 = await ctx2.agents.create({"sessionId": "wd-disposed-assemble"})
    agent2 = handle2["agent"]

    async def _dispose_downstream(assembly: Any, _context: Any, next_fn: Any) -> Any:
        res = await next_fn()
        await ctx2.fiber.dispose()
        return res

    ctx2.on("system-prompt/assemble", _dispose_downstream, mode=EventMode.WATERFALL)
    with pytest.raises(RuntimeError, match="disposed"):
        await ctx2.systemPrompt.assemble({"agent": agent2, "scope": agent2})
    await handle2["dispose"]()


@pytest.mark.asyncio
async def test_working_directory_requires_dependencies() -> None:
    """Fails loud when sessionProjections or systemPrompt are missing."""
    ctx_no_proj = Context()
    with pytest.raises(RuntimeError, match="requires sessionProjections"):
        WorkingDirectoryService(ctx_no_proj)

    ctx_no_prompt = Context()
    ctx_no_prompt.use(SessionProjectionRegistry)
    await ctx_no_prompt.flush()
    with pytest.raises(RuntimeError, match="requires systemPrompt"):
        WorkingDirectoryService(ctx_no_prompt)
