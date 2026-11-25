"""SubagentManager for spawning, coordinating, and managing delegated subagents."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from animas import Context, Service
from magitech_harness import check_bundle_entries, create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.cli import summarize_headless_outcome
from magitech_harness.llm.message import create_user_message
from magitech_harness.scope import create_scope
from magitech_harness.subagent.types import (
    RESEARCH_PRESET_TOOLS,
    SubagentRecord,
    SubagentSpec,
    SubagentState,
)
from magitech_harness.tools.service import ToolService

logger = logging.getLogger(__name__)

SUBAGENT_TRIAD_TOOLS = frozenset(
    {"invoke_subagent", "send_message", "manage_subagents"}
)


def _make_abort_forwarder(
    parent_signal: AbortSignal, child_signal: AbortSignal
) -> Callable[[], None]:
    """Creates a zero-argument callback propagating parent abort to child."""

    def _forward() -> None:
        child_signal.abort(parent_signal.reason)

    return _forward


class SubagentManager(Service):
    """Manages conversational subagent lifecycles and workspace isolation."""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(ctx, "_subagentWorker")
        cfg = config or {}
        self.subsessions_dir = cfg.get("subsessions_dir") or (
            Path.home() / ".agents" / "sessions" / "subsessions"
        )
        self.subsessions_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root = (
            cfg.get("workspace_root")
            or getattr(ctx, "workspace_dir", None)
            or Path.cwd()
        )

        self._records: dict[str, SubagentRecord] = {}
        self._child_contexts: dict[str, Context] = {}
        self._signals: dict[str, AbortSignal] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._worktrees: dict[str, Path] = {}
        self._allowed_tools: dict[str, frozenset[str]] = {}
        self._child_scopes: dict[str, Any] = {}
        self._child_agents: dict[str, Any] = {}

    def list_records(self) -> list[SubagentRecord]:
        """Returns all tracked subagent execution records."""
        return list(self._records.values())

    async def invoke(
        self,
        specs: Sequence[SubagentSpec],
        *,
        parent_signal: AbortSignal | None = None,
    ) -> list[SubagentRecord]:
        """Spawns and executes subagents according to given specifications."""
        records: list[SubagentRecord] = []
        waiting_coroutines: list[tuple[str, str]] = []

        for spec in specs:
            subagent_id = f"sub-{uuid.uuid4().hex[:8]}"
            workspace_path = await self._resolve_workspace(subagent_id, spec.workspace)

            child_model = (
                getattr(getattr(self.ctx, "llm", None), "default_model", "inherit")
                if spec.model == "inherit"
                else spec.model
            )
            child_name = spec.agent_name or "subagent"

            # The child context is built by loading the default bundle,
            # so it carries the full set of bundle rows (sessions with
            # persistence, projections, title service, checkpoint policy,
            # projection cache, agent loop) instead of a hand-built subset.
            # With a running loop the rows settle on the caller's flush;
            # fail loud here if any row did not mount.
            child_ctx = create_harness(storage_dir=self.subsessions_dir)
            setattr(child_ctx, "workspace_dir", workspace_path)  # noqa: B010
            await child_ctx.flush()
            check_bundle_entries(child_ctx)

            child_sessions = child_ctx.sessions
            session_id = f"session-{subagent_id}"
            child_sessions.create(session_id)
            session_path = child_sessions.session_path(session_id)

            # Copy the parent's providers through the public seams.
            self._copy_providers(child_ctx)

            # Set up scoped tools
            effective_tools = self._populate_child_tools(child_ctx.tools, spec)

            # Cancellation propagation
            child_signal = AbortSignal()
            if parent_signal is not None:
                parent_signal.on_abort(
                    _make_abort_forwarder(parent_signal, child_signal)
                )
            self._signals[subagent_id] = child_signal

            record = SubagentRecord(
                subagent_id=subagent_id,
                role=spec.role,
                agent_name=child_name,
                model_id=child_model or "inherit",
                workspace_path=workspace_path,
                session_id=session_id,
                session_path=session_path,
                state=SubagentState.PENDING,
                state_detail="Initialized",
                effective_tools=effective_tools,
            )

            allowed_set = self._allowed_tools_for(spec)
            if allowed_set is not None:
                self._allowed_tools[subagent_id] = allowed_set
            self._records[subagent_id] = record
            self._child_contexts[subagent_id] = child_ctx
            records.append(record)

            prompt = (
                spec.prompt or f"You are a subagent with role {spec.role}. Acknowledge."
            )
            if spec.wait:
                waiting_coroutines.append((subagent_id, prompt))
            else:
                task = asyncio.create_task(self._run_child(subagent_id, prompt))
                self._tasks[subagent_id] = task

        if waiting_coroutines:
            await asyncio.gather(
                *[self._run_child(sid, prompt) for sid, prompt in waiting_coroutines]
            )

        return records

    def _copy_providers(self, child_ctx: Context) -> None:
        """Copies the parent's providers onto the child, preserving the active one."""
        parent_llm = getattr(self.ctx, "llm", None)
        if parent_llm is None:
            return
        child_llm = child_ctx.llm
        active = parent_llm.active_provider
        active_name: str | None = None
        for name, provider in parent_llm.providers.items():
            child_llm.register_provider(name, provider)
            if provider is active:
                active_name = name
        if active_name is not None:
            child_llm.set_active_provider(active_name)

    @staticmethod
    def _allowed_tools_for(spec: SubagentSpec) -> frozenset[str] | None:
        """Resolve the child's allowed tool set from the spec and role preset."""
        if spec.allowed_tools is not None:
            return spec.allowed_tools
        if "research" in spec.role.lower():
            return RESEARCH_PRESET_TOOLS
        return None

    def _populate_child_tools(
        self,
        child_tools: ToolService,
        spec: SubagentSpec,
    ) -> list[str]:
        """Copies parent tools the child does not already mount."""
        allowed = self._allowed_tools_for(spec)

        effective: list[str] = []
        parent_tools = getattr(self.ctx, "tools", None)
        if parent_tools is not None:
            for tool in parent_tools.definitions():
                name = tool["name"]
                if name in SUBAGENT_TRIAD_TOOLS:
                    continue
                if allowed is not None and name not in allowed:
                    continue
                # The child bundle mounts the base tool set itself; copying a
                # parent tool of the same name would double-register it.
                if child_tools.get(name) is None:
                    child_tools.register(tool)
                effective.append(name)
        return effective

    @staticmethod
    def _pin_fail_closed_approval(child_ctx: Context, agent: Any) -> None:
        """Pin the child to a fail-closed approval scope (source=delegation).

        A child never inherits the parent's interactive approver or one-shot
        grants.
        """
        approval = getattr(child_ctx, "approval", None)
        if approval is not None:  # pragma: no branch
            try:
                approval.set_policy(agent, "never")
            except (ValueError, TypeError):  # pragma: no cover
                # defensive: a stub approval service may reject policies
                pass

    def _restrict_child_tools(
        self, child_ctx: Context, agent: Any, subagent_id: str
    ) -> None:
        """Restrict the child's own scope to its allowed tool set.

        The child bundle mounts the base tools itself, so the restriction is the
        only thing that makes ``allowed_tools`` an enforcement rather than a
        report. A preset or caller may name tools this composition does not
        mount; an absent name is not a reason to fail the child, but it must
        narrow the restriction. An empty intersection therefore means *no*
        tool is visible, never no restriction: failing open would hand a child
        that asked for a restricted set every bundle tool, shell included.
        """
        allowed = self._allowed_tools.get(subagent_id)
        if allowed is None:
            return
        available = {definition["name"] for definition in child_ctx.tools.definitions()}
        allow = sorted(allowed & available)
        child_scope = create_scope(child_ctx, agent)
        # ``{"allow": []}`` is a valid restriction that denies every name.
        child_scope.ctx.tools.restrict({"allow": allow})
        # Retained with the child for its lifetime, like the other per-child
        # registries; the scope owns the restriction layer.
        self._child_scopes[subagent_id] = child_scope

    async def _resolve_workspace(
        self,
        subagent_id: str,
        workspace_mode: str,
    ) -> Path:
        """Resolves workspace path, creating a git worktree if branch mode."""
        parent_ws = self.workspace_root
        if workspace_mode != "branch":
            return parent_ws

        git_dir = parent_ws / ".git"
        if not git_dir.exists():
            return parent_ws

        worktree_dir = self.subsessions_dir / "worktrees" / subagent_id
        worktree_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "worktree",
                "add",
                "--detach",
                str(worktree_dir),
                "HEAD",
                cwd=str(parent_ws),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode == 0:
                self._worktrees[subagent_id] = worktree_dir
                return worktree_dir
        except (OSError, subprocess.SubprocessError) as ex:
            logger.warning("Failed to create worktree for %s: %s", subagent_id, ex)

        return parent_ws

    async def _run_child(self, subagent_id: str, prompt: str) -> None:
        """Executes a single conversational turn in the child subagent loop."""
        record = self._records[subagent_id]
        child_ctx = self._child_contexts[subagent_id]
        signal = self._signals.get(subagent_id)

        record.state = SubagentState.RUNNING
        record.state_detail = "Executing turn"

        try:
            agent = self._child_agents.get(subagent_id)
            if agent is not None and child_ctx.sessions.get(agent.session.id) is None:
                raise RuntimeError(
                    f"Session {record.session_id} not found in child context"
                )
            if agent is None:
                # Resolve provider/model for the unified driver.
                model = None if record.model_id == "inherit" else record.model_id
                if model is not None:
                    if "/" in model:
                        provider, model_name = model.split("/", 1)
                    else:
                        provider, model_name = "mock", model
                else:
                    registered = child_ctx.llm.list_providers()
                    provider = (
                        registered[0]["id"] if registered else "mock"
                    )  # pragma: no cover - no-provider fallback
                    model_name = "mock"
                options: dict[str, Any] = {
                    "agentOptions": {"provider": provider, "model": model_name},
                }
                if signal is not None:  # pragma: no branch - optional parent signal
                    options["signal"] = signal
                handle = await child_ctx.agents.create(options)
                agent = handle["agent"]
                self._pin_fail_closed_approval(child_ctx, agent)
                self._restrict_child_tools(child_ctx, agent, subagent_id)
                self._child_agents[subagent_id] = agent
                session = agent.session
                record.session_id = session.id
            else:
                session = agent.session

            first_seq = len(session.events)
            agent.followup(
                create_user_message(
                    content=[{"type": "text", "text": prompt}],
                    source={"kind": "user"},
                )
            )
            await agent.when_idle()
            outcome = summarize_headless_outcome(session, first_seq)
            res = outcome.text
            record.state = SubagentState.COMPLETED
            record.state_detail = "Finished"
            record.summary = res
        except Exception as ex:
            record.state = SubagentState.ERRORED
            record.state_detail = "Failed"
            record.error = str(ex)
            logger.exception("Error executing subagent turn for %s", subagent_id)

    async def send_message(self, subagent_id: str, message: str) -> SubagentRecord:
        """Sends a follow-up instruction to an existing active subagent."""
        record = self._records.get(subagent_id)
        if record is None:
            raise ValueError(f"Subagent '{subagent_id}' does not exist.")

        if record.state is SubagentState.TERMINATED:
            raise ValueError(f"Cannot message terminated subagent '{subagent_id}'.")

        await self._run_child(subagent_id, message)
        return record

    async def manage(
        self,
        action: str,
        subagent_ids: list[str] | None = None,
    ) -> list[SubagentRecord]:
        """Inspects or terminates active subagents."""
        if action == "list":
            return self.list_records()

        if action == "kill":
            ids = subagent_ids or []
            results: list[SubagentRecord] = []
            for sid in ids:
                if sid in self._records:
                    await self._terminate_child(sid)
                    results.append(self._records[sid])
            return results

        if action == "kill_all":
            results = []
            for sid in list(self._records.keys()):
                await self._terminate_child(sid)
                results.append(self._records[sid])
            return results

        raise ValueError(f"Unknown manage_subagents action '{action}'")

    async def _terminate_child(self, subagent_id: str) -> None:
        """Terminates an individual subagent and cleans up worktrees."""
        if subagent_id in self._signals:
            self._signals[subagent_id].abort("Subagent terminated by manager.")

        if subagent_id in self._tasks:
            task = self._tasks[subagent_id]
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if subagent_id in self._worktrees:
            worktree_dir = self._worktrees.pop(subagent_id)
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree_dir),
                    cwd=str(self.workspace_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            except (OSError, subprocess.SubprocessError) as ex:
                logger.warning("Failed to remove worktree %s: %s", worktree_dir, ex)

        record = self._records.get(subagent_id)
        if record is not None:
            record.state = SubagentState.TERMINATED
            record.state_detail = "Terminated"
            # A terminated subagent never runs again: close its session
            # first so the write handle is released and flushed.
            self._child_contexts[subagent_id].sessions.close(record.session_id)

    async def shutdown(self) -> None:
        """Terminates all active subagents and releases allocated resources."""
        await self.manage("kill_all")
