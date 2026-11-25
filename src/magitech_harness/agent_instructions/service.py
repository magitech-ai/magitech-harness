"""Workspace instruction loader plugin (dsh ``@deepseek-ai/dsh-agent-instructions``).

Instructions enter durable context as ``user/message`` records with a typed
``agent-instructions`` source. There is no system-prompt section: one context can
host agents with different working directories, so repository text is lower
authority than system, developer and direct user instructions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from animas import Context, EventMode

from magitech_harness.agent_instructions.config import (
    AgentInstructionsError,
    ResolvedConfig,
    resolve_config,
)
from magitech_harness.agent_instructions.files import (
    load_instruction_files,
)
from magitech_harness.agent_instructions.render import (
    render_instruction_chains,
)
from magitech_harness.agent_instructions.state import (
    SOURCE_KIND,
    agent_instructions_message,
    baseline_instruction_state,
    reconcile,
    visible_baseline,
)

TOUCH_TOOLS = frozenset({"read_file", "write_file", "edit_file"})


def _messages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    return (
        [message for message in messages if isinstance(message, dict)]
        if isinstance(messages, list)
        else []
    )


def _session_messages(agent: Any) -> list[dict[str, Any]]:
    session = getattr(agent, "session", None)
    if session is None:
        return []
    derive = getattr(session, "derive_messages", None)
    return list(derive()) if callable(derive) else []


class AgentInstructionsPlugin:
    """Listener bundle installed by :func:`apply`."""

    def __init__(self, ctx: Context, config: ResolvedConfig) -> None:
        self.ctx = ctx
        self.config = config
        self.touches: dict[str, set[str]] = {}

    async def _cwd(self, agent: Any, signal: Any) -> str:
        working_directory = self.ctx.get("workingDirectory", None, strict=False)
        ensure = getattr(working_directory, "ensure", None)
        if callable(ensure):
            result = ensure(agent, signal)
            if hasattr(result, "__await__"):
                result = await result
            return str(result)
        cwd = getattr(agent, "cwd", None) or getattr(self.ctx, "cwd", None)
        return str(cwd or ".")

    async def _notice(
        self, agent: Any, payload: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        from pathlib import Path

        cwd = Path(await self._cwd(agent, payload.get("signal")))
        files = load_instruction_files(self.config, cwd)
        messages = [*_session_messages(agent), *_messages(payload)]
        baseline = visible_baseline(messages)
        identity = self.config.identity(cwd, cwd)
        if baseline is None or baseline.get("baselineIdentity") != identity:
            changes, _versions = baseline_instruction_state(files)
            if not files:
                text, changes = render_instruction_chains(
                    [], self.config.max_bytes, baseline=True
                )
            else:
                text, changes = render_instruction_chains(
                    files, self.config.max_bytes, baseline=True
                )
            if not text:
                return None
            return agent_instructions_message(
                text, changes, baseline=True, identity=identity
            )
        deltas = reconcile(messages, files)
        if not deltas:
            return None
        text, _ = render_instruction_chains(
            files, self.config.max_bytes, baseline=False
        )
        if not text:
            return None
        return agent_instructions_message(text, deltas)

    async def pre_step(self, payload: Mapping[str, Any], next_fn: Any) -> Any:
        decision = next_fn()
        if hasattr(decision, "__await__"):
            decision = await decision
        if decision.get("kind") == "reject":
            return decision
        agent = payload.get("agent")
        if agent is None:
            return decision
        try:
            notice = await self._notice(agent, payload)
        except Exception:  # noqa: BLE001 -- instruction discovery must not block a step
            return decision
        if notice is None:
            return decision
        return {
            **decision,
            "messages": [*decision.get("messages", []), notice],
        }

    def on_tools_result(self, execution: Any, _result: Any) -> None:
        if not isinstance(execution, Mapping):
            return
        name = execution.get("name")
        if name not in TOUCH_TOOLS:
            return
        arguments = execution.get("arguments")
        if not isinstance(arguments, Mapping):
            return
        path = arguments.get("file_path") or arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return
        agent = execution.get("agent")
        if agent is None:
            return
        self.touches.setdefault(str(id(agent)), set()).add(path.strip())


async def apply(ctx: Context, config: dict[str, Any] | None = None) -> None:
    """Mount the workspace instruction listeners."""
    if not config or config.get("maxBytes") is None:
        raise AgentInstructionsError("agent-instructions requires maxBytes")
    resolved = resolve_config(config)
    if resolved.max_bytes <= 0:
        return
    plugin = AgentInstructionsPlugin(ctx, resolved)
    ctx.on("agent/pre-step", plugin.pre_step, mode=EventMode.WATERFALL)
    ctx.on("tools/result", plugin.on_tools_result)


apply.inject = ["agents", "workingDirectory"]  # type: ignore[attr-defined]

__all__ = ["SOURCE_KIND", "TOUCH_TOOLS", "AgentInstructionsPlugin", "apply"]
