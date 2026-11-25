"""Same-session subagent domain (``ctx.subagents``).

Ports dsh's ``@deepseek-ai/dsh-subagent`` at
``packages/subagent/subagent/src/index.ts``: a provider-backed delegation seam
with a live child registry, lineage, capacity, naming, messaging, and
interruption. Child creation is delegated to a registered provider; the
in-process spawn/fork providers live in this repository (base bundle rows
``subagent-spawn-in-process`` and ``subagent-fork-in-process``).

Lineage facts are logged as ``subagent/catalog`` and ``subagent/descriptor``
session events so the model-visible tree is reconstructable from the log.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from animas import Context, Service

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.error import HarnessError
from magitech_harness.subagent.manager import SubagentManager
from magitech_harness.subagent.types import (
    SubagentSpec,
)

ROOT_AGENT_NAME = "root"
DEFAULT_MAX_ACTIVE = 10
DEFAULT_MAX_DEPTH = 1


class SubagentError(HarnessError):
    """Stable error taxonomy for the subagent domain."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


@dataclass(frozen=True)
class SubagentProvider:
    """A named backend able to start one delegated child."""

    name: str
    start: Callable[[SubagentSpec, Any], Awaitable[dict[str, Any]]]
    capabilities: frozenset[str] = frozenset()


@dataclass
class _Member:
    """One registered child in the live tree."""

    child_id: str
    name: str
    parent_id: str | None
    depth: int
    record: Any = None
    state: str = "running"
    label: str = ""


@dataclass
class SubagentService(Service):
    """``ctx.subagents``: provider registry plus the live child tree."""

    ctx: Context

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        Service.__init__(self, ctx, "subagents")
        raw = dict(config or {})
        self._max_active = int(raw.get("maxActiveSubagents", DEFAULT_MAX_ACTIVE))
        self._max_depth = int(raw.get("maxDepth", DEFAULT_MAX_DEPTH))
        self._providers: dict[str, SubagentProvider] = {}
        self._members: dict[str, _Member] = {}
        self._manager: SubagentManager | None = None
        self._registration: Callable[[], Any] | None = None

    def register_provider(self, provider: SubagentProvider) -> Callable[[], None]:
        """Register a delegation provider; returns its disposer."""
        self._providers[provider.name] = provider

        def dispose() -> None:
            if self._providers.get(provider.name) is provider:
                self._providers.pop(provider.name)

        return dispose

    def get_provider(self, name: str) -> SubagentProvider | None:
        return self._providers.get(name)

    @property
    def max_active_subagents(self) -> int:
        return self._max_active

    def resolve_max_depth(self, configured: int | str | None = None) -> int | None:
        """Resolve the depth cap; ``'provider-managed'`` means no local cap."""
        if configured == "provider-managed":
            return None
        depth = self._max_depth if configured is None else int(configured)
        if depth < 0:
            raise SubagentError("maxDepth must be non-negative", "INVALID_MAX_DEPTH")
        return depth

    def _manager_for(self, ctx: Context) -> SubagentManager:
        if self._manager is None:
            self._manager = SubagentManager(ctx)
        return self._manager

    async def startActivation(
        self,
        spec: SubagentSpec,
        *,
        provider: str = "spawn",
        parent_id: str | None = None,
        signal: AbortSignal | None = None,
    ) -> dict[str, Any]:
        """Start one managed child and register it in the live tree."""
        backend = self._providers.get(provider)
        if backend is None:
            raise SubagentError(
                f'no subagent provider named "{provider}" is registered',
                "PROVIDER_NOT_FOUND",
            )
        # A root caller is not itself a registered child; only a known child
        # contributes delegation depth.
        depth = 0
        if parent_id is not None and parent_id in self._members:
            depth = self._members[parent_id].depth + 1
        cap = self.resolve_max_depth()
        if cap is not None and depth > cap:
            raise SubagentError(
                f"subagent depth {depth} exceeds the configured cap {cap}",
                "MAX_DEPTH_EXCEEDED",
            )
        if (
            sum(1 for member in self._members.values() if member.state == "running")
            >= self._max_active
        ):
            raise SubagentError(
                "background subagent limit reached", "CAPACITY_EXCEEDED"
            )

        child_id = f"sub-{uuid.uuid4().hex[:8]}"
        name = spec.agent_name or f"agent-{len(self._members) + 1}"
        if name == ROOT_AGENT_NAME:
            raise SubagentError("'root' is reserved", "AGENT_NAME_RESERVED")
        if any(member.name == name for member in self._members.values()):
            raise SubagentError(
                f'agent name "{name}" is already in use', "AGENT_NAME_EXISTS"
            )

        manager = self._manager_for(self.ctx)
        records = await manager.invoke([spec], parent_signal=signal)
        record = records[0]
        member = SubagentRecordMember(
            child_id=child_id,
            name=name,
            parent_id=parent_id,
            depth=depth,
            record=record,
            label=spec.role,
        )
        self._members[child_id] = member
        session = getattr(record, "session_id", None)
        self._log_event(
            "subagent/descriptor",
            {
                "childId": child_id,
                "name": name,
                "parentId": parent_id,
                "depth": depth,
                "label": spec.role,
                "sessionId": session,
            },
        )
        result = await backend.start(spec, self.ctx)
        return {
            "kind": "activation",
            "id": child_id,
            "name": name,
            "sessionId": session,
            "provider": provider,
            **result,
        }

    def _member(self, child_id: str) -> _Member:
        member = self._members.get(child_id)
        if member is None:
            raise SubagentError(
                f'unknown subagent "{child_id}"', "AGENT_NAME_NOT_FOUND"
            )
        return member

    def root_of(self, caller: Any) -> Any:
        """The root agent of the caller's tree (the caller when no parent)."""
        del caller
        return ROOT_AGENT_NAME

    def names_of(self, agent: Any) -> dict[str, str | None]:
        """Return ``self`` and ``parent`` names for one live agent."""
        agent_id = getattr(agent, "id", None)
        member = self._members.get(agent_id) if isinstance(agent_id, str) else None
        if member is None:
            return {"self": ROOT_AGENT_NAME, "parent": None}
        return {"self": member.name, "parent": member.parent_id or ROOT_AGENT_NAME}

    def capacity_of(self, caller: Any) -> dict[str, int]:
        """Current running count and capacity for the caller's tree."""
        del caller
        running = sum(
            1 for member in self._members.values() if member.state == "running"
        )
        return {"running": running, "limit": self._max_active}

    async def list_descendants(
        self, root_id: str, signal: AbortSignal | None = None
    ) -> list[dict[str, Any]]:
        """List the tree below one root as model-facing rows."""
        del signal
        return [
            {
                "id": member.child_id,
                "name": member.name,
                "parentId": member.parent_id,
                "depth": member.depth,
                "status": member.state,
                "description": member.label,
            }
            for member in self._members.values()
        ]

    def resolve_name(
        self, caller: Any, name: str, signal: AbortSignal | None = None
    ) -> str:
        """Resolve a child name or path to its stable id."""
        del caller, signal
        if name == ROOT_AGENT_NAME:
            return ROOT_AGENT_NAME
        for child_id, member in self._members.items():
            if member.name == name:
                return child_id
        raise SubagentError(
            f'no agent named "{name}" in this tree', "AGENT_NAME_NOT_FOUND"
        )

    async def send_message(
        self,
        caller: Any,
        target_id: str,
        content: list[dict[str, Any]],
        *,
        signal: AbortSignal | None = None,
    ) -> dict[str, Any]:
        """Deliver a follow-up instruction to an existing child."""
        del caller, signal
        member = self._member(target_id)
        if member.state != "running":
            raise SubagentError(f'agent "{member.name}" is not running', "UNAUTHORIZED")
        text = "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
        manager = self._manager_for(self.ctx)
        await manager.send_message(member.child_id, text)
        return {"messageId": f"msg-{uuid.uuid4().hex[:12]}"}

    def interrupt(
        self, target_id: str, authority: Mapping[str, Any]
    ) -> dict[str, bool]:
        """Request cooperative interruption of an active child."""
        del authority
        member = self._member(target_id)
        manager = self._manager_for(self.ctx)
        task = manager._tasks.get(member.child_id)
        if task is not None and not task.done():
            task.cancel()
        member.state = "terminated"
        return {"accepted": True}

    def prompt(
        self,
        caller: Any,
        message: dict[str, Any],
        *,
        signal: AbortSignal | None = None,
    ) -> dict[str, Any]:
        """Inject a model-visible message into the caller's next step."""
        del signal
        agent_ctx = getattr(caller, "ctx", None)
        registry = agent_ctx.get("agents", None, strict=False) if agent_ctx else None
        if registry is None:
            return {"accepted": False}
        registry.get(caller.id).inbox.splice("next-turn", 0, 0, [message])
        return {"accepted": True}

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        sessions = self.ctx.get("sessions", None, strict=False)
        if sessions is None:  # pragma: no cover - base bundle mounts sessions
            return
        session = getattr(sessions, "active_session", None)
        if session is None:
            return
        session.append(event_type, data)


# Alias used by the tools and providers.
SubagentRecordMember = _Member


__all__ = [
    "DEFAULT_MAX_ACTIVE",
    "DEFAULT_MAX_DEPTH",
    "ROOT_AGENT_NAME",
    "SubagentError",
    "SubagentProvider",
    "SubagentService",
]
