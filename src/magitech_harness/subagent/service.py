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
    SubagentState,
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
    start: Callable[..., Awaitable[dict[str, Any]]]
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
    session_id: str | None = None


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
        session: Any = None,
    ) -> dict[str, Any]:
        """Start one managed child and register it in the live tree.

        ``session`` is the caller's actual session. Lineage is appended there so
        the model-visible tree is reconstructable from the log.
        """
        backend = self._providers.get(provider)
        if backend is None:
            raise SubagentError(
                f'no subagent provider named "{provider}" is registered',
                "PROVIDER_NOT_FOUND",
            )
        cap = self.resolve_max_depth(spec.max_depth)
        parent_member = self._member_by_session(parent_id)
        depth = 0 if parent_member is None else parent_member.depth + 1
        if cap is not None and depth > cap:
            raise SubagentError(
                f"subagent depth {depth} exceeds the configured cap {cap}",
                "MAX_DEPTH_EXCEEDED",
            )
        self._sync_states()
        if len(self._running_members()) >= self._max_active:
            raise SubagentError(
                "background subagent limit reached", "CAPACITY_EXCEEDED"
            )
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
        # The manager owns the child id namespace; reuse it so messaging and
        # interruption reach the actual child.
        child_id = record.subagent_id
        member = _Member(
            child_id=child_id,
            name=name,
            parent_id=parent_id,
            depth=depth,
            record=record,
            label=spec.role,
            session_id=record.session_id,
        )
        self._members[child_id] = member
        self._sync_states()
        if session is not None:
            session.append(
                "subagent/descriptor",
                {
                    "childId": child_id,
                    "name": name,
                    "parentId": parent_id,
                    "depth": depth,
                    "label": spec.role,
                    "sessionId": record.session_id,
                },
            )
            session.append(
                "subagent/catalog",
                {
                    "version": 1,
                    "members": [
                        {
                            "id": m.child_id,
                            "name": m.name,
                            "parentId": m.parent_id,
                            "depth": m.depth,
                        }
                        for m in self._members.values()
                    ],
                },
            )
        result = await backend.start(spec, self.ctx, session=session)
        return {
            "kind": "activation",
            "id": child_id,
            "name": name,
            "sessionId": record.session_id,
            "provider": provider,
            **result,
        }

    def _running_members(self) -> list[_Member]:
        self._sync_states()
        return [
            member for member in self._members.values() if member.state == "running"
        ]

    def _sync_states(self) -> None:
        """Project each manager record's lifecycle onto its member row."""
        terminal = {
            SubagentState.COMPLETED.value: "completed",
            SubagentState.ERRORED.value: "errored",
            SubagentState.TERMINATED.value: "terminated",
        }
        for member in self._members.values():
            record = member.record
            state = getattr(record, "state", None)
            if state is None:
                continue
            value = getattr(state, "value", state)
            member.state = terminal.get(value, "running")

    def _member_by_session(self, session_id: Any) -> _Member | None:
        """Resolve a caller session id to its member row, if any."""
        if not isinstance(session_id, str):
            return None
        for member in self._members.values():
            if member.session_id == session_id or member.child_id == session_id:
                return member
        return None

    def _caller_member(self, caller: Any) -> _Member | None:
        """The member row for one live caller, or None when it is the root."""
        return self._member_by_session(getattr(caller, "id", None))

    def _ancestor_of(self, caller: Any, target: _Member) -> bool:
        """Whether *caller* is the root or an ancestor of *target*."""
        origin = self._caller_member(caller)
        if origin is None:
            return True
        # A live non-root caller is an ancestor only of its descendants, never
        # of itself.
        current: _Member | None = (
            self._members.get(target.parent_id)
            if target.parent_id is not None
            else None
        )
        while current is not None:
            if current.child_id == origin.child_id:
                return True
            current = (
                self._members.get(current.parent_id)
                if current.parent_id is not None
                else None
            )
        return False

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
        running = len(self._running_members())
        return {"running": running, "limit": self._max_active}

    async def list_descendants(
        self, root_id: str, signal: AbortSignal | None = None
    ) -> list[dict[str, Any]]:
        """List the tree below one root as model-facing rows."""
        del signal
        self._sync_states()
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
        del signal
        if name == ROOT_AGENT_NAME:
            return ROOT_AGENT_NAME
        for child_id, member in self._members.items():
            if member.name == name:
                if not self._ancestor_of(caller, member):
                    raise SubagentError(
                        f'agent "{name}" is outside the caller tree',
                        "UNAUTHORIZED",
                    )
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
        del signal
        member = self._member(target_id)
        if not self._ancestor_of(caller, member):
            raise SubagentError(
                "messaging requires an ancestor of the target", "UNAUTHORIZED"
            )
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
        caller = authority.get("agent")
        member = self._member(target_id)
        if not self._ancestor_of(caller, member):
            raise SubagentError(
                "interruption requires an ancestor of the target", "UNAUTHORIZED"
            )
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
        live = registry.get(caller.id)
        if live is None:  # pragma: no cover - live callers own an agent
            return {"accepted": False}
        live.inbox.splice("next-turn", 0, 0, [message])
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
