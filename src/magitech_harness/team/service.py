"""``ctx.agentTeams``: shared task boards owned by subagent tree roots.

Ports ``packages/team/agent-team/src/{index,task-board}.ts``. The board is the
one owned by the caller's *root* Session (not the caller's own child session),
so teammates and fork children see the same tasks. Every mutation is one
append-only ``team/task`` event on that root log; the ``agentTeam`` projection
folds it, so resume and fork restore the board.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from animas import Context, Service

from magitech_harness.session.projection_registry import ProjectionHandle
from magitech_harness.team.fold import TEAM_PROJECTION
from magitech_harness.team.task_graph import (
    assert_task_graph,
    project_task_view,
    task_ready,
)
from magitech_harness.team.types import (
    TeamError,
    required_text,
    task_id,
    write_scope,
)

DEFAULT_MAX_TASKS = 256

_REMOVED_CONFIG_KEYS = ("maxMembers", "maxMessageBytes", "disposalTimeoutMs")


def _positive_limit(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > 2**53 - 1
    ):
        raise TeamError(
            f"{name} must be a positive safe integer", "TEAM_INVALID_CONFIG"
        )
    return value


class TeamService(Service):
    """Event-sourced shared task board with compare-and-set revisions."""

    inject: ClassVar[list[str]] = ["agents", "sessionProjections", "subagents"]

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "agentTeams")
        cfg = dict(config or {})
        for key in _REMOVED_CONFIG_KEYS:
            if key in cfg:
                raise TeamError(
                    f"agent-team: {key} is no longer supported; teammates are subagents",
                    "TEAM_INVALID_CONFIG",
                )
        self._max_tasks = _positive_limit(
            "maxTasks", cfg.get("maxTasks", DEFAULT_MAX_TASKS)
        )
        self._default_enabled = bool(cfg.get("defaultEnabled", False))
        self._handle: ProjectionHandle[dict[str, Any]] | None = None

    async def init(self) -> Callable[[], None]:
        """Register the ``agentTeam`` projection and auto-enable new roots."""
        registry = self.ctx.get("sessionProjections", None, strict=True)
        if registry is None:
            raise RuntimeError("agentTeams requires the sessionProjections service")
        self._handle = registry.register(TEAM_PROJECTION)

        if self._default_enabled:

            def on_created(payload: Mapping[str, Any]) -> None:
                agent = payload.get("agent")
                if agent is not None and not self.is_enabled(agent):
                    self._session_of(agent).append("team/mode", {"enabled": True})

            dispose_listener = self.ctx.on("agent/created", on_created)

            def dispose() -> None:
                dispose_listener()
                self._handle.dispose() if self._handle is not None else None

            return dispose
        return self._handle.dispose

    # -- root resolution ---------------------------------------------------

    def _subagents(self) -> Any:
        return self.ctx.get("subagents", None, strict=False)

    def _members(self) -> dict[str, Any]:
        subagents = self._subagents()
        return getattr(subagents, "_members", {}) if subagents is not None else {}

    def _member_for(self, agent: Any) -> Any:
        """Resolve a live agent to its subagent member by session or child id."""
        members = self._members()
        agent_id = getattr(agent, "id", None)
        if not isinstance(agent_id, str):
            return None
        return next(
            (
                member
                for member in members.values()
                if member.session_id == agent_id or member.child_id == agent_id
            ),
            None,
        )

    def _caller_id(self, agent: Any) -> Any:
        """The stable tree id used for task ownership (child id, or root id)."""
        member = self._member_for(agent)
        if member is not None:
            return member.child_id
        return getattr(agent, "id", None)

    def _root_session(self, agent: Any) -> Any:
        """The root Session of the caller's subagent tree."""
        member = self._member_for(agent)
        if member is None:
            return agent.session
        members = self._members()
        seen: set[str] = set()
        current = member
        while current.parent_id is not None and current.parent_id not in seen:
            seen.add(current.parent_id)
            parent = next(
                (
                    candidate
                    for candidate in members.values()
                    if candidate.session_id == current.parent_id
                    or candidate.child_id == current.parent_id
                ),
                None,
            )
            if parent is None:
                break
            current = parent
        root_id = current.parent_id
        agents = self.ctx.get("agents", None, strict=False)
        if root_id is not None and agents is not None:
            root_agent = agents.get(root_id)
            if root_agent is not None:
                return root_agent.session
            for candidate in agents.list():
                session = getattr(candidate, "session", None)
                if (
                    session is not None
                    and getattr(session, "session_id", None) == root_id
                ):
                    return session
        sessions = self.ctx.get("sessions", None, strict=False)
        if root_id is not None and sessions is not None:
            root_session = sessions.get(root_id)
            if root_session is not None:
                return root_session
        return agent.session

    def _session_of(self, agent: Any) -> Any:
        return self._root_session(agent)

    def _is_lead(self, caller: Any) -> bool:
        """Exact Team Lead identity: the caller is the root agent."""
        session = self._root_session(caller)
        return getattr(caller, "id", None) == getattr(session, "session_id", None)

    # -- state -------------------------------------------------------------

    def _state(self, agent: Any) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("agentTeams requires the sessionProjections service")
        state = self._handle.state_of(self._session_of(agent))
        if state.get("failure") is not None:
            raise TeamError(
                f"the shared task board is corrupt: {state['failure']}",
                "TEAM_STATE_CORRUPT",
            )
        return state

    def is_enabled(self, agent: Any) -> bool:
        """Whether the shared task board is enabled for the agent's tree."""
        try:
            return self._state(agent).get("enabled") is True
        except TeamError as error:
            if getattr(error, "code", None) == "UNAUTHORIZED":
                return False
            raise

    def enable(self, agent: Any) -> None:
        """Enable the task board (idempotent)."""
        if not self.is_enabled(agent):
            self._session_of(agent).append("team/mode", {"enabled": True})

    def disable(self, agent: Any) -> None:
        """Disable the task board (idempotent)."""
        if self.is_enabled(agent):
            self._session_of(agent).append("team/mode", {"enabled": False})

    # -- reads ------------------------------------------------------------

    def _views(self, agent: Any) -> list[dict[str, Any]]:
        state = self._state(agent)
        return [
            project_task_view(state["tasks"], task)
            for task in state["tasks"]
            if task["status"] != "deleted"
        ]

    def list_tasks(self, agent: Any) -> list[dict[str, Any]]:
        """List all current task views for the caller's tree."""
        return [dict(task) for task in self._views(agent)]

    def get_task(self, agent: Any, identifier: str) -> dict[str, Any]:
        """Read one exact task view, or fail with ``TEAM_TASK_NOT_FOUND``."""
        state = self._state(agent)
        for task in state["tasks"]:
            if task["id"] == identifier:
                return project_task_view(state["tasks"], task)
        raise TeamError(f'team task "{identifier}" not found', "TEAM_TASK_NOT_FOUND")

    # -- mutations --------------------------------------------------------

    def _assert_enabled(self, agent: Any) -> None:
        if not self.is_enabled(agent):
            raise TeamError(
                "enable Agent Teams for this Session before using the task board",
                "TEAM_DISABLED",
            )

    def create_task(self, caller: Any, request: Mapping[str, Any]) -> dict[str, Any]:
        """Create one unowned pending task on the shared board."""
        self._assert_enabled(caller)
        subject = required_text(request.get("subject"), "subject", 200)
        description = required_text(request.get("description"), "description", 16_384)
        state = self._state(caller)
        active = sum(task["status"] != "deleted" for task in state["tasks"])
        if active >= self._max_tasks:
            raise TeamError(
                f"Team task limit {self._max_tasks} reached", "TEAM_TASK_LIMIT"
            )
        identifier = task_id(state["nextTaskNumber"])
        if any(task["id"] == identifier for task in state["tasks"]):
            raise TeamError("Team task id space exhausted", "TEAM_TASK_LIMIT")
        blocked_by = self._dependencies(
            request.get("blockedBy") or [], state["tasks"], None
        )
        scopes = self._write_scopes(request.get("writeScopes") or [])
        task = {
            "id": identifier,
            "revision": 1,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": blocked_by,
            "writeScopes": scopes,
        }
        assert_task_graph(state["tasks"], task)
        self._append_task(caller, task)
        return self.get_task(caller, identifier)

    def update_task(self, caller: Any, request: Mapping[str, Any]) -> dict[str, Any]:
        """Compare-and-set one task transition."""
        self._assert_enabled(caller)
        state = self._state(caller)
        identifier = request.get("taskId")
        current = next(
            (task for task in state["tasks"] if task["id"] == identifier), None
        )
        if current is None:
            raise TeamError(
                f'team task "{identifier}" not found', "TEAM_TASK_NOT_FOUND"
            )
        if current["revision"] != request.get("expectedRevision"):
            raise TeamError(
                f'team task "{identifier}" revision is stale; reread before updating',
                "TEAM_TASK_STALE_REVISION",
            )
        if current["status"] == "deleted":
            raise TeamError(f'team task "{identifier}" is deleted', "TEAM_TASK_DELETED")
        next_task = self._transition(caller, current, request, state)
        assert_task_graph(state["tasks"], next_task)
        self._append_task(caller, next_task)
        return self.get_task(caller, str(identifier))

    def _transition(
        self,
        caller: Any,
        current: dict[str, Any],
        request: Mapping[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        action = request.get("action")
        lead = self._is_lead(caller)
        owner = current.get("ownerId")
        caller_id = self._caller_id(caller)
        is_owner = owner is not None and owner == caller_id
        next_task = {**current, "revision": current["revision"] + 1}

        def authorize() -> None:
            if not lead and not is_owner:
                raise TeamError(
                    "task mutation requires its owner or Team Lead",
                    "TEAM_TASK_UNAUTHORIZED",
                )

        def without_owner(task: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in task.items() if k not in ("ownerId", "ownerName")}

        if action == "claim":
            if owner is not None and owner != caller_id:
                raise TeamError(
                    f'team task "{current["id"]}" is owned by another member',
                    "TEAM_TASK_ALREADY_CLAIMED",
                )
            if current["status"] != "pending":
                raise TeamError(
                    f'team task "{current["id"]}" is {current["status"]}; claim requires pending',
                    "TEAM_TASK_INVALID_TRANSITION",
                )
            if not task_ready(state["tasks"], current):
                raise TeamError(
                    f'team task "{current["id"]}" has unfinished dependencies',
                    "TEAM_TASK_BLOCKED",
                )
            names = self._names(caller)
            next_task.update(
                status="in_progress",
                ownerId=caller_id,
                **(
                    {}
                    if names.get("self") in (None, "root")
                    else {"ownerName": names["self"]}
                ),
            )
        elif action == "release":
            authorize()
            if current["status"] != "in_progress":
                raise TeamError(
                    "only an in-progress task can be released",
                    "TEAM_TASK_INVALID_TRANSITION",
                )
            next_task = without_owner({**next_task, "status": "pending"})
        elif action == "edit":
            authorize()
            if not any(
                key in request for key in ("subject", "description", "writeScopes")
            ):
                raise TeamError(
                    "task edit requires subject, description, or write_scopes",
                    "TEAM_INVALID_ARGUMENT",
                )
            if "subject" in request:
                next_task["subject"] = required_text(request["subject"], "subject", 200)
            if "description" in request:
                next_task["description"] = required_text(
                    request["description"], "description", 16_384
                )
            if "writeScopes" in request:
                next_task["writeScopes"] = self._write_scopes(request["writeScopes"])
        elif action == "set_dependencies":
            authorize()
            if "blockedBy" not in request:
                raise TeamError(
                    "set_dependencies requires blocked_by", "TEAM_INVALID_ARGUMENT"
                )
            next_task["blockedBy"] = self._dependencies(
                request["blockedBy"], state["tasks"], current["id"]
            )
        elif action == "complete":
            authorize()
            if current["status"] != "in_progress":
                raise TeamError(
                    "only an in-progress task can complete",
                    "TEAM_TASK_INVALID_TRANSITION",
                )
            next_task["status"] = "completed"
        elif action == "reopen":
            authorize()
            if current["status"] != "completed":
                raise TeamError(
                    "only a completed task can reopen",
                    "TEAM_TASK_INVALID_TRANSITION",
                )
            next_task = without_owner({**next_task, "status": "pending"})
        elif action == "reassign":
            if not lead:
                raise TeamError(
                    "only the Team Lead can reassign tasks", "TEAM_LEAD_REQUIRED"
                )
            if current["status"] not in ("pending", "in_progress"):
                raise TeamError(
                    "only a pending or in-progress task can be reassigned",
                    "TEAM_TASK_INVALID_TRANSITION",
                )
            target = request.get("owner")
            if target is None:
                next_task = without_owner({**next_task, "status": "pending"})
            else:
                if not task_ready(state["tasks"], current):
                    raise TeamError(
                        f'team task "{current["id"]}" is blocked', "TEAM_TASK_BLOCKED"
                    )
                owner_id, owner_name = self._resolve_assignee(caller, str(target))
                next_task = without_owner({**next_task, "status": "in_progress"})
                next_task["ownerId"] = owner_id
                next_task["ownerName"] = owner_name
        elif action == "delete":
            authorize()
            dependent = next(
                (
                    task
                    for task in state["tasks"]
                    if task["status"] != "deleted"
                    and task["id"] != current["id"]
                    and current["id"] in task["blockedBy"]
                ),
                None,
            )
            if dependent is not None:
                raise TeamError(
                    f'team task "{current["id"]}" still blocks "{dependent["id"]}"',
                    "TEAM_TASK_HAS_DEPENDENTS",
                )
            next_task["status"] = "deleted"
        else:
            raise TeamError(
                f"unsupported task action {action!r}", "TEAM_INVALID_ARGUMENT"
            )
        return next_task

    def _dependencies(
        self, values: Any, tasks: list[dict[str, Any]], self_id: str | None
    ) -> list[str]:
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise TeamError(
                "blockedBy must be an array of strings", "TEAM_INVALID_ARGUMENT"
            )
        seen: set[str] = set()
        result: list[str] = []
        for identifier in values:
            if identifier == self_id:
                raise TeamError(
                    "a team task cannot block itself",
                    "TEAM_TASK_DEPENDENCY_CYCLE",
                )
            if identifier in seen:
                raise TeamError(
                    f'duplicate blocker "{identifier}"', "TEAM_INVALID_ARGUMENT"
                )
            task = next((t for t in tasks if t["id"] == identifier), None)
            if task is None or task["status"] == "deleted":
                raise TeamError(
                    f'blocker task "{identifier}" not found', "TEAM_TASK_NOT_FOUND"
                )
            seen.add(identifier)
            result.append(identifier)
        return result

    @staticmethod
    def _write_scopes(values: Any) -> list[str]:
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise TeamError(
                "write_scopes must be an array of strings", "TEAM_INVALID_ARGUMENT"
            )
        return list(dict.fromkeys(write_scope(scope) for scope in values))

    def _resolve_assignee(self, caller: Any, target: str) -> tuple[str, str | None]:
        root_session = self._root_session(caller)
        if target == getattr(root_session, "session_id", None) or target == self._caller_id(
            caller
        ):
            return target, self._names(caller).get("self")
        subagents = self._subagents()
        if subagents is None:
            raise TeamError(
                f'agent "{target}" is not a continuable descendant of this task board\'s root',
                "TEAM_MEMBER_NOT_FOUND",
            )
        try:
            identifier = subagents.resolve_name(caller, target)
        except Exception as error:
            if getattr(error, "code", None) == "UNAUTHORIZED":
                raise TeamError(
                    f'agent "{target}" is outside the caller tree',
                    "TEAM_MEMBER_NOT_FOUND",
                ) from error
            raise TeamError(
                f'agent "{target}" is not a continuable descendant of this task board\'s root',
                "TEAM_MEMBER_NOT_FOUND",
            ) from error
        return identifier, target

    def _names(self, agent: Any) -> dict[str, Any]:
        subagents = self._subagents()
        if subagents is None:
            return {"self": "root"}
        return dict(subagents.names_of(agent))

    def _append_task(self, caller: Any, task: dict[str, Any]) -> None:
        self._session_of(caller).append(
            "team/task",
            {"version": 2, "teamId": self._state(caller)["id"], "task": task},
        )


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> TeamService:
    """Mount ``ctx.agentTeams`` (service entrypoint for the loader)."""
    return TeamService(ctx, config)


__all__ = ["DEFAULT_MAX_TASKS", "TeamService", "apply"]
