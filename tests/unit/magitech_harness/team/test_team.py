"""Tests for the Agent Teams domain and task tools."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.team import TeamError, TeamService
from magitech_harness.team.fold import (
    TEAM_PROJECTION,
    apply_event,
    apply_task,
    init_state,
    projection_view,
    validate_state,
)
from magitech_harness.team.task_graph import (
    assert_task_graph,
    project_task_view,
    task_ready,
)
from magitech_harness.team.tools import TEAM_POLICY, _list, _named, _update
from magitech_harness.team.tools import apply as apply_tools
from magitech_harness.team.types import (
    required_text,
    task_id,
    task_number,
    task_view,
    write_scope,
)


def _task(identifier: str, revision: int = 1, **extra: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "revision": revision,
        "subject": "subject",
        "description": "description",
        "status": "pending",
        "blockedBy": [],
        "writeScopes": [],
        **extra,
    }


def _empty_state(identifier: str = "session-1") -> dict[str, Any]:
    return init_state({"id": identifier}, 0)


# -- types / graph / fold (pure) -------------------------------------------


def test_types() -> None:
    assert required_text("  hi  ", "x", 10) == "hi"
    for value in (1, "   ", "abcd"):
        with pytest.raises(TeamError):
            required_text(value, "x", 3)
    assert write_scope(".\\src\\lib/") == "src/lib"
    assert write_scope("./a/b") == "a/b"
    for value in (1, "", "/abs", "C:/x", "a//b", "a/./b", "a/../b"):
        with pytest.raises(TeamError):
            write_scope(value)
    assert task_id(3) == "task-3"
    assert task_number("task-3") == 3
    assert task_number("abc") is None
    assert task_number("task-x") is None
    assert task_number("task-999999999999999999999") is None
    assert task_view(_task("t", ownerId="r", ownerName="n"))["ownerName"] == "n"
    assert "ownerId" not in task_view(_task("t"))


def test_graph() -> None:
    current = [_task("task-1", status="completed"), _task("task-2")]
    assert_task_graph(current, _task("task-3", blockedBy=["task-1"]))
    with pytest.raises(TeamError, match="cannot block itself"):
        assert_task_graph([], _task("task-1", blockedBy=["task-1"]))
    with pytest.raises(TeamError, match="repeats blocker"):
        assert_task_graph(
            [_task("task-1", status="completed")],
            _task("task-2", blockedBy=["task-1", "task-1"]),
        )
    with pytest.raises(TeamError, match="missing or deleted"):
        assert_task_graph([], _task("task-2", blockedBy=["task-9"]))
    with pytest.raises(TeamError, match="cycle"):
        assert_task_graph(
            [_task("task-1", blockedBy=["task-2"])],
            _task("task-2", blockedBy=["task-1"]),
        )

    tasks = [
        _task("task-1", status="completed"),
        _task("task-2", status="in_progress", writeScopes=["src"]),
    ]
    assert task_ready(tasks, _task("task-3", blockedBy=["task-1"])) is True
    assert task_ready(tasks, _task("task-3", blockedBy=["task-2"])) is False
    assert (
        project_task_view(tasks, _task("task-3", writeScopes=["docs"]))[
            "writeScopeWarnings"
        ]
        == []
    )
    assert any(
        "write scopes overlap with" in warning
        for warning in project_task_view(
            tasks, _task("task-4", writeScopes=["src/lib"])
        )["writeScopeWarnings"]
    )
    assert (
        project_task_view(
            tasks, _task("task-4", status="completed", writeScopes=["src/lib"])
        )["writeScopeWarnings"]
        == []
    )


def test_fold() -> None:
    from magitech_harness.team.fold import _validate_task, _validate_wire

    assert _validate_task(_task("task-1"))["id"] == "task-1"
    for value in (
        None,
        {**_task("task-1"), "id": 1},
        {**_task("task-1"), "revision": True},
        {**_task("task-1"), "subject": None},
        {**_task("task-1"), "status": "unknown"},
        {**_task("task-1"), "blockedBy": "no"},
        {**_task("task-1"), "writeScopes": [1]},
        {**_task("task-1"), "ownerId": ""},
        {**_task("task-1"), "ownerName": ""},
        {**_task("task-1"), "ownerName": "x"},
        {**_task("task-0")},
    ):
        with pytest.raises((ValueError, TypeError)):
            _validate_task(value)

    state = _empty_state()
    assert validate_state(state) == state
    for value in (
        None,
        {**state, "unknown": 1},
        {"id": "x"},
        {**state, "inheritedEventCount": -1},
        {**state, "id": ""},
        {**state, "tasks": "no"},
        {**state, "nextTaskNumber": 0},
        {**state, "enabled": "yes"},
        {**state, "failure": 1},
    ):
        with pytest.raises((ValueError, TypeError)):
            validate_state(value)

    mode = SimpleNamespace(type="team/mode", seq=0, payload={"enabled": True})
    assert apply_event(state, mode)["enabled"] is True
    inherited = {**state, "inheritedEventCount": 1}
    assert apply_event(inherited, mode) == inherited
    assert "failure" in apply_event(
        state, SimpleNamespace(type="team/mode", seq=0, payload={})
    )
    task_event = SimpleNamespace(
        type="team/task",
        seq=0,
        payload={"version": 2, "teamId": "session-1", "task": _task("task-1")},
    )
    result = apply_event(state, task_event)
    assert len(result["tasks"]) == 1
    assert result["nextTaskNumber"] == 2
    assert "failure" in apply_event(
        state,
        SimpleNamespace(
            type="team/task",
            seq=0,
            payload={"version": 1, "teamId": "session-1", "task": _task("task-1")},
        ),
    )
    assert (
        apply_event(
            state,
            SimpleNamespace(
                type="team/task",
                seq=0,
                payload={"version": 2, "teamId": "other", "task": _task("task-1")},
            ),
        )
        == state
    )
    failed = {**state, "failure": "prior"}
    assert apply_event(failed, task_event) == failed
    assert apply_event(state, SimpleNamespace(type="other", seq=0, payload={})) == state

    with pytest.raises(ValueError, match="must begin at revision 1"):
        apply_task(state, _task("task-1", revision=2))
    applied = apply_task(state, _task("task-1"))
    with pytest.raises(ValueError, match="not contiguous"):
        apply_task(applied, _task("task-1", revision=3))
    assert apply_task(applied, _task("task-9"))["nextTaskNumber"] == 10
    with pytest.raises(TeamError, match="cannot block itself"):
        apply_task(applied, _task("task-1", revision=2, blockedBy=["task-1"]))
    assert apply_task(applied, _task("custom"))["nextTaskNumber"] == 2

    view_state = {
        **state,
        "enabled": True,
        "tasks": [_task("task-1"), _task("task-2", status="deleted")],
    }
    view = projection_view(view_state)
    assert view["enabled"] is True and len(view["tasks"]) == 1 and "failure" not in view
    assert projection_view({**view_state, "failure": "boom"})["failure"] == "boom"
    assert _validate_wire(view) == view
    with pytest.raises((ValueError, TypeError)):
        _validate_wire({"enabled": True, "tasks": [], "extra": 1})
    assert TEAM_PROJECTION.key == "agentTeam"


# -- service ---------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Any) -> Any:
    return create_harness(storage_dir=tmp_path / "teams")


def _agent(ctx: Any, identifier: str = "root") -> Any:
    session = ctx.sessions.create(identifier)
    return SimpleNamespace(id=identifier, session=session, options={})


@pytest.mark.asyncio
async def test_board_is_shared_with_children(harness: Any) -> None:
    await harness.flush()
    service: TeamService = harness.agentTeams
    root = _agent(harness, "root-session")
    child_session = harness.sessions.create("child-session")
    child = SimpleNamespace(id="child-agent", session=child_session, options={})

    # Register the child in the live tree, parented at the root session.
    from magitech_harness.subagent.service import _Member

    harness.subagents._members["child-agent"] = _Member(
        child_id="child-agent",
        name="worker",
        parent_id="root-session",
        session_id="child-session",
        depth=1,
    )
    try:
        service.enable(root)
        assert service.is_enabled(child) is True
        service.create_task(root, {"subject": "s", "description": "d"})
        assert [task["id"] for task in service.list_tasks(child)] == ["task-1"]
        assert service.get_task(child, "task-1")["subject"] == "s"
        claimed = service.update_task(
            child, {"taskId": "task-1", "expectedRevision": 1, "action": "claim"}
        )
        assert (
            claimed["owner"]
            if "owner" in claimed
            else claimed.get("ownerId") is not None
        )
        assert service.list_tasks(root)[0]["status"] == "in_progress"
    finally:
        harness.subagents._members.clear()


@pytest.mark.asyncio
async def test_lead_authorization_uses_real_identity(harness: Any) -> None:
    await harness.flush()
    service: TeamService = harness.agentTeams
    root_agent = _agent(harness, "real-root")
    service.enable(root_agent)
    service.create_task(root_agent, {"subject": "s", "description": "d"})
    # id == session id, so exact lead identity is recognized.
    reassigned = service.update_task(
        root_agent,
        {
            "taskId": "task-1",
            "expectedRevision": 1,
            "action": "reassign",
            "owner": "real-root",
        },
    )
    assert reassigned["status"] == "in_progress"
    assert reassigned["ownerId"] == "real-root"


@pytest.mark.asyncio
async def test_task_lifecycle_and_guards(harness: Any) -> None:
    await harness.flush()
    service: TeamService = harness.agentTeams
    agent = _agent(harness)
    service.enable(agent)
    service.enable(agent)

    with pytest.raises(TeamError, match="subject"):
        service.create_task(agent, {"subject": "", "description": "d"})
    with pytest.raises(TeamError, match="description"):
        service.create_task(agent, {"subject": "s", "description": ""})
    with pytest.raises(TeamError, match="not found"):
        service.create_task(
            agent, {"subject": "s", "description": "d", "blockedBy": ["task-9"]}
        )

    created = service.create_task(
        agent,
        {
            "subject": " work ",
            "description": "details",
            "writeScopes": ["./src/", "src", "lib"],
        },
    )
    service.create_task(agent, {"subject": "other", "description": "d"})
    assert created["id"] == "task-1"
    assert created["ready"] is True
    assert created["writeScopes"] == ["src", "lib"]
    assert service.get_task(agent, "task-1")["subject"] == "work"
    with pytest.raises(TeamError, match="not found"):
        service.get_task(agent, "task-9")

    with pytest.raises(TeamError, match="not found"):
        service.update_task(
            agent, {"taskId": "task-9", "expectedRevision": 1, "action": "claim"}
        )
    with pytest.raises(TeamError, match="stale"):
        service.update_task(
            agent, {"taskId": "task-1", "expectedRevision": 9, "action": "claim"}
        )
    with pytest.raises(TeamError, match="only an in-progress task can complete"):
        service.update_task(
            agent, {"taskId": "task-1", "expectedRevision": 1, "action": "complete"}
        )
    with pytest.raises(TeamError, match="requires subject"):
        service.update_task(
            agent, {"taskId": "task-1", "expectedRevision": 1, "action": "edit"}
        )
    with pytest.raises(TeamError, match="requires blocked_by"):
        service.update_task(
            agent,
            {"taskId": "task-1", "expectedRevision": 1, "action": "set_dependencies"},
        )

    claimed = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 1, "action": "claim"}
    )
    assert claimed["status"] == "in_progress"
    with pytest.raises(TeamError, match="claim requires pending"):
        service.update_task(
            agent, {"taskId": "task-1", "expectedRevision": 2, "action": "claim"}
        )
    completed = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 2, "action": "complete"}
    )
    assert completed["status"] == "completed"
    with pytest.raises(TeamError, match="only an in-progress task can complete"):
        service.update_task(
            agent, {"taskId": "task-1", "expectedRevision": 3, "action": "complete"}
        )
    reopened = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 3, "action": "reopen"}
    )
    assert reopened["status"] == "pending"

    reclaimed = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 4, "action": "claim"}
    )
    assert reclaimed["status"] == "in_progress"
    edited = service.update_task(
        agent,
        {
            "taskId": "task-1",
            "expectedRevision": 5,
            "action": "edit",
            "subject": " new ",
            "description": "d2",
            "writeScopes": ["lib"],
        },
    )
    assert edited["subject"] == "new" and edited["writeScopes"] == ["lib"]
    deps = service.update_task(
        agent,
        {
            "taskId": "task-1",
            "expectedRevision": 6,
            "action": "set_dependencies",
            "blockedBy": [],
        },
    )
    assert deps["blockedBy"] == []
    dependent = service.create_task(
        agent, {"subject": "dep", "description": "d", "blockedBy": ["task-1"]}
    )
    with pytest.raises(TeamError, match="still blocks"):
        service.update_task(
            agent, {"taskId": "task-1", "expectedRevision": 7, "action": "delete"}
        )
    deleted = service.update_task(
        agent, {"taskId": dependent["id"], "expectedRevision": 1, "action": "delete"}
    )
    assert deleted["status"] == "deleted"
    assert dependent["id"] not in [task["id"] for task in service.list_tasks(agent)]
    deleted = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 7, "action": "delete"}
    )
    assert deleted["status"] == "deleted"
    with pytest.raises(TeamError, match="is deleted"):
        service.update_task(
            agent, {"taskId": "task-1", "expectedRevision": 8, "action": "claim"}
        )
    with pytest.raises(TeamError, match="unsupported task action"):
        service.update_task(
            agent, {"taskId": "task-2", "expectedRevision": 1, "action": "bogus"}
        )


@pytest.mark.asyncio
async def test_limits_and_config(harness: Any) -> None:
    await harness.flush()
    agent = _agent(harness)
    limited = TeamService(harness, {"maxTasks": 1})
    await limited.init()
    limited.enable(agent)
    limited.create_task(agent, {"subject": "s", "description": "d"})
    with pytest.raises(TeamError, match="limit 1 reached"):
        limited.create_task(agent, {"subject": "s2", "description": "d2"})

    bare = Context()
    service = TeamService(bare)
    with pytest.raises(RuntimeError, match="sessionProjections"):
        await service.init()
    with pytest.raises(RuntimeError, match="sessionProjections"):
        service.is_enabled(agent)
    with pytest.raises(TeamError, match="positive safe integer"):
        TeamService(bare, {"maxTasks": 0})
    with pytest.raises(TeamError, match="no longer supported"):
        TeamService(bare, {"maxMembers": 1})


@pytest.mark.asyncio
async def test_corrupt_board_and_id_exhaustion(harness: Any) -> None:
    await harness.flush()
    service: TeamService = harness.agentTeams
    agent = _agent(harness)
    service.enable(agent)

    original = service._state
    service._state = lambda a: {  # type: ignore[method-assign]
        "inheritedEventCount": 0,
        "id": "root",
        "enabled": True,
        "tasks": [_task("task-1")],
        "nextTaskNumber": 1,
        "failure": None,
    }
    with pytest.raises(TeamError, match="id space exhausted"):
        service.create_task(agent, {"subject": "s", "description": "d"})
    service._state = original  # type: ignore[method-assign]

    # A malformed durable event latches a projection failure, which reads surface.
    agent.session.append(
        "team/task",
        {"version": 99, "teamId": agent.session.session_id, "task": _task("task-9")},
    )
    with pytest.raises(TeamError, match="corrupt"):
        service.list_tasks(agent)


@pytest.mark.asyncio
async def test_default_enabled(harness: Any) -> None:
    await harness.flush()
    service = TeamService(harness, {"defaultEnabled": True})
    await service.init()
    agent = _agent(harness, "auto-root")
    harness.emit("agent/created", {"agent": agent})
    assert service.is_enabled(agent) is True


# -- tools -----------------------------------------------------------------


def test_owner_mapping_and_list_filters(harness: Any) -> None:
    assert _named(_task("t", ownerId="r", ownerName="n"))["owner"] == "n"
    assert "ownerId" not in _named(_task("t", ownerId="r"))
    assert "owner" not in _named(_task("t"))

    service = SimpleNamespace(
        list_tasks=lambda agent: [
            {"id": "task-1", "status": "pending", "ownerId": None, "ready": True},
            {
                "id": "task-2",
                "status": "completed",
                "ownerId": "worker-id",
                "ready": False,
            },
        ]
    )
    from magitech_harness.tools.schema import ToolArgsError

    ctx = SimpleNamespace(get=lambda name, default=None, strict=True: None)
    agent = object()
    assert _list(service, ctx, agent, {"limit": 1})["nextCursor"] == 1
    assert len(_list(service, ctx, agent, {"status": "completed"})["tasks"]) == 1
    assert len(_list(service, ctx, agent, {"unowned": True})["tasks"]) == 1
    assert len(_list(service, ctx, agent, {"ready": False})["tasks"]) == 1
    with pytest.raises(ToolArgsError):
        _list(service, ctx, agent, {"owner": "worker", "unowned": True})
    for args in ({"cursor": -1}, {"limit": 0}, {"limit": 101}):
        with pytest.raises(ToolArgsError):
            _list(service, ctx, agent, args)
    assert _list(service, ctx, agent, {"owner": "worker"})["tasks"] == []

    updated = SimpleNamespace(update_task=lambda a, req: req)
    assert (
        _update(
            updated, agent, {"task_id": "t", "expected_revision": 1, "action": "edit"}
        )["taskId"]
        == "t"
    )


def test_tool_apply_and_policy(tmp_path: Any) -> None:
    registered: list[Any] = []
    sections: list[Any] = []
    ctx = Context()
    ctx.provide("agentTeams", SimpleNamespace(is_enabled=lambda a: False))
    ctx.provide(
        "tools",
        type(
            "T",
            (),
            {
                "register": lambda self, definition, options=None: registered.append(
                    definition
                )
            },
        )(),
    )
    apply_tools(ctx, {})
    assert [definition["name"] for definition in registered] == [
        "team_task_create",
        "team_task_list",
        "team_task_get",
        "team_task_update",
    ]

    ctx2 = Context()
    registered2: list[Any] = []
    ctx2.provide(
        "tools",
        type(
            "T",
            (),
            {
                "register": lambda self, definition, options=None: registered2.append(
                    definition
                )
            },
        )(),
    )
    ctx2.provide("agentTeams", SimpleNamespace(is_enabled=lambda a: True))
    ctx2.provide(
        "systemPrompt",
        type(
            "SP",
            (),
            {
                "section": lambda self, section, sections=sections: sections.append(
                    section
                ),
                "get_section_order": lambda self, name: 1000,
            },
        )(),
    )
    apply_tools(ctx2, {})
    assert sections[0]["name"] == "team:policy"
    text = sections[0]["text"]
    assert text({"agent": object()}) == TEAM_POLICY
    assert text({}) == ""


# -- exhaustive branch coverage -------------------------------------------


class _FakeSubagents:
    def __init__(
        self, members: dict[str, Any] | None = None, fail_code: str | None = None
    ) -> None:
        self._members = members or {}
        self.fail_code = fail_code
        self.names = {"self": "worker", "parent": "root"}

    def names_of(self, agent: Any) -> dict[str, Any]:
        return self.names

    def resolve_name(self, caller: Any, name: str) -> str:
        from magitech_harness.subagent.service import SubagentError

        if self.fail_code:
            raise SubagentError("nope", self.fail_code)
        return "resolved-id"


@pytest.mark.asyncio
async def test_root_resolution_paths(harness: Any) -> None:
    await harness.flush()
    from magitech_harness.subagent.service import _Member

    service: TeamService = harness.agentTeams
    plain = _agent(harness, "plain")
    assert service._root_session(plain) is plain.session

    root_session = harness.sessions.create("deep-root")
    SimpleNamespace(id="deep-root", session=root_session, options={})
    child = SimpleNamespace(
        id="child", session=harness.sessions.create("child-s"), options={}
    )
    grandchild = SimpleNamespace(
        id="grand", session=harness.sessions.create("grand-s"), options={}
    )
    harness.subagents._members = {
        "child": _Member("child", "child-name", "deep-root", 1, session_id="child-s"),
        "grand": _Member("grand", "grand-name", "child-s", 2, session_id="grand-s"),
    }
    try:
        assert service._root_session(grandchild) is root_session
        assert service._root_session(child) is root_session
    finally:
        harness.subagents._members = {}

    # Unresolvable parent falls back to the caller's session.
    harness.subagents._members = {
        "child": _Member("child", "child-name", "gone", 1, session_id="child-s"),
    }
    try:
        assert service._root_session(child) is child.session
    finally:
        harness.subagents._members = {}

    assert service._session_of(plain) is plain.session
    assert service._is_lead(plain) is True
    assert service._is_lead(SimpleNamespace(id="other", session=plain.session)) is False


@pytest.mark.asyncio
async def test_enable_disable_and_authorization(harness: Any) -> None:
    await harness.flush()
    service: TeamService = harness.agentTeams
    agent = _agent(harness)
    service.enable(agent)
    service.disable(agent)
    service.disable(agent)
    assert service.is_enabled(agent) is False

    raw = TeamService.__new__(TeamService)
    raw.ctx = harness
    raw._handle = SimpleNamespace(
        state_of=lambda session: (_ for _ in ()).throw(TeamError("x", "UNAUTHORIZED"))
    )
    assert raw.is_enabled(agent) is False
    raw._handle = SimpleNamespace(
        state_of=lambda session: (_ for _ in ()).throw(TeamError("x", "OTHER"))
    )
    with pytest.raises(TeamError):
        raw.is_enabled(agent)

    service.enable(agent)
    service.create_task(agent, {"subject": "s", "description": "d"})


@pytest.mark.asyncio
async def test_transition_codes_and_assignee(harness: Any) -> None:
    await harness.flush()
    service: TeamService = harness.agentTeams
    agent = _agent(harness)
    service.enable(agent)
    service.create_task(agent, {"subject": "s", "description": "d", "blockedBy": []})
    child = SimpleNamespace(id="child", session=agent.session, options={})

    # Non-lead, non-owner cannot mutate.
    with pytest.raises(TeamError, match="owner or Team Lead"):
        service.update_task(
            child,
            {
                "taskId": "task-1",
                "expectedRevision": 1,
                "action": "edit",
                "subject": "x",
            },
        )
    with pytest.raises(TeamError, match="only the Team Lead"):
        service.update_task(
            child,
            {
                "taskId": "task-1",
                "expectedRevision": 1,
                "action": "reassign",
                "owner": "x",
            },
        )

    # Owner claims with a real name so ownerName is recorded.
    service._subagents = lambda: _FakeSubagents()  # type: ignore[method-assign]
    claimed = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 1, "action": "claim"}
    )
    assert claimed.get("ownerName") == "worker"
    with pytest.raises(TeamError, match="owned by another member"):
        service.update_task(
            child, {"taskId": "task-1", "expectedRevision": 2, "action": "claim"}
        )

    completed = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 2, "action": "complete"}
    )
    assert completed["status"] == "completed"
    with pytest.raises(TeamError, match="only a completed task can reopen"):
        other = service.create_task(agent, {"subject": "z", "description": "d"})
        service.update_task(
            agent, {"taskId": other["id"], "expectedRevision": 1, "action": "reopen"}
        )
    with pytest.raises(TeamError, match="pending or in-progress"):
        service.update_task(
            agent,
            {
                "taskId": "task-1",
                "expectedRevision": 3,
                "action": "reassign",
                "owner": "x",
            },
        )

    # Assignee resolution against a fake subagent tree.
    target = service.create_task(agent, {"subject": "target", "description": "d"})
    assigned = service.update_task(
        agent,
        {
            "taskId": target["id"],
            "expectedRevision": 1,
            "action": "reassign",
            "owner": "worker",
        },
    )
    assert assigned["ownerId"] == "resolved-id"
    assert assigned["ownerName"] == "worker"
    service._subagents = lambda: _FakeSubagents(fail_code="UNAUTHORIZED")  # type: ignore[method-assign]
    blocked = service.create_task(agent, {"subject": "blocked", "description": "d"})
    blocked_dep = service.create_task(
        agent, {"subject": "dep", "description": "d", "blockedBy": [blocked["id"]]}
    )
    del blocked_dep
    with pytest.raises(TeamError, match="outside the caller tree"):
        service.update_task(
            agent,
            {
                "taskId": blocked["id"],
                "expectedRevision": 1,
                "action": "reassign",
                "owner": "worker",
            },
        )
    service._subagents = lambda: _FakeSubagents(fail_code="OTHER")  # type: ignore[method-assign]
    with pytest.raises(TeamError, match="not a continuable descendant"):
        service.update_task(
            agent,
            {
                "taskId": blocked["id"],
                "expectedRevision": 1,
                "action": "reassign",
                "owner": "worker",
            },
        )
    service._subagents = lambda: None  # type: ignore[method-assign]
    with pytest.raises(TeamError, match="not a continuable descendant"):
        service.update_task(
            agent,
            {
                "taskId": blocked["id"],
                "expectedRevision": 1,
                "action": "reassign",
                "owner": "worker",
            },
        )


@pytest.mark.asyncio
async def test_dependency_and_scope_validation(harness: Any) -> None:
    await harness.flush()
    service: TeamService = harness.agentTeams
    agent = _agent(harness)
    service.enable(agent)
    first = service.create_task(agent, {"subject": "a", "description": "d"})
    with pytest.raises(TeamError, match="must be an array"):
        service.create_task(
            agent, {"subject": "b", "description": "d", "blockedBy": "no"}
        )
    with pytest.raises(TeamError, match="duplicate blocker"):
        service.create_task(
            agent,
            {
                "subject": "b",
                "description": "d",
                "blockedBy": [first["id"], first["id"]],
            },
        )
    with pytest.raises(TeamError, match="must be an array"):
        service.create_task(
            agent, {"subject": "b", "description": "d", "writeScopes": "no"}
        )
    with pytest.raises(TeamError, match="unfinished dependencies"):
        dep = service.create_task(
            agent, {"subject": "b", "description": "d", "blockedBy": [first["id"]]}
        )
        service.update_task(
            agent, {"taskId": dep["id"], "expectedRevision": 1, "action": "claim"}
        )
    with pytest.raises(TeamError, match="not found"):
        service.update_task(
            agent,
            {
                "taskId": first["id"],
                "expectedRevision": 1,
                "action": "set_dependencies",
                "blockedBy": ["task-9"],
            },
        )
    with pytest.raises(TeamError, match="cannot block itself"):
        service.update_task(
            agent,
            {
                "taskId": first["id"],
                "expectedRevision": 1,
                "action": "set_dependencies",
                "blockedBy": [first["id"]],
            },
        )
    second = service.create_task(agent, {"subject": "second", "description": "d"})
    with pytest.raises(TeamError, match="duplicate blocker"):
        service.update_task(
            agent,
            {
                "taskId": first["id"],
                "expectedRevision": 1,
                "action": "set_dependencies",
                "blockedBy": [second["id"], second["id"]],
            },
        )


def test_service_helpers_and_apply(harness: Any) -> None:
    from magitech_harness.team.service import apply as apply_entry

    assert isinstance(apply_entry(harness), TeamService)
    service: TeamService = harness.agentTeams
    assert TeamService._write_scopes(["src", "src", "lib"]) == ["src", "lib"]
    with pytest.raises(TeamError, match="must be an array"):
        TeamService._write_scopes("src")
    agent = _agent(harness, "helper")
    service._subagents = lambda: None  # type: ignore[method-assign]
    assert service._names(agent) == {"self": "root"}
    service._subagents = lambda: _FakeSubagents()  # type: ignore[method-assign]
    assert service._names(agent)["self"] == "worker"


@pytest.mark.asyncio
async def test_default_enabled_dispose(harness: Any) -> None:
    service = TeamService(harness, {"defaultEnabled": True})
    dispose = await service.init()
    harness.emit("agent/created", {"agent": None})
    agent = _agent(harness, "dispose-root")
    harness.emit("agent/created", {"agent": agent})
    assert service.is_enabled(agent) is True
    dispose()


@pytest.mark.asyncio
async def test_final_coverage_branches(harness: Any) -> None:
    await harness.flush()
    from magitech_harness.subagent.service import _Member
    from magitech_harness.team.tools import _render, _require_agent

    service: TeamService = harness.agentTeams
    agent = _agent(harness)

    # Disabled board rejects mutations.
    fresh = _agent(harness, "fresh")
    with pytest.raises(TeamError, match="enable Agent Teams"):
        service.create_task(fresh, {"subject": "s", "description": "d"})

    # Root resolution via a member whose parent_id is None and via a fake registry.
    harness.subagents._members = {
        "top": _Member("top", "top-name", None, 0, session_id="top-s"),
    }
    try:
        top = SimpleNamespace(
            id="top", session=harness.sessions.create("top-s"), options={}
        )
        assert service._root_session(top) is top.session
    finally:
        harness.subagents._members = {}

    fake = TeamService.__new__(TeamService)
    fake.ctx = SimpleNamespace(
        get=lambda name, default=None, strict=True: (
            SimpleNamespace(
                get=lambda identifier: SimpleNamespace(session="via-get"),
                list=lambda: [
                    SimpleNamespace(session=SimpleNamespace(session_id="via-list"))
                ],
            )
            if name == "agents"
            else SimpleNamespace(get=lambda identifier: None)
            if name == "sessions"
            else None
        ),
    )
    descendant = SimpleNamespace(
        id="d",
        session="ignored",
        options={},
    )
    fake._members = lambda: {  # type: ignore[method-assign]
        "d": SimpleNamespace(parent_id="p", session_id="d-s", child_id="d")
    }
    assert fake._root_session(descendant) == "via-get"
    fake.ctx = SimpleNamespace(
        get=lambda name, default=None, strict=True: (
            SimpleNamespace(
                get=lambda identifier: None,
                list=lambda: [SimpleNamespace(session=SimpleNamespace(session_id="p"))],
            )
            if name == "agents"
            else None
        ),
    )
    assert isinstance(fake._root_session(descendant), SimpleNamespace)
    fake.ctx = SimpleNamespace(get=lambda name, default=None, strict=True: None)
    assert fake._root_session(descendant) == "ignored"

    # Release, partial edits, reassign-unassign and blocked reassign.
    service.enable(agent)
    service.create_task(agent, {"subject": "s", "description": "d"})
    claimed = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 1, "action": "claim"}
    )
    assert claimed["status"] == "in_progress"
    released = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 2, "action": "release"}
    )
    assert released["status"] == "pending"
    service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 3, "action": "claim"}
    )
    service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 4, "action": "complete"}
    )
    service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 5, "action": "reopen"}
    )
    service.update_task(
        agent,
        {
            "taskId": "task-1",
            "expectedRevision": 6,
            "action": "edit",
            "subject": "only-subject",
        },
    )
    service.update_task(
        agent,
        {
            "taskId": "task-1",
            "expectedRevision": 7,
            "action": "edit",
            "description": "only-description",
        },
    )
    service.update_task(
        agent,
        {
            "taskId": "task-1",
            "expectedRevision": 8,
            "action": "edit",
            "writeScopes": ["only"],
        },
    )
    unassigned = service.update_task(
        agent, {"taskId": "task-1", "expectedRevision": 9, "action": "reassign"}
    )
    assert "ownerId" not in unassigned
    blocker = service.create_task(agent, {"subject": "blocker", "description": "d"})
    blocked = service.create_task(
        agent, {"subject": "blocked", "description": "d", "blockedBy": [blocker["id"]]}
    )
    with pytest.raises(TeamError, match="is blocked"):
        service.update_task(
            agent,
            {
                "taskId": blocked["id"],
                "expectedRevision": 1,
                "action": "reassign",
                "owner": "anyone",
            },
        )

    # Tool helpers and conditional admission errors.
    assert _render({}, {"tasks": []})[0]["text"] == '{"tasks": []}'
    assert _require_agent({"agent": "a"}) == "a"
    from magitech_harness.llm.error import HarnessError

    with pytest.raises(HarnessError):
        _require_agent({})
    with pytest.raises(HarnessError):
        _require_agent(None)
    assert _list(
        SimpleNamespace(list_tasks=lambda a: []),
        SimpleNamespace(get=lambda *a, **k: None),
        agent,
        {},
    ) == {"tasks": []}
    from magitech_harness.llm.error import HarnessError

    for code, should_raise in (("UNAUTHORIZED", False), ("OTHER", True)):
        SimpleNamespace(
            get=lambda name, default=None, strict=True: None,
        )
        raw = Context()
        raw.provide(
            "agentTeams",
            SimpleNamespace(
                is_enabled=lambda a, code=code: (_ for _ in ()).throw(
                    HarnessError("x", code)
                )
            ),
        )
        raw.provide(
            "tools",
            type("T", (), {"register": lambda self, definition, options=None: None})(),
        )
        apply_tools(raw, {})
        sections: list[Any] = []
        raw.provide(
            "systemPrompt",
            type(
                "SP",
                (),
                {
                    "section": lambda self, section, sections=sections: sections.append(
                        section
                    ),
                    "get_section_order": lambda self, name: 1000,
                },
            )(),
        )
        apply_tools(raw, {})
        text = sections[0]["text"]
        if should_raise:
            with pytest.raises(HarnessError):
                text({"agent": object()})
        else:
            assert text({"agent": object()}) == ""


def test_last_branches(harness: Any) -> None:
    from magitech_harness.team.fold import _validate_wire
    from magitech_harness.team.tools import _list

    with pytest.raises((ValueError, TypeError)):
        _validate_wire({"enabled": "x", "tasks": []})

    fake = TeamService.__new__(TeamService)
    fake.ctx = SimpleNamespace(
        get=lambda name, default=None, strict=True: (
            SimpleNamespace(
                get=lambda identifier: None,
                list=lambda: [SimpleNamespace(session=None)],
            )
            if name == "agents"
            else None
        ),
    )
    fake._members = lambda: {  # type: ignore[method-assign]
        "d": SimpleNamespace(parent_id="p", session_id="d-s", child_id="d")
    }
    assert fake._root_session(SimpleNamespace(id="d", session="fallback")) == "fallback"

    service: TeamService = harness.agentTeams
    agent = _agent(harness, "last-agent")
    service.enable(agent)
    service.create_task(agent, {"subject": "s", "description": "d"})
    with pytest.raises(TeamError, match="only an in-progress task can be released"):
        service.update_task(
            agent, {"taskId": "task-1", "expectedRevision": 1, "action": "release"}
        )

    ctx = SimpleNamespace(
        get=lambda name, default=None, strict=True: SimpleNamespace(
            resolve_name=lambda a, n: "resolved"
        )
    )
    svc = SimpleNamespace(
        list_tasks=lambda a: [
            {"id": "task-1", "status": "pending", "ownerId": "resolved", "ready": True}
        ]
    )
    assert len(_list(svc, ctx, agent, {"owner": "worker"})["tasks"]) == 1
