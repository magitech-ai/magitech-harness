"""Tests for the plan-mode domain and tools."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.context.user_questions import Answer, UserQuestionError
from magitech_harness.plan.fold import PLAN_PROJECTION, apply_event
from magitech_harness.plan.service import (
    APPROVE_LABEL,
    EXIT_PLAN_MODE,
    KEEP_PLANNING_LABEL,
    PlanModeController,
    resolve_config,
)
from magitech_harness.plan.tools import PLAN_SECTION, _first_heading
from magitech_harness.plan.tools import apply as apply_tools
from magitech_harness.plan.types import (
    PlanError,
    init_state,
    pending_view,
    validate_state,
    validate_wire,
)
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import WireView

SECTION = "Explore first, then present a plan."


def _event(event_type: str, payload: dict[str, Any]) -> SessionEvent:
    return SessionEvent(id="e", session_id="s", type=event_type, payload=payload, seq=0)


def test_configuration() -> None:
    assert resolve_config({"section": SECTION}) == SECTION
    for cfg in ({}, {"section": 1}, {"section": "  "}, {"section": SECTION, "x": 1}):
        with pytest.raises(PlanError):
            resolve_config(cfg)


def test_types_and_fold() -> None:
    state = init_state()
    assert state == {"active": False, "wanted": None, "running": None, "entered": False}
    assert pending_view(state) == {"active": False, "pending": False}
    assert pending_view({**state, "wanted": True}) == {"active": False, "pending": True}
    assert pending_view({**state, "running": {"commandId": "c", "wanted": True}}) == {
        "active": False,
        "pending": True,
    }
    assert validate_state(state) == state
    for value in (
        None,
        {},
        {**state, "active": "yes"},
        {**state, "wanted": "yes"},
        {**state, "running": {}},
        {**state, "running": {"commandId": "c", "wanted": "yes"}},
    ):
        with pytest.raises((ValueError, TypeError)):
            validate_state(value)
    wire = {"active": False, "pending": False}
    assert validate_wire(wire) == wire
    for value in ({}, {"active": "x", "pending": False}):
        with pytest.raises((ValueError, TypeError)):
            validate_wire(value)

    run = apply_event(
        state, _event("command/run", {"name": "plan", "args": "on", "commandId": "c"})
    )
    assert run["running"]["wanted"] is True
    assert apply_event(state, _event("command/run", {"name": "plan"})) == state
    assert (
        apply_event(state, _event("command/run", {"name": "other", "args": "x"}))
        == state
    )
    done = apply_event(
        run, _event("command/done", {"commandId": "c", "kind": "success"})
    )
    assert done["wanted"] is True and done["running"] is None
    keep = apply_event(
        {**run, "running": {"commandId": "c", "wanted": False}},
        _event("command/done", {"commandId": "c", "kind": "success"}),
    )
    assert keep["wanted"] is None
    assert (
        apply_event(
            run, _event("command/done", {"commandId": "other", "kind": "success"})
        )
        == run
    )
    mode = apply_event(state, _event("plan/mode", {"active": True}))
    assert mode["active"] is True and mode["entered"] is True
    assert apply_event(state, _event("plan/mode", {"active": "yes"})) == state
    assert apply_event(state, _event("other", {})) == state
    assert PLAN_PROJECTION.key == "plan"
    assert isinstance(PLAN_PROJECTION.wire, WireView)


@pytest.fixture
def harness(tmp_path: Any) -> Any:
    return create_harness(storage_dir=tmp_path / "plan")


def _agent(ctx: Any, identifier: str = "root") -> Any:
    return SimpleNamespace(
        id=identifier,
        session=ctx.sessions.create(identifier),
        options={},
    )


@pytest.mark.asyncio
async def test_controller_state_and_transitions(harness: Any) -> None:
    await harness.flush()
    service: PlanModeController = harness.planMode
    agent = _agent(harness)

    assert service.logged_active(agent) is False
    assert service.entered(agent) is False
    assert service.selected_active(agent) is False
    assert service.set(agent, False) == "noop"

    assert service.set(agent, True) in ("committed", "queued")
    assert service.selected_active(agent) is True
    # Accept a step so the staged selection is committed.
    decision = {"kind": "enter", "messages": []}
    wrapped = service._on_pre_step(
        {"agent": agent, "signal": AbortSignal()}, lambda: decision
    )
    result = await wrapped if hasattr(wrapped, "__await__") else wrapped
    assert result["kind"] == "enter"
    assert service.logged_active(agent) is True
    assert service.entered(agent) is True

    assert service.set(agent, False) in ("committed", "queued", "cancelled")
    await service._on_pre_step(
        {"agent": agent, "signal": AbortSignal()}, lambda: decision
    )
    assert service.logged_active(agent) is False
    assert service.entered(agent) is True


@pytest.mark.asyncio
async def test_controller_pre_step_edges(harness: Any) -> None:
    await harness.flush()
    service: PlanModeController = harness.planMode
    agent = _agent(harness)
    decision = {"kind": "enter", "messages": []}

    rejected = service._on_pre_step(
        {"agent": agent, "signal": AbortSignal()}, lambda: {"kind": "reject"}
    )
    assert (await rejected)["kind"] == "reject"

    aborted = AbortSignal()
    aborted.abort()
    assert (
        await service._on_pre_step(
            {"agent": agent, "signal": aborted}, lambda: decision
        )
        == decision
    )
    assert await service._on_pre_step({}, lambda: decision) == decision
    assert await service._on_pre_step({"agent": agent}, lambda: decision) == decision

    service._pending[id(agent.session)] = {"active": True, "narrate": False}
    assert await service._on_pre_step({"agent": agent}, lambda: decision) == decision

    service._pending[id(agent.session)] = {"active": True, "narrate": True}
    result = await service._on_pre_step({"agent": agent}, lambda: decision)
    assert len(result["messages"]) == 1

    # A failed append leaves the decision unchanged.
    service._pending[id(agent.session)] = {"active": True, "narrate": True}
    agent.session.append = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    assert await service._on_pre_step({"agent": agent}, lambda: decision) == decision


@pytest.mark.asyncio
async def test_prompt_sections_and_dispose(harness: Any) -> None:
    await harness.flush()
    service: PlanModeController = harness.planMode
    agent = _agent(harness)
    assert service._policy_text({}) == ""
    assert service._policy_text({"agent": agent}) == ""
    assert service._status_text({}) == ""
    assert service._status_text({"agent": agent}) == ""

    service._pending[id(agent.session)] = {"active": True, "narrate": True}
    assert service.section in service._policy_text({"agent": agent})
    assert "active" in service._status_text({"agent": agent})
    service._pending.clear()
    service.set(agent, True)
    await service._on_pre_step(
        {"agent": agent}, lambda: {"kind": "enter", "messages": []}
    )
    assert "active" in service._status_text({"agent": agent})
    service.set(agent, False)
    await service._on_pre_step(
        {"agent": agent}, lambda: {"kind": "enter", "messages": []}
    )
    assert "inactive" in service._status_text({"agent": agent})

    bare = Context()
    bare.provide("sessionProjections", harness.sessionProjections)
    fresh = PlanModeController(bare, {"section": SECTION})
    dispose = await fresh.init()
    dispose()
    with pytest.raises(PlanError, match="not ready"):
        fresh.logged_active(agent)


@pytest.mark.asyncio
async def test_open_turn_detection() -> None:
    bare = Context()
    service = PlanModeController(bare, {"section": SECTION})
    agent = SimpleNamespace(session=SimpleNamespace())
    assert service._in_open_turn(agent.session) is False
    bare.provide(
        "agents",
        SimpleNamespace(turn_boundary_of=lambda session: {"openTurnStartSeq": 1}),
    )
    assert service._in_open_turn(agent.session) is True
    bare.provide(
        "agents",
        SimpleNamespace(turn_boundary_of=lambda session: {"openTurnStartSeq": None}),
    )
    assert service._in_open_turn(agent.session) is False
    bare.provide(
        "agents",
        SimpleNamespace(
            turn_boundary_of=lambda session: (_ for _ in ()).throw(ValueError())
        ),
    )
    assert service._in_open_turn(agent.session) is False
    bare.provide("agents", SimpleNamespace())
    assert service._in_open_turn(agent.session) is False


# -- tools -----------------------------------------------------------------


def test_first_heading_and_apply() -> None:
    assert _first_heading("intro\n## The Plan\nbody") == "The Plan"
    assert _first_heading("no heading") is None

    registered: list[Any] = []
    sections: list[Any] = []
    ctx = Context()
    ctx.provide("planMode", SimpleNamespace())
    ctx.provide(
        "tools",
        type(
            "T",
            (),
            {
                "register": lambda self, definition, options=None: registered.append(
                    (definition, options)
                )
            },
        )(),
    )
    apply_tools(ctx, {})
    assert [definition["name"] for definition, _ in registered] == [
        "plan",
        EXIT_PLAN_MODE,
    ]
    assert registered[1][1]["when"]({"agent": None}) is False

    ctx2 = Context()
    ctx2.provide("planMode", SimpleNamespace())
    ctx2.provide(
        "tools",
        type("T", (), {"register": lambda self, definition, options=None: None})(),
    )
    ctx2.provide(
        "systemPrompt",
        type(
            "SP",
            (),
            {
                "section": lambda self, section: sections.append(section),
                "get_section_order": lambda self, name: 1000,
            },
        )(),
    )
    apply_tools(ctx2, {})
    assert sections[0]["name"] == "tool:plan"
    assert PLAN_SECTION


@pytest.mark.asyncio
async def test_plan_tool(harness: Any) -> None:
    from magitech_harness.plan.tools import _plan

    service: PlanModeController = harness.planMode
    with pytest.raises(PlanError, match="calling agent"):
        _plan(service, {"action": "on"}, {})
    agent = _agent(harness)
    from magitech_harness.tools.schema import ToolArgsError

    with pytest.raises(ToolArgsError):
        _plan(service, {"action": "bad"}, {"agent": agent})
    assert _plan(service, {"action": "on"}, {"agent": agent})["status"] in (
        "committed",
        "queued",
    )


@pytest.mark.asyncio
async def test_exit_tool(harness: Any) -> None:
    from magitech_harness.plan.tools import _exit

    service: PlanModeController = harness.planMode
    agent = _agent(harness)
    ctx = harness
    with pytest.raises(PlanError, match="calling agent"):
        await _exit(ctx, service, {"plan": "# P"}, {})
    with pytest.raises(PlanError, match="only available in plan mode"):
        await _exit(ctx, service, {"plan": "# P"}, {"agent": agent})
    service.set(agent, True)
    await service._on_pre_step(
        {"agent": agent}, lambda: {"kind": "enter", "messages": []}
    )
    from magitech_harness.tools.schema import ToolArgsError

    with pytest.raises(ToolArgsError):
        await _exit(ctx, service, {"plan": "no heading"}, {"agent": agent})
    bare = Context()
    with pytest.raises(PlanError, match="no user-questions channel"):
        await _exit(bare, service, {"plan": "# P"}, {"agent": agent})

    answers = [Answer(id="plan-review", selected=[APPROVE_LABEL])]
    ctx.userQuestions.register_answerer(lambda pending, signal: answers)
    concluded: list[Any] = []
    result = await _exit(
        ctx,
        service,
        {"plan": "# P"},
        {"agent": agent, "callId": "c1", "concludeTurn": lambda: concluded.append(1)},
    )
    assert result == {"approved": True}

    # Keep planning, with and without feedback.
    agent.session.append("plan/mode", {"active": True})
    ctx.userQuestions._answerers[-1] = lambda pending, signal: [
        Answer(id="plan-review", selected=[KEEP_PLANNING_LABEL])
    ]
    with pytest.raises(PlanError, match="keep planning"):
        await _exit(ctx, service, {"plan": "# P"}, {"agent": agent})
    ctx.userQuestions._answerers[-1] = lambda pending, signal: [
        Answer(id="plan-review", selected=[], custom="try again")
    ]
    with pytest.raises(PlanError, match="try again"):
        await _exit(ctx, service, {"plan": "# P"}, {"agent": agent})

    # Cancelled review concludes the turn.
    ctx.userQuestions._answerers[-1] = lambda pending, signal: (_ for _ in ()).throw(
        UserQuestionError("cancelled", "ASK_CANCELLED")
    )
    assert await _exit(
        ctx,
        service,
        {"plan": "# P"},
        {"agent": agent, "concludeTurn": lambda: concluded.append(2)},
    ) == {"approved": False}
    assert len(concluded) == 1

    # Other question errors propagate.
    ctx.userQuestions._answerers[-1] = lambda pending, signal: (_ for _ in ()).throw(
        UserQuestionError("bad", "OTHER")
    )
    with pytest.raises(UserQuestionError):
        await _exit(ctx, service, {"plan": "# P"}, {"agent": agent})

    # A disposed service refuses to commit an approval.
    ctx.userQuestions._answerers[-1] = lambda pending, signal: answers
    service._disposed = True
    with pytest.raises(PlanError, match="reloaded"):
        await _exit(ctx, service, {"plan": "# P"}, {"agent": agent})


@pytest.mark.asyncio
async def test_final_coverage_branches(harness: Any) -> None:
    from magitech_harness.plan.service import apply as apply_service

    bare = Context()
    bad = PlanModeController(bare, {"section": SECTION})
    with pytest.raises(RuntimeError, match="sessionProjections"):
        await bad.init()

    ctx2 = Context()
    ctx2.provide(
        "sessionProjections",
        SimpleNamespace(
            register=lambda definition: SimpleNamespace(dispose=lambda: None)
        ),
    )
    service = PlanModeController(ctx2, {"section": SECTION})
    dispose = await service.init()
    service._handle = None
    dispose()
    live: PlanModeController = harness.planMode
    agent = _agent(harness, "commit-root")
    live._pending[id(agent.session)] = {"active": True, "narrate": False}
    assert live.set(agent, False) == "committed"
    assert isinstance(apply_service(harness, {"section": SECTION}), PlanModeController)


@pytest.mark.asyncio
async def test_exit_cancelled_without_conclude(harness: Any) -> None:
    from magitech_harness.plan.tools import _exit

    service: PlanModeController = harness.planMode
    agent = _agent(harness, "cancel-root")
    agent.session.append("plan/mode", {"active": True})
    harness.userQuestions.register_answerer(
        lambda pending, signal: (_ for _ in ()).throw(
            UserQuestionError("cancelled", "ASK_CANCELLED")
        )
    )
    assert await _exit(harness, service, {"plan": "# P"}, {"agent": agent}) == {
        "approved": False
    }
