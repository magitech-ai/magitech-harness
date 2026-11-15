"""Unit tests for agent model selection projection, facade, and switch notice."""

from __future__ import annotations

from typing import Any

import pytest
from anima import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.agent.dispatch import agent_events
from magitech_harness.agent.model_selection import (
    AgentModel,
    install_model_selection,
    model_selection_projection_definition,
)
from magitech_harness.llm.message import create_user_message
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session

INITIAL = {"provider": "alpha", "model": "a0"}
INPUT = create_user_message(
    {"content": [{"type": "text", "text": "continue"}], "source": {"kind": "user"}}
)


class _StubAgent:
    def __init__(self, ctx: Context, session: Session, options: dict[str, Any]) -> None:
        self.id = session.id
        self.session = session
        self.ctx = ctx
        self.options = options
        self.status = "idle"


def _setup(
    options: dict[str, Any] | None = None,
) -> tuple[Context, _StubAgent, Session, AgentModel, Any]:
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    projection = ctx.sessionProjections.register(model_selection_projection_definition)
    session = ctx.sessions.create("model-selection")
    opts = dict(INITIAL) if options is None else options
    model = AgentModel(lambda: projection.state_of(session), session, opts)
    agent = _StubAgent(ctx, session, opts)
    return ctx, agent, session, model, lambda: projection.state_of(session)


async def _async_setup(
    options: dict[str, Any] | None = None,
) -> tuple[Context, _StubAgent, Session, AgentModel, Any]:
    ctx = Context()
    await ctx.use(SessionProjectionRegistry)
    await ctx.use(SessionService)
    projection = ctx.sessionProjections.register(model_selection_projection_definition)
    session = ctx.sessions.create("model-selection")
    opts = dict(INITIAL) if options is None else options
    model = AgentModel(lambda: projection.state_of(session), session, opts)
    agent = _StubAgent(ctx, session, opts)
    return ctx, agent, session, model, lambda: projection.state_of(session)


def _header(
    session: Session,
    config: dict[str, Any],
    adapter_defaults: dict[str, Any] | None = None,
) -> None:
    header: dict[str, Any] = {"config": config}
    if adapter_defaults is not None:
        header["adapterDefaults"] = adapter_defaults
    session.append(
        "request/header",
        {"header": header, "reason": "initial"},
    )


def _expected_notice(from_label: str, to_label: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "[model changed: assistant turns above this point were generated "
                    f"by {from_label}; the session continues with {to_label}]"
                ),
            }
        ],
        "source": {
            "kind": "model-selection",
            "form": "notice",
            "summary": f"{from_label} \u2192 {to_label}",
        },
    }


async def _pre_step(
    agent: _StubAgent,
    selection: dict[str, Any],
    *,
    messages: list[dict[str, Any]] | None = None,
    offered: list[dict[str, Any]] | None = None,
    step: int = 1,
    signal: AbortSignal | None = None,
) -> dict[str, Any]:
    final_msgs = [INPUT] if messages is None else messages
    final_offered = [INPUT] if offered is None else offered
    final_signal = AbortSignal() if signal is None else signal
    return await agent_events(agent.ctx, agent).waterfall(  # type: ignore[arg-type]
        "agent/pre-step",
        {
            "turn": 1,
            "step": step,
            "messages": final_offered,
            "signal": final_signal,
            "modelSelection": selection,
        },
        default_handler=lambda *_args: {
            "kind": "enter",
            "messages": final_msgs,
            "startsRequestSeries": True,
        },
    )


def test_initial_options_until_recorded_and_detached_values() -> None:
    ctx, agent, session, model, _state = _setup({})
    assert model.current() is None
    agent.options["provider"] = "alpha"
    assert model.current() is None
    agent.options["model"] = "a0"
    assert model.current() == INITIAL
    assert ctx.sessionProjections.snapshot(session)["values"]["modelSelection"] == {
        "lastUsed": None,
        "next": None,
    }

    model.select(INITIAL)
    assert len(session.events) == 1
    assert session.events[0].type == "model/selection"
    assert session.events[0].payload == INITIAL

    selected = {"provider": "beta", "model": "b1", "reasoningEffort": "high"}
    model.select(selected)
    selected["model"] = "mutated"
    read = model.current()
    assert read is not None
    read["model"] = "also mutated"
    assert model.current() == {
        "provider": "beta",
        "model": "b1",
        "reasoningEffort": "high",
    }

    current_snapshot = model.current()
    assert current_snapshot is not None
    model.select(current_snapshot)
    assert len(session.events) == 2
    agent.options["model"] = "other initial model"
    after = model.current()
    assert after is not None and after["model"] == "b1"


def test_retires_only_matching_pending_selection_and_reconstructs_from_log() -> None:
    ctx, _agent, session, _model, read_state = _setup()
    _header(session, INITIAL)
    selected = {"provider": "beta", "model": "b1", "reasoningEffort": "high"}
    _model = AgentModel(read_state, session, INITIAL)
    _model.select(selected)

    for cfg in (
        {**selected, "provider": "other"},
        {**selected, "model": "other"},
        {**selected, "reasoningEffort": "low"},
    ):
        _header(session, cfg)
        assert read_state()["pending"] == selected

    restored = Session.create("restored", list(session.events))
    projection = ctx.sessionProjections.register(model_selection_projection_definition)
    restored_model = AgentModel(lambda: projection.state_of(restored), restored, {})
    assert restored_model.current() == selected

    _header(session, selected)
    assert read_state() == {"lastUsed": selected, "pending": None}
    assert _model.current() == selected

    stable_state = read_state()
    _header(session, selected)
    assert read_state() is stable_state


def test_adapter_defaulted_reasoning_stays_unselected() -> None:
    ctx, _agent, session, model, read_state = _setup()
    selected = {"provider": "beta", "model": "b1"}
    model.select(selected)
    _header(session, {**selected, "reasoningEffort": "high"}, {"reasoningEffort": True})
    assert read_state() == {"lastUsed": selected, "pending": None}
    assert model.current() == selected
    assert ctx.sessionProjections.snapshot(session)["values"]["modelSelection"] == {
        "lastUsed": selected,
        "next": selected,
    }

    _header(session, {**selected, "reasoningEffort": "low"})
    assert model.current() == {**selected, "reasoningEffort": "low"}


def test_reuses_fold_state_for_duplicates_and_unrelated_events() -> None:
    _ctx, _agent, session, model, read_state = _setup()
    selected = {"provider": "beta", "model": "b1"}
    model.select(selected)
    state = read_state()
    session.append("model/selection", selected)
    session.append("user/message", INPUT, surface_op="append")
    assert read_state() is state


def test_replaces_complete_selection_and_clears_omitted_effort() -> None:
    _ctx, _agent, session, model, _state = _setup()
    model.select({**INITIAL, "reasoningEffort": "high"})
    model.select(INITIAL)
    assert model.current() == INITIAL
    assert len(session.events) == 2
    model.select(INITIAL)
    assert len(session.events) == 2


async def test_install_model_selection_announces_route_switch_and_repeats_until_header() -> (
    None
):
    ctx, agent, session, model, _state = await _async_setup()
    _header(session, INITIAL)
    install_model_selection(ctx)
    model.select({"provider": "alpha", "model": "future"})
    selected = {"provider": "beta", "model": "b1"}

    for _ in range(2):
        res = await _pre_step(agent, selected)
        assert res["startsRequestSeries"] is True
        assert res["messages"][0] == INPUT
        notice = res["messages"][1]
        exp = _expected_notice("alpha/a0", "beta/b1")
        assert notice["role"] == exp["role"]
        assert notice["content"] == exp["content"]
        assert notice["source"] == exp["source"]

    _header(session, selected)
    res_same = await _pre_step(agent, selected)
    assert res_same["messages"] == [INPUT]

    res_model_only = await _pre_step(agent, {"provider": "beta", "model": "b2"})
    notice2 = res_model_only["messages"][1]
    exp2 = _expected_notice("b1", "b2")
    assert notice2["content"] == exp2["content"]
    assert notice2["source"] == exp2["source"]


async def test_install_model_selection_skips_initial_same_route_effort_reject_abort_and_dispose() -> (
    None
):
    ctx, agent, session, _model, _state = await _async_setup()
    dispose = install_model_selection(ctx)

    # No header yet -> no notice
    res = await _pre_step(agent, INITIAL)
    assert res["messages"] == [INPUT]

    _header(session, INITIAL)
    assert (await _pre_step(agent, INITIAL))["messages"] == [INPUT]
    assert (await _pre_step(agent, {**INITIAL, "reasoningEffort": "high"}))[
        "messages"
    ] == [INPUT]

    selected = {"provider": "alpha", "model": "a1"}
    rejected = await agent_events(ctx, agent).waterfall(  # type: ignore[arg-type]
        "agent/pre-step",
        {
            "turn": 1,
            "step": 1,
            "messages": [],
            "signal": AbortSignal(),
            "modelSelection": selected,
        },
        default_handler=lambda *_args: {"kind": "reject"},
    )
    assert rejected == {"kind": "reject"}

    aborted = AbortSignal()
    aborted.abort("stop")
    assert (await _pre_step(agent, selected, signal=aborted))["messages"] == [INPUT]

    dispose()
    assert (await _pre_step(agent, selected))["messages"] == [INPUT]


async def test_install_model_selection_handles_empty_step_and_empty_tool_continuation() -> (
    None
):
    ctx, agent, session, _model, _state = await _async_setup()
    _header(session, INITIAL)
    install_model_selection(ctx)
    selected = {"provider": "alpha", "model": "a1"}

    assert (await _pre_step(agent, selected, messages=[]))["messages"] == []
    assert (await _pre_step(agent, selected, messages=[], step=2))["messages"] == []
    continuation = await _pre_step(agent, selected, messages=[], offered=[], step=2)
    assert len(continuation["messages"]) == 1
    assert (
        continuation["messages"][0]["content"]
        == _expected_notice("a0", "a1")["content"]
    )


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-dict",
        {"lastUsed": "bad", "pending": None},
        {"lastUsed": {"provider": "", "model": "m"}, "pending": None},
        {"lastUsed": {"provider": "p", "model": ""}, "pending": None},
        {
            "lastUsed": {"provider": "p", "model": "m", "reasoningEffort": ""},
            "pending": None,
        },
        {"lastUsed": None},
        {"pending": None},
        {"next": None},
    ],
)
def test_model_selection_schemas_reject_invalid_values(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        model_selection_projection_definition.state_schema(bad)
    assert model_selection_projection_definition.wire is not None
    with pytest.raises((TypeError, ValueError)):
        model_selection_projection_definition.wire.view_schema(bad)


def test_model_selection_schema_accepts_reasoning_effort() -> None:
    valid_with_effort = {
        "lastUsed": {"provider": "p", "model": "m", "reasoningEffort": "medium"},
        "pending": None,
    }
    assert (
        model_selection_projection_definition.state_schema(valid_with_effort)
        == valid_with_effort
    )
