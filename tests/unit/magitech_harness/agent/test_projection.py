"""Unit tests for turnBoundary projection definition."""

from __future__ import annotations

import pytest
from anima import Context

from magitech_harness.agent.projection import turn_boundary_projection_definition
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.store import Session


def test_turn_boundary_projection_folds_turns_and_steps_and_validates_state() -> None:
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    handle = ctx.sessionProjections.register(turn_boundary_projection_definition)
    session = Session.create("tb-1")

    initial = handle.state_of(session)
    assert initial == {
        "openTurnStartSeq": None,
        "lastStepStartSeq": None,
        "lastStepBoundary": None,
        "lastTurn": 0,
    }
    assert turn_boundary_projection_definition.wire is None
    assert turn_boundary_projection_definition.state_version == 2

    # Unrelated event preserves identity
    session.append(
        "user/message",
        {"content": [{"type": "text", "text": "hi"}], "source": {"kind": "user"}},
        surface_op="append",
    )
    assert handle.state_of(session) is initial

    turn = session.append("turn/start", {"turn": 3})
    step = session.append("step/start", {"turn": 3, "step": 1})
    assert handle.state_of(session) == {
        "openTurnStartSeq": turn.seq,
        "lastStepStartSeq": step.seq,
        "lastStepBoundary": {"kind": "start", "seq": step.seq},
        "lastTurn": 3,
    }

    step_end = session.append("step/end", {"turn": 3, "step": 1})
    assert handle.state_of(session) == {
        "openTurnStartSeq": turn.seq,
        "lastStepStartSeq": step.seq,
        "lastStepBoundary": {"kind": "end", "seq": step_end.seq},
        "lastTurn": 3,
    }

    session.append("turn/end", {"turn": 3, "reason": {"kind": "completed"}})
    assert handle.state_of(session) == {
        "openTurnStartSeq": None,
        "lastStepStartSeq": step.seq,
        "lastStepBoundary": {"kind": "end", "seq": step_end.seq},
        "lastTurn": 3,
    }


@pytest.mark.parametrize(
    "bad_state",
    [
        "not-a-dict",
        {"lastTurn": 0},
        {"openTurnStartSeq": None, "lastTurn": 0},
        {"openTurnStartSeq": None, "lastStepStartSeq": None, "lastTurn": 0},
        {
            "openTurnStartSeq": True,
            "lastStepStartSeq": None,
            "lastStepBoundary": None,
            "lastTurn": 0,
        },
        {
            "openTurnStartSeq": -1,
            "lastStepStartSeq": None,
            "lastStepBoundary": None,
            "lastTurn": 0,
        },
        {
            "openTurnStartSeq": None,
            "lastStepStartSeq": "bad",
            "lastStepBoundary": None,
            "lastTurn": 0,
        },
        {
            "openTurnStartSeq": None,
            "lastStepStartSeq": None,
            "lastStepBoundary": "bad",
            "lastTurn": 0,
        },
        {
            "openTurnStartSeq": None,
            "lastStepStartSeq": None,
            "lastStepBoundary": {"kind": "mid", "seq": 1},
            "lastTurn": 0,
        },
        {
            "openTurnStartSeq": None,
            "lastStepStartSeq": None,
            "lastStepBoundary": {"kind": "start", "seq": None},
            "lastTurn": 0,
        },
        {
            "openTurnStartSeq": None,
            "lastStepStartSeq": None,
            "lastStepBoundary": {"kind": "start", "seq": -1},
            "lastTurn": 0,
        },
        {
            "openTurnStartSeq": None,
            "lastStepStartSeq": None,
            "lastStepBoundary": None,
            "lastTurn": -1,
        },
    ],
)
def test_turn_boundary_state_schema_rejects_invalid_values(bad_state: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        turn_boundary_projection_definition.state_schema(bad_state)


def test_turn_boundary_state_schema_accepts_valid_value() -> None:
    valid = {
        "openTurnStartSeq": 0,
        "lastStepStartSeq": 1,
        "lastStepBoundary": {"kind": "end", "seq": 2},
        "lastTurn": 1,
    }
    assert turn_boundary_projection_definition.state_schema(valid) == valid
