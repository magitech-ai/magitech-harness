"""Unit tests for fork seed construction (build_fork_seed)."""

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.fork import build_fork_seed


def _event(
    seq: int,
    type: str,
    payload: dict | None = None,
    **kwargs: object,
) -> SessionEvent:
    return SessionEvent(
        id=f"s:{seq}",
        session_id="s",
        type=type,
        timestamp=1000.0 + seq,
        payload=payload or {},
        seq=seq,
        **kwargs,  # type: ignore[arg-type]
    )


def _closed_log() -> list[SessionEvent]:
    return [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _event(
            2,
            "user/message",
            {"role": "user", "id": "m2", "content": [{"type": "text", "text": "hi"}]},
            surface_op="append",
        ),
        _event(3, "step/end", {"turn": 1, "step": 1}),
        _event(4, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        _event(5, "turn/start", {"turn": 2}),
        _event(6, "step/start", {"turn": 2, "step": 1}),
        _event(
            7,
            "user/message",
            {"role": "user", "id": "m7", "content": [{"type": "text", "text": "go"}]},
            surface_op="append",
        ),
    ]


def test_build_fork_seed_mid_log_boundary() -> None:
    """Verifies a mid-log fork keeps the exact prefix plus a tagged seed."""
    events = _closed_log()
    seed = build_fork_seed(events, 4)
    # Prefix [0..4] unchanged, then the tagged end-seed at the cut.
    assert seed[:5] == events[:5]
    marker = seed[5]
    assert marker.type == "session/end-seed"
    assert marker.payload == {"inherited": True}
    assert marker.seq == 5
    assert marker.timestamp == events[4].timestamp
    # Turn 1 was closed, so no synthetic closers follow.
    assert len(seed) == 6


def test_build_fork_seed_open_turn_gets_synthetic_closers() -> None:
    """Verifies an open turn at the boundary is closed by forked closers."""
    events = _closed_log()
    seed = build_fork_seed(events, 7)
    assert seed[:8] == events[:8]
    marker = seed[8]
    assert marker.type == "session/end-seed"
    assert marker.payload == {"inherited": True}
    assert marker.seq == 8
    closers = seed[9:]
    assert [c.type for c in closers] == ["step/end", "turn/end"]
    assert closers[1].payload["reason"] == {"kind": "forked"}
    assert [c.seq for c in closers] == [9, 10]


def test_build_fork_seed_closed_steps_and_turns_preserved() -> None:
    """Verifies closed steps/turns survive the fork unchanged."""
    events = _closed_log()
    seed = build_fork_seed(events, 4)
    assert seed[3].type == "step/end"
    assert seed[3].payload == {"turn": 1, "step": 1}
    assert seed[4].type == "turn/end"
    assert seed[4].payload == {"turn": 1, "reason": {"kind": "completed"}}
    # The child owns the marker: its seq is outside the inherited prefix.
    assert seed[5].seq == 5


def test_build_fork_seed_boundary_validation() -> None:
    """Verifies boundary must be a valid inclusive source seq."""
    events = _closed_log()
    import pytest

    with pytest.raises(ValueError, match="must be between 0 and"):
        build_fork_seed(events, -1)
    with pytest.raises(ValueError, match="must be between 0 and"):
        build_fork_seed(events, 8)
    with pytest.raises(ValueError, match="must be between 0 and"):
        build_fork_seed([], 0)
