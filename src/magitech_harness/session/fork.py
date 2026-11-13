"""Fork seed construction over an exact source-event prefix."""

from __future__ import annotations

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.repair import open_turn_closers


def build_fork_seed(events: list[SessionEvent], boundary: int) -> list[SessionEvent]:
    """Copy an inclusive event prefix, mark its inherited cut, close its tail.

    Copies the exact prefix ``[0..boundary]``, appends a tagged
    ``session/end-seed {inherited: true}`` at ``boundary + 1`` (child-owned,
    outside the inherited count of ``boundary + 1``), then appends
    synthetic closers for the open tail. Closed steps and turns are
    preserved unchanged. The caller validates the boundary.
    """
    if not isinstance(boundary, int) or boundary < 0 or boundary >= len(events):
        raise ValueError(
            f"fork boundary {boundary} must be between 0 and {len(events) - 1}"
        )
    prefix = list(events[: boundary + 1])
    boundary_event = events[boundary]
    prefix.append(
        SessionEvent(
            id=f"{boundary_event.session_id}:{boundary + 1}",
            session_id=boundary_event.session_id,
            type="session/end-seed",
            timestamp=boundary_event.timestamp,
            payload={"inherited": True},
            seq=boundary + 1,
        )
    )
    return prefix + open_turn_closers(prefix, "forked")
