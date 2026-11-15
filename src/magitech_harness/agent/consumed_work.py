"""Pure single-pass fold accounting for work consumed by an agent log."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from magitech_harness.session.events import SessionEvent


@dataclass(frozen=True)
class ConsumedWork:
    """Summary of consumed turns and unrun cancellations in a session suffix."""

    end: SessionEvent | None = None
    dropped_unrun: bool = False


def _accounts_for_claim(reason: Mapping[str, Any]) -> bool:
    """Whether a pre-step turn ending accounts for claimed inbox input."""
    return reason.get("kind") != "completed"


def fold_consumed_work(events: Sequence[SessionEvent]) -> ConsumedWork:
    """Fold an agent session log (or owned suffix) into consumed-work state."""
    stepped: set[int] = set()
    claimed: set[int] = set()
    open_turn: int | None = None
    end: SessionEvent | None = None
    dropped_unrun = False

    for event in events:
        if event.type == "turn/start":
            open_turn = event.payload["turn"]
        elif event.type == "step/start":
            stepped.add(event.payload["turn"])
        elif event.type == "agent/inbox/spliced":
            if event.payload.get("removedCount") is None:
                continue
            if event.payload.get("outcome") == "canceled":
                if len(event.payload.get("inserted", ())) == 0:
                    dropped_unrun = True
            elif open_turn is not None:
                claimed.add(open_turn)
        elif event.type == "turn/end":
            turn = event.payload["turn"]
            reason = event.payload["reason"]
            open_turn = None
            was_stepped = turn in stepped
            stepped.discard(turn)
            was_claimed = turn in claimed
            claimed.discard(turn)
            if was_stepped or (was_claimed and _accounts_for_claim(reason)):
                end = event
                dropped_unrun = False

    return ConsumedWork(end=end, dropped_unrun=dropped_unrun)
