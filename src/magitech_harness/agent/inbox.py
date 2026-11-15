"""Durable agent inbox projection and command facade."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from magitech_harness.agent.contracts import InboxState, InboxTarget
from magitech_harness.agent.dispatch import AgentEventDispatch
from magitech_harness.llm.values import is_non_negative_safe_integer
from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    WireView,
)
from magitech_harness.session.store import Session


def _validate_inbox_list(value: Any, target: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"inbox.{target} must be a list")
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"inbox.{target} entries must be dicts")
        msg_id = item.get("id")
        if not isinstance(msg_id, str) or not msg_id:
            raise ValueError(f"inbox.{target} message id must be a non-empty string")
    return value


def _validate_inbox_state(value: Any) -> InboxState:
    if not isinstance(value, dict):
        raise TypeError("inbox state must be a dict")
    return {
        "next-turn": _validate_inbox_list(value.get("next-turn"), "next-turn"),
        "next-step": _validate_inbox_list(value.get("next-step"), "next-step"),
    }


def _init_inbox(_header: dict[str, Any], _inherited_event_count: int) -> InboxState:
    return {"next-turn": [], "next-step": []}


def _assert_unique_pending_ids(
    next_turn: Sequence[dict[str, Any]], next_step: Sequence[dict[str, Any]]
) -> None:
    seen: set[str] = set()
    for message in (*next_turn, *next_step):
        msg_id = message["id"]
        if msg_id in seen:
            raise ValueError(f'message "{msg_id}" is already pending')
        seen.add(msg_id)


def _apply_inbox(state: InboxState, event: SessionEvent) -> InboxState:
    if event.type != "agent/inbox/spliced":
        return state
    splice = event.payload
    try:
        target = splice["target"]
        inbox = state[target]
        start = splice["start"]
        removed_count = splice.get("removedCount", 0)
        if (
            not is_non_negative_safe_integer(start)
            or start > len(inbox)
            or not is_non_negative_safe_integer(removed_count)
            or start + removed_count > len(inbox)
        ):
            raise ValueError("invalid inbox splice")
        inserted = splice["inserted"]
        next_list = [*inbox[:start], *inserted, *inbox[start + removed_count :]]
        if target == "next-turn":
            _assert_unique_pending_ids(next_list, state["next-step"])
            return {"next-turn": next_list, "next-step": state["next-step"]}
        _assert_unique_pending_ids(state["next-turn"], next_list)
        return {"next-turn": state["next-turn"], "next-step": next_list}
    except Exception as err:
        raise ValueError(
            f"invalid persisted inbox splice at session seq {event.seq}"
        ) from err


inbox_projection_definition: ProjectionDefinition[InboxState] = ProjectionDefinition(
    key="inbox",
    state_version=1,
    state_schema=_validate_inbox_state,
    init=_init_inbox,
    apply=_apply_inbox,
    wire=WireView(
        view_schema=_validate_inbox_state,
        view=lambda state: state,
    ),
)


def _splice_start(length: int, start: float) -> int:
    if isinstance(start, float):
        if math.isnan(start):
            return 0
        if math.isinf(start):
            return length if start > 0 else 0
    offset = int(start)
    return max(length + offset, 0) if offset < 0 else min(offset, length)


def _splice_delete_count(remaining: int, delete_count: float) -> int:
    if isinstance(delete_count, float):
        if math.isnan(delete_count):
            return 0
        if math.isinf(delete_count):
            return remaining if delete_count > 0 else 0
    return min(max(int(delete_count), 0), remaining)


class AgentInbox:
    """Durable pending-work inbox backed by ``agent/inbox/spliced`` session events."""

    def __init__(
        self,
        read_state: Callable[[], InboxState],
        session: Session,
        dispatch: AgentEventDispatch,
    ) -> None:
        self._read_state = read_state
        self._session = session
        self._dispatch = dispatch

    @property
    def next_turn(self) -> list[dict[str, Any]]:
        """Prompts awaiting individual turns."""
        return self._read_state()["next-turn"]

    @property
    def next_step(self) -> list[dict[str, Any]]:
        """Input awaiting the next step boundary."""
        return self._read_state()["next-step"]

    @property
    def has_pending(self) -> bool:
        """Whether either pending-message list contains work."""
        state = self._read_state()
        return len(state["next-turn"]) > 0 or len(state["next-step"]) > 0

    def clear(self) -> None:
        """Durably cancel all pending input, clearing next-step before next-turn."""
        self.splice("next-step", 0, len(self.next_step), [])
        self.splice("next-turn", 0, len(self.next_turn), [])

    def claim(self, target: InboxTarget, turn: int) -> list[dict[str, Any]]:
        """Remove and return the batch proposed for one step without discarding."""
        claimed = self._mutate("next-step", 0, len(self.next_step), [], False)
        if target == "next-turn":
            claimed.extend(self._mutate("next-turn", 0, 1, [], False))
        for message in claimed:
            self._dispatch.emit(
                "agent/inbox/claimed", {"message": message, "turn": turn}
            )
        return claimed

    def append(self, target: InboxTarget, message: dict[str, Any]) -> None:
        """Append one message to a pending list."""
        self.splice(target, len(self._read_state()[target]), 0, [message])

    def prepend(self, target: InboxTarget, message: dict[str, Any]) -> None:
        """Prepend one message to a pending list."""
        self.splice(target, 0, 0, [message])

    def replace(self, message_id: str, new_message: dict[str, Any]) -> bool:
        """Replace one pending message in place across both lists."""
        location = self._locate(message_id)
        if location is None:
            return False
        target, index = location
        self.splice(target, index, 1, [new_message])
        return True

    def remove(self, message_id: str) -> bool:
        """Remove one pending message by identity across both lists."""
        location = self._locate(message_id)
        if location is None:
            return False
        target, index = location
        self.splice(target, index, 1, [])
        return True

    def splice(
        self,
        target: InboxTarget,
        start: float,
        delete_count: float,
        inserted: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply normalized splice semantics and durably record the mutation."""
        return self._mutate(target, start, delete_count, inserted, True)

    def _locate(self, message_id: str) -> tuple[InboxTarget, int] | None:
        state = self._read_state()
        for target in ("next-turn", "next-step"):
            for idx, message in enumerate(state[target]):
                if message["id"] == message_id:
                    return target, idx  # type: ignore[return-value]
        return None

    def _mutate(
        self,
        target: InboxTarget,
        start: float,
        delete_count: float,
        inserted: Sequence[dict[str, Any]],
        discard_removed: bool,
    ) -> list[dict[str, Any]]:
        state = self._read_state()
        inbox = state[target]
        actual_start = _splice_start(len(inbox), start)
        actual_delete = _splice_delete_count(len(inbox) - actual_start, delete_count)
        if actual_delete == 0 and len(inserted) == 0:
            return []
        candidate = [
            *inbox[:actual_start],
            *inserted,
            *inbox[actual_start + actual_delete :],
        ]
        if target == "next-turn":
            _assert_unique_pending_ids(candidate, state["next-step"])
        else:
            _assert_unique_pending_ids(state["next-turn"], candidate)
        removed = list(inbox[actual_start : actual_start + actual_delete])
        splice_payload: dict[str, Any] = {
            "target": target,
            "start": actual_start,
            "inserted": list(inserted),
        }
        if actual_delete > 0:
            splice_payload["removedCount"] = actual_delete
            if discard_removed:
                splice_payload["outcome"] = "canceled"
        event = self._session.append(EventType.AGENT_INBOX_SPLICED, splice_payload)
        if discard_removed:
            for message in removed:
                self._dispatch.emit("agent/inbox/discarded", {"message": message})
        for message in event.payload["inserted"]:
            self._dispatch.emit("agent/inbox/inserted", {"message": message})
        return removed


ReactLoopInbox = AgentInbox
