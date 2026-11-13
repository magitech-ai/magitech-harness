"""Crash-repair closers for open tail turns.

Synthetic error tool results plus step/turn closers balance a log whose
tail turn never ended. The fork seed reuses the ``forked`` cause; the
agent-loop resume path consumes the ``interrupted`` cause. Closed steps
and turns are preserved unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from magitech_harness.session.events import SessionEvent

TOOL_NOT_STARTED = "TOOL_NOT_STARTED"
"""Recovery code for a tool request that never reached a recorded call start."""

TOOL_OUTCOME_UNKNOWN = "TOOL_OUTCOME_UNKNOWN"
"""Recovery code for a recorded tool call whose outcome was not recorded."""

_CLOSER_TEXT: dict[str, dict[str, str]] = {
    "interrupted": {
        "started": (
            "The tool call was interrupted after it was recorded, but no result "
            "was durably recorded. Its outcome is unknown. Decide whether to "
            "retry from the tool semantics: retry only if the operation is "
            "read-only or idempotent; if it may have side effects, first verify "
            "external state or ask the user. Do not retry blindly."
        ),
        "not_started": (
            "The tool call was interrupted before the Harness recorded it as "
            "started. Retry it if it is still needed."
        ),
    },
    "forked": {
        "started": (
            "The history inherited by this branch records this tool call "
            "starting but does not include its result. The parent session may "
            "have completed it after the fork point. Decide whether to retry "
            "from the tool semantics: retry only if the operation is read-only "
            "or idempotent; if it may have side effects, first verify external "
            "state or ask the user. Do not retry blindly."
        ),
        "not_started": (
            "The history inherited by this branch has no record of this tool "
            "call starting. The parent session may have executed it after the "
            "fork point. Decide whether to retry from the tool semantics: "
            "retry only if the operation is read-only or idempotent; if it may "
            "have side effects, first verify external state or ask the user. "
            "Do not retry blindly."
        ),
    },
}

CloseCause = Literal["interrupted", "forked"]


def _tool_call_ids(event: SessionEvent) -> list[str]:
    """Tool-call ids an assistant message requests, in transcript order."""
    ids: list[str] = []
    message = event.payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool-call"
                    and isinstance(block.get("id"), str)
                ):
                    ids.append(block["id"])
    for call in event.payload.get("tool_calls") or []:
        if isinstance(call, dict):
            call_id = call.get("call_id") or call.get("id")
            if isinstance(call_id, str):
                ids.append(call_id)
    return ids


def _call_id_of(event: SessionEvent) -> str | None:
    """The call id a tool/call or tool/result event names, if any."""
    call_id = event.payload.get("callId") or event.payload.get("call_id")
    if isinstance(call_id, str):
        return call_id
    message = event.payload.get("message")
    if isinstance(message, dict):
        source = message.get("source")
        if isinstance(source, dict):
            call_id = source.get("callId")
            if isinstance(call_id, str):
                return call_id
        tool_call_id = message.get("toolCallId")
        if isinstance(tool_call_id, str):
            return tool_call_id
    return None


class ToolCallRecovery:
    """Track unanswered assistant tool requests from committed events."""

    def __init__(self, cause: CloseCause = "interrupted") -> None:
        self._cause = cause
        self._pending: dict[str, dict[str, Any]] = {}
        self._last: SessionEvent | None = None

    def observe(self, event: SessionEvent) -> None:
        """Consume the next committed event in sequence order."""
        self._last = event
        if event.type in ("turn/start", "turn/end", "step/end"):
            self._pending.clear()
        elif event.type == "assistant/message":
            for call_id in _tool_call_ids(event):
                self._pending[call_id] = {
                    "turn": event.payload.get("turn"),
                    "step": event.payload.get("step"),
                }
        elif event.type == "tool/call":
            call_ref = _call_id_of(event)
            entry = self._pending.get(call_ref) if call_ref else None
            if entry is not None:
                entry["call_seq"] = event.seq
        elif event.type == "tool/result":
            result_ref = _call_id_of(event)
            entry = self._pending.get(result_ref) if result_ref else None
            if (
                entry is not None
                and result_ref is not None
                and event.surface_op == "append"
                and entry.get("turn") == event.payload.get("turn")
                and entry.get("step") == event.payload.get("step")
            ):
                del self._pending[result_ref]

    def results(self) -> list[SessionEvent]:
        """Build conservative error results in assistant order.

        Sequences follow the latest observed event and timestamps reuse
        its time. Returns an empty list when no request remains unanswered.
        """
        if self._last is None:
            return []
        seq = self._last.seq + 1
        text = _CLOSER_TEXT[self._cause]
        session_id = self._last.session_id
        results: list[SessionEvent] = []
        for call_id, entry in self._pending.items():
            started = "call_seq" in entry
            message: dict[str, Any] = {
                "id": f"{self._cause}-tool-result-{call_id}-{seq}",
                "role": "tool",
                "toolCallId": call_id,
                "isError": True,
                "source": {"kind": "tool", "callId": call_id},
                "content": [
                    {
                        "type": "text",
                        "text": text["started"] if started else text["not_started"],
                    }
                ],
            }
            results.append(
                SessionEvent(
                    id=f"{session_id}:{seq}",
                    session_id=session_id,
                    type="tool/result",
                    timestamp=self._last.timestamp,
                    payload={
                        "turn": entry["turn"],
                        "step": entry["step"],
                        "message": message,
                        "error": {
                            "name": (
                                "ToolOutcomeUnknownError"
                                if started
                                else "ToolNotStartedError"
                            ),
                            "code": (
                                TOOL_OUTCOME_UNKNOWN if started else TOOL_NOT_STARTED
                            ),
                        },
                    },
                    seq=seq,
                    surface_op="append",
                    source_event_seqs=([entry["call_seq"]] if started else None),
                )
            )
            seq += 1
        return results


def open_turn_closers(
    events: list[SessionEvent], cause: CloseCause
) -> list[SessionEvent]:
    """Return synthetic events that close an open tail turn.

    Unmatched calls receive error results first, followed by an open
    ``step/end`` and a cause-specific ``turn/end``; sequences continue
    the log and timestamps reuse the last real event. A balanced or
    empty log returns no events.
    """
    open_turn: int | None = None
    open_step: int | None = None
    recovery = ToolCallRecovery(cause)
    for event in events:
        recovery.observe(event)
        if event.type == "turn/start":
            turn = event.payload.get("turn")
            open_turn = turn if isinstance(turn, int) else None
            open_step = None
        elif event.type == "turn/end":
            open_turn = None
            open_step = None
        elif event.type == "step/start":
            step = event.payload.get("step")
            open_step = step if isinstance(step, int) else None
        elif event.type == "step/end":
            open_step = None
    last = events[-1] if events else None
    if open_turn is None or last is None:
        return []
    closers = recovery.results()
    seq = last.seq + len(closers) + 1
    if open_step is not None:
        closers.append(
            SessionEvent(
                id=f"{last.session_id}:{seq}",
                session_id=last.session_id,
                type="step/end",
                timestamp=last.timestamp,
                payload={"turn": open_turn, "step": open_step},
                seq=seq,
            )
        )
        seq += 1
    closers.append(
        SessionEvent(
            id=f"{last.session_id}:{seq}",
            session_id=last.session_id,
            type="turn/end",
            timestamp=last.timestamp,
            payload={"turn": open_turn, "reason": {"kind": cause}},
            seq=seq,
        )
    )
    return closers


def interrupted_turn_closers(events: list[SessionEvent]) -> list[SessionEvent]:
    """Crash-recovery closers for a persisted log whose tail turn was interrupted."""
    return open_turn_closers(events, "interrupted")
