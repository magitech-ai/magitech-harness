"""
Tool pairing boundary verification for safe compaction cuts.
"""

from __future__ import annotations

from magitech_harness.session.events import EventType, SessionEvent


def is_tool_pairing_balanced(events: list[SessionEvent], cut_index: int) -> bool:
    """Verifies that cutting event history at cut_index does not sever an assistant

    tool call from its corresponding tool result.

    Returns True if every tool call in events[:cut_index + 1] has a matching
    tool result also within events[:cut_index + 1].
    """
    if cut_index < 0:
        return True

    slice_events = events[: cut_index + 1]
    pending_calls: set[str] = set()

    for ev in slice_events:
        if ev.type == EventType.ASSISTANT_MESSAGE:
            tool_calls = ev.payload.get("tool_calls", [])
            for tc in tool_calls:
                call_id = (
                    tc.get("call_id") or tc.get("id")
                    if isinstance(tc, dict)
                    else getattr(tc, "call_id", None)
                )
                if call_id:
                    pending_calls.add(call_id)
        elif ev.type == EventType.TOOL_RESULT:
            message = ev.payload.get("message")
            if isinstance(message, dict):
                # Admission-shaped result: the call id lives on the
                # tool-role message.
                call_id = message.get("toolCallId")
            else:
                # Legacy flat shape, kept readable for older logs.
                call_id = ev.payload.get("call_id")
            if call_id:
                pending_calls.discard(call_id)

    return len(pending_calls) == 0
