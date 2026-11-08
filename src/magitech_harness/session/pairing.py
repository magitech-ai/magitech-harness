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
                call_id = tc.get("call_id") or tc.get("id")
                if call_id:
                    pending_calls.add(call_id)
        elif ev.type == EventType.TOOL_RESULT:
            call_id = ev.payload.get("call_id")
            if call_id:
                pending_calls.discard(call_id)

    return len(pending_calls) == 0
