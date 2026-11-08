"""Unit tests for tool pairing boundary verification."""

from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.pairing import is_tool_pairing_balanced


def _event(ev_type: EventType, payload: dict) -> SessionEvent:
    return SessionEvent(
        id="ev_test",
        session_id="sess_1",
        type=ev_type,
        payload=payload,
    )


def test_pairing_balanced_no_tools() -> None:
    events = [
        _event(EventType.USER_MESSAGE, {"text": "hello"}),
        _event(EventType.ASSISTANT_MESSAGE, {"text": "hi"}),
    ]
    assert is_tool_pairing_balanced(events, 0) is True
    assert is_tool_pairing_balanced(events, 1) is True


def test_pairing_unbalanced_severed_tool_call() -> None:
    events = [
        _event(EventType.USER_MESSAGE, {"text": "run command"}),
        _event(
            EventType.ASSISTANT_MESSAGE,
            {
                "text": None,
                "tool_calls": [{"call_id": "call_1", "name": "bash", "arguments": {}}],
            },
        ),
        _event(EventType.TOOL_RESULT, {"call_id": "call_1", "content": "output"}),
        _event(EventType.ASSISTANT_MESSAGE, {"text": "done"}),
    ]
    # At index 0 (user message): balanced (no tool calls)
    assert is_tool_pairing_balanced(events, 0) is True
    # At index 1 (assistant called tool, result not yet included): unbalanced
    assert is_tool_pairing_balanced(events, 1) is False
    # At index 2 (tool result included): balanced
    assert is_tool_pairing_balanced(events, 2) is True
    # At index 3 (final assistant message): balanced
    assert is_tool_pairing_balanced(events, 3) is True


def test_pairing_multiple_tool_calls() -> None:
    events = [
        _event(
            EventType.ASSISTANT_MESSAGE,
            {
                "text": None,
                "tool_calls": [
                    {"call_id": "c1", "name": "f1", "arguments": {}},
                    {"call_id": "c2", "name": "f2", "arguments": {}},
                ],
            },
        ),
        _event(EventType.TOOL_RESULT, {"call_id": "c1", "content": "res1"}),
        _event(EventType.TOOL_RESULT, {"call_id": "c2", "content": "res2"}),
    ]
    # At index 0: both c1 and c2 are pending -> False
    assert is_tool_pairing_balanced(events, 0) is False
    # At index 1: c2 is still pending -> False
    assert is_tool_pairing_balanced(events, 1) is False
    # At index 2: both c1 and c2 answered -> True
    assert is_tool_pairing_balanced(events, 2) is True


def test_pairing_bounds() -> None:
    events = [_event(EventType.USER_MESSAGE, {"text": "hi"})]
    # Negative index
    assert is_tool_pairing_balanced(events, -1) is True
    # Index greater than length
    assert is_tool_pairing_balanced(events, 10) is True


def test_pairing_edge_cases_empty_call_id_and_untracked_result() -> None:
    """Verifies branches for tool calls missing call_id and results not in pending."""
    from magitech_harness.session.events import ToolCallData

    events = [
        # Tool call missing both 'call_id' and 'id'
        _event(
            EventType.ASSISTANT_MESSAGE,
            {"text": None, "tool_calls": [{"name": "no_id"}]},
        ),
        # ToolCallData instance
        _event(
            EventType.ASSISTANT_MESSAGE,
            {
                "text": None,
                "tool_calls": [
                    ToolCallData(call_id="c_obj", name="test", arguments={})
                ],
            },
        ),
        # Tool result resolving c_obj
        _event(EventType.TOOL_RESULT, {"call_id": "c_obj", "content": "res"}),
        # Tool result with unknown call_id not in pending_calls
        _event(EventType.TOOL_RESULT, {"call_id": "orphan_result", "content": "123"}),
        # Tool result without call_id
        _event(EventType.TOOL_RESULT, {"content": "no_call_id"}),
    ]
    # Should evaluate cleanly as balanced
    assert is_tool_pairing_balanced(events, 4) is True
