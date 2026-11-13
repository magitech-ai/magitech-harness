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


def test_pairing_message_shaped_tool_result() -> None:
    """Verifies pairing matches a tool-role message result to its call."""
    events = [
        _event(
            EventType.ASSISTANT_MESSAGE,
            {
                "text": None,
                "tool_calls": [{"call_id": "c1", "name": "search", "arguments": {}}],
            },
        ),
        _event(
            EventType.TOOL_RESULT,
            {
                "message": {
                    "toolCallId": "c1",
                    "content": [{"type": "text", "text": "hit"}],
                }
            },
        ),
    ]
    # At index 0 (assistant called tool, result not yet included): unbalanced.
    assert is_tool_pairing_balanced(events, 0) is False
    # At index 1 (message-shaped tool result included): balanced.
    assert is_tool_pairing_balanced(events, 1) is True


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


def test_pairing_canonical_message_shape() -> None:
    """Verifies pairing reads tool-call blocks from canonical messages."""
    events = [
        _event(
            EventType.ASSISTANT_MESSAGE,
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "m1",
                    "role": "assistant",
                    "content": [
                        "not-a-block",
                        {"type": "text", "text": "working"},
                        {"type": "tool-call", "name": "nameless"},
                        {"type": "tool-call", "id": "", "name": "empty"},
                        {
                            "type": "tool-call",
                            "id": "c9",
                            "name": "search",
                            "arguments": "{}",
                        },
                    ],
                    "source": {"kind": "model", "provider": "p", "model": "m"},
                },
                "stream": [],
            },
        ),
        _event(
            EventType.TOOL_RESULT,
            {
                "message": {
                    "toolCallId": "c9",
                    "content": [{"type": "text", "text": "hit"}],
                }
            },
        ),
        _event(
            EventType.ASSISTANT_MESSAGE,
            {"message": {"content": "not-a-list"}},
        ),
    ]
    # At index 0 the canonical call c9 is unresolved: unbalanced.
    assert is_tool_pairing_balanced(events, 0) is False
    # At index 1 the message-shaped result resolves c9: balanced.
    assert is_tool_pairing_balanced(events, 1) is True
    # At index 2 the malformed message shape adds no calls: balanced.
    assert is_tool_pairing_balanced(events, 2) is True


def test_pairing_mixed_flat_and_canonical_calls() -> None:
    """Verifies flat and canonical calls pair in one log."""
    events = [
        _event(
            EventType.ASSISTANT_MESSAGE,
            {"tool_calls": [{"call_id": "c_flat", "name": "bash", "arguments": {}}]},
        ),
        _event(
            EventType.ASSISTANT_MESSAGE,
            {
                "message": {
                    "id": "m2",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool-call",
                            "id": "c_canon",
                            "name": "search",
                            "arguments": "{}",
                        }
                    ],
                    "source": {"kind": "model", "provider": "p", "model": "m"},
                },
            },
        ),
        _event(EventType.TOOL_RESULT, {"call_id": "c_flat", "content": "ok"}),
        _event(
            EventType.TOOL_RESULT,
            {"message": {"toolCallId": "c_canon", "content": []}},
        ),
    ]
    assert is_tool_pairing_balanced(events, 0) is False
    assert is_tool_pairing_balanced(events, 1) is False
    assert is_tool_pairing_balanced(events, 2) is False
    assert is_tool_pairing_balanced(events, 3) is True
