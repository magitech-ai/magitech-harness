"""Unit tests for SessionEvent and related event models."""

import time

from magitech_harness.session.events import (
    EventType,
    SessionEvent,
    ToolCallData,
    ToolResultData,
)
from magitech_harness.session.format import KNOWN_SESSION_EVENT_TYPES


def test_event_type_covers_known_session_event_types() -> None:
    """Verifies EventType covers every format-admitted v4 event type.

    Session.load must not refuse the format layer's own validated output,
    so the vocabulary lives here; per-type payload and lifecycle validation
    belongs to the event-vocabulary work, not to this mapping.
    """
    assert KNOWN_SESSION_EVENT_TYPES <= {member.value for member in EventType}


def test_tool_call_data_serialization() -> None:
    """Verifies ToolCallData serialization."""
    tc = ToolCallData(call_id="call_1", name="bash", arguments={"command": "ls"})
    dump = tc.model_dump()
    assert dump["call_id"] == "call_1"
    assert dump["name"] == "bash"
    assert dump["arguments"] == {"command": "ls"}


def test_tool_result_data_serialization() -> None:
    """Verifies ToolResultData serialization and error flags."""
    res = ToolResultData(
        call_id="call_1", name="bash", content="output", is_error=False
    )
    assert not res.is_error
    assert res.content == "output"


def test_session_event_defaults() -> None:
    """Verifies SessionEvent default timestamp and fields."""
    now = time.time()
    event = SessionEvent(
        id="evt_1",
        session_id="sess_1",
        type=EventType.USER_MESSAGE,
        payload={"text": "hello"},
    )
    assert event.id == "evt_1"
    assert event.session_id == "sess_1"
    assert event.type == EventType.USER_MESSAGE
    assert event.payload == {"text": "hello"}
    assert not event.ignorable
    assert event.timestamp >= now
