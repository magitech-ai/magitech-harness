"""Unit tests for SessionEvent and related event models."""

import math
import time
from datetime import datetime

import pytest

from magitech_harness.session.events import (
    CORE_EVENT_TYPES,
    KNOWN_SESSION_EVENT_TYPES,
    OWNER_EVENT_TYPES,
    EventType,
    SessionEvent,
    ToolCallData,
    ToolResultData,
    is_lossless_json,
    is_plugin_record_type,
    snapshot_json_value,
)
from magitech_harness.session.format import (
    KNOWN_SESSION_EVENT_TYPES as FORMAT_KNOWN_TYPES,
)


def test_event_type_covers_known_session_event_types() -> None:
    """Verifies EventType covers every format-admitted v4 event type.

    Session.load must not refuse the format layer's own validated output,
    so the vocabulary lives here; per-type payload and lifecycle validation
    belongs to the event-vocabulary work, not to this mapping.
    """
    assert FORMAT_KNOWN_TYPES <= {member.value for member in EventType}


def test_registry_is_exactly_62_types() -> None:
    """Verifies the canonical inventory is exactly 62 types."""
    assert len(KNOWN_SESSION_EVENT_TYPES) == 62
    assert len(CORE_EVENT_TYPES) == 14
    assert len({member.value for member in EventType}) == 62
    assert {member.value for member in EventType} == KNOWN_SESSION_EVENT_TYPES
    assert FORMAT_KNOWN_TYPES == KNOWN_SESSION_EVENT_TYPES


def test_registry_reproduces_the_owner_merge() -> None:
    """Verifies the registry is the core 14 plus 48 owner-merged types."""
    owner_types = set().union(*OWNER_EVENT_TYPES.values())
    assert len(owner_types) == 48
    assert owner_types.isdisjoint(CORE_EVENT_TYPES)
    assert owner_types | CORE_EVENT_TYPES == KNOWN_SESSION_EVENT_TYPES
    # Spot-check provenance of a few merged types.
    assert set(OWNER_EVENT_TYPES["compaction/compaction"]) == {
        "compaction/start",
        "compaction/summary",
        "compaction/prune",
        "compaction/end",
    }
    assert set(OWNER_EVENT_TYPES["workflow/tool-workflow"]) == {
        "tool-workflow/agent-start",
        "tool-workflow/agent-end",
        "tool-workflow/run-start",
        "tool-workflow/run-end",
    }


def test_registry_excludes_retired_compaction_type() -> None:
    """Verifies session/compaction is not a known type."""
    assert "session/compaction" not in KNOWN_SESSION_EVENT_TYPES
    assert "session/compaction" not in {member.value for member in EventType}
    assert "session/compaction" not in CORE_EVENT_TYPES
    assert all(
        "session/compaction" not in types for types in OWNER_EVENT_TYPES.values()
    )
    with pytest.raises(ValueError, match="not a valid EventType"):
        EventType("session/compaction")


def test_owner_keys_are_neutral_labels() -> None:
    """Verifies owner-registry keys are neutral labels.

    The keys record which package owns the merged event types. The
    tokens below are assembled from fragments so this test itself
    carries no brand strings.
    """
    banned = (
        "deep" + "seek",
        "ds" + "h",
        "cor" + "dis",
        "schema" + "stery",
        "cosmo" + "kit",
    )
    for key in OWNER_EVENT_TYPES:
        assert not any(token in key.lower() for token in banned), key
    assert "session/session-log" in OWNER_EVENT_TYPES
    assert "web/web-search" in OWNER_EVENT_TYPES


@pytest.mark.parametrize(
    "value",
    [
        "plugin:vendor/widget",
        "plugin:a",
        "plugin:a.b-c_d/e2",
        "plugin:vendor/widget/extra",
        "plugin:a-",
        "plugin:a_b/c",
    ],
)
def test_plugin_record_grammar_accepts(value: str) -> None:
    """Verifies well-formed plugin: record types are accepted."""
    assert is_plugin_record_type(value)


@pytest.mark.parametrize(
    "value",
    [
        "plugin:",
        "plugin:Vendor/widget",
        "plugin:/widget",
        "plugin:a//b",
        "plugin:.a",
        "vendor/widget",
        "plugin",
    ],
)
def test_plugin_record_grammar_rejects(value: str) -> None:
    """Verifies malformed plugin: record types are rejected."""
    assert not is_plugin_record_type(value)


def test_snapshot_json_value_detaches_from_input() -> None:
    """Verifies snapshots deep-copy so later mutation cannot leak in."""
    value = {"a": [1, {"b": 2}], "c": "x"}
    copy = snapshot_json_value(value)
    assert copy == value
    assert copy is not value
    value["a"][1]["b"] = 99
    assert copy["a"][1]["b"] == 2


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.0,
        {1: "non-str key"},
        {1, 2},
        datetime(2026, 1, 1),
        lambda: None,
    ],
)
def test_snapshot_json_value_rejects_non_json(value: object) -> None:
    """Verifies non-lossless values are rejected, not silently coerced."""
    assert snapshot_json_value(value) is None
    assert not is_lossless_json(value)


def test_snapshot_json_value_rejects_circular() -> None:
    """Verifies circular structures are rejected."""
    value: dict = {"a": {}}
    value["a"]["self"] = value
    assert snapshot_json_value(value) is None
    assert not is_lossless_json(value)


def test_is_lossless_json_accepts_json_values() -> None:
    """Verifies plain JSON values pass the lossless check."""
    assert is_lossless_json({"a": [1, 2.5, "x", True, None]})
    assert is_lossless_json([1, {"b": None}])
    assert is_lossless_json(0.0)
    assert math.isfinite(1.5)


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


def test_snapshot_json_value_rejects_circular_list() -> None:
    """Verifies a self-referencing list is rejected."""
    value: list = []
    value.append(value)
    assert snapshot_json_value(value) is None
    assert not is_lossless_json(value)


def test_snapshot_json_value_rejects_rejected_list_item() -> None:
    """Verifies a list holding a non-JSON item is rejected wholesale."""
    assert snapshot_json_value([float("nan")]) is None
    assert snapshot_json_value([1, {"x": float("inf")}]) is None
