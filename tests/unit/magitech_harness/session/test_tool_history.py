"""Unit tests for ToolHistoryProjection."""

import pytest

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.tool_history import ToolHistoryProjection, tool_name


def _tool(name: str, description: str = "d") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _header_event(
    seq: int, tools: list, reason: str = "initial", **extra
) -> SessionEvent:
    return SessionEvent(
        id=f"e:{seq}",
        session_id="s",
        type="request/header",
        timestamp=float(seq),
        payload={
            "header": {"config": {"provider": "p", "model": "m"}, "tools": tools},
            "reason": reason,
            **extra,
        },
        seq=seq,
    )


def _developer_event(
    seq: int, blocks: list, header_seq: int | None = None, message_id: str = "m1"
) -> SessionEvent:
    payload: dict = {
        "turn": 1,
        "step": 1,
        "message": {
            "id": message_id,
            "role": "developer",
            "content": blocks,
            "source": {"kind": "tool-registry"},
        },
    }
    if header_seq is not None:
        payload["headerSeq"] = header_seq
    return SessionEvent(
        id=f"e:{seq}",
        session_id="s",
        type="developer/message",
        timestamp=float(seq),
        payload=payload,
        seq=seq,
    )


def _other_event(seq: int) -> SessionEvent:
    return SessionEvent(
        id=f"e:{seq}",
        session_id="s",
        type="request/context",
        timestamp=float(seq),
        payload={"provider": "p", "model": "m"},
        seq=seq,
    )


def test_tool_name_reads_nested_function_name() -> None:
    assert tool_name(_tool("search")) == "search"
    with pytest.raises(ValueError, match="header tool without a name"):
        tool_name({"type": "function", "function": {"description": "d"}})
    with pytest.raises(ValueError, match="header tool without a name"):
        tool_name({"name": "flat"})


def test_first_header_starts_baseline_series() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a"), _tool("b")]))
    assert projection.snapshot() == {"tools": [_tool("a"), _tool("b")], "updates": []}


def test_non_baseline_header_leaves_history_untouched() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    projection.apply(_header_event(1, [_tool("a")], reason="change"))
    assert projection.snapshot() == {"tools": [_tool("a")], "updates": []}


def test_series_reason_and_starts_series_rebaseline() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    projection.apply(
        _developer_event(1, [{"type": "tool-addition", "toolName": "a"}], header_seq=0)
    )
    assert len(projection.snapshot()["updates"]) == 1
    projection.apply(_header_event(2, [_tool("a")], reason="series"))
    assert projection.snapshot() == {"tools": [_tool("a")], "updates": []}
    projection.apply(
        _developer_event(3, [{"type": "tool-addition", "toolName": "a"}], header_seq=2)
    )
    assert len(projection.snapshot()["updates"]) == 1
    projection.apply(_header_event(4, [_tool("a")], reason="change", startsSeries=True))
    assert projection.snapshot() == {"tools": [_tool("a")], "updates": []}


def test_redeclare_with_different_definition_rebaselines() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a", "one")]))
    projection.apply(_header_event(1, [_tool("a", "two")], reason="change"))
    assert projection.snapshot() == {"tools": [_tool("a", "two")], "updates": []}


def test_redeclare_with_identical_definition_keeps_series() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    projection.apply(
        _developer_event(1, [{"type": "tool-addition", "toolName": "a"}], header_seq=0)
    )
    projection.apply(_header_event(2, [_tool("a")], reason="change"))
    snapshot = projection.snapshot()
    assert len(snapshot["updates"]) == 1
    assert snapshot["tools"] == [_tool("a")]


def test_developer_addition_resolves_from_named_header() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    projection.apply(_header_event(1, [_tool("a"), _tool("b")], reason="change"))
    projection.apply(
        _developer_event(2, [{"type": "tool-addition", "toolName": "b"}], header_seq=1)
    )
    snapshot = projection.snapshot()
    assert snapshot["updates"] == [{"messageId": "m1", "additions": [_tool("b")]}]


def test_developer_addition_missing_definition_throws() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    with pytest.raises(ValueError, match="tool history: missing definition for nope"):
        projection.apply(
            _developer_event(
                1, [{"type": "tool-addition", "toolName": "nope"}], header_seq=0
            )
        )


def test_developer_addition_without_header_seq_throws() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    with pytest.raises(ValueError, match="tool history: missing definition for a"):
        projection.apply(
            _developer_event(1, [{"type": "tool-addition", "toolName": "a"}])
        )


def test_developer_without_additions_records_empty_update() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    projection.apply(
        _developer_event(1, [{"type": "text", "text": "note"}], message_id="m9")
    )
    assert projection.snapshot()["updates"] == [{"messageId": "m9", "additions": []}]


def test_removal_updates_available_set_and_snapshot_falls_back() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a"), _tool("b")]))
    projection.apply(_developer_event(1, [{"type": "tool-removal", "toolName": "b"}]))
    # Active tools (a, b) disagree with available (a): snapshot falls back
    # to the active header's tools with no updates.
    assert projection.snapshot() == {
        "tools": [_tool("a"), _tool("b")],
        "updates": [],
    }


def test_snapshot_falls_back_when_tool_added_without_record() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    projection.apply(_header_event(1, [_tool("a"), _tool("b")], reason="change"))
    assert projection.snapshot() == {
        "tools": [_tool("a"), _tool("b")],
        "updates": [],
    }


def test_snapshot_is_detached_copy() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    snapshot = projection.snapshot()
    snapshot["tools"][0]["function"]["name"] = "mutated"
    snapshot["updates"].append({"messageId": "x", "additions": []})
    assert projection.snapshot() == {"tools": [_tool("a")], "updates": []}


def test_other_event_types_are_ignored() -> None:
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    projection.apply(_other_event(1))
    assert projection.snapshot() == {"tools": [_tool("a")], "updates": []}


def test_header_tool_without_name_raises() -> None:
    projection = ToolHistoryProjection()
    with pytest.raises(ValueError, match="header tool without a name"):
        projection.apply(_header_event(0, [{"type": "function", "function": {}}]))


def test_header_tool_without_dict_raises_type_error() -> None:
    """Verifies a non-record header tool is a type error, not a name miss."""
    projection = ToolHistoryProjection()
    with pytest.raises(TypeError, match="header tool without a name"):
        projection.apply(_header_event(0, ["not-a-record"]))


def test_developer_non_dict_blocks_are_skipped() -> None:
    """Verifies non-record blocks in developer content are ignored."""
    projection = ToolHistoryProjection()
    projection.apply(_header_event(0, [_tool("a")]))
    projection.apply(
        _developer_event(
            1,
            [{"type": "tool-addition", "toolName": "a"}, 42, "text"],
            header_seq=0,
        )
    )
    assert projection.snapshot() == {
        "tools": [_tool("a")],
        "updates": [{"messageId": "m1", "additions": [_tool("a")]}],
    }
