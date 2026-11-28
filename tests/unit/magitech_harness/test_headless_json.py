"""Behavioral probes for the reference-shaped headless NDJSON projection."""

from __future__ import annotations

import json
from typing import Any

from magitech_harness.headless_json import (
    MAX_DEPTH,
    HeadlessJsonProjection,
    add_usage,
    bound_json_event,
    bound_json_line,
    parse_arguments,
    result_text,
    truncate_utf8,
)


def test_truncate_utf8_respects_byte_boundaries() -> None:
    assert truncate_utf8("héllo", 3) == "hé"
    assert truncate_utf8("abc", 10) == "abc"


def test_bound_json_event_flags_cut_strings_keys_and_depth() -> None:
    assert bound_json_event({"a": "x"}, 4) == {"a": "x"}
    assert bound_json_event({"a": "xxxxx"}, 4) == {"a": "xxxx", "truncated": True}
    assert bound_json_event({"abcdef": 1}, 4) == {"abcd": 1, "truncated": True}
    deep_list: Any = "leaf"
    for _ in range(MAX_DEPTH + 1):
        deep_list = [deep_list]
    assert bound_json_event({"v": deep_list})["truncated"] is True
    deep_dict: Any = "leaf"
    for _ in range(MAX_DEPTH + 1):
        deep_dict = {"n": deep_dict}
    assert bound_json_event({"v": deep_dict})["truncated"] is True


def test_bound_json_line_applies_both_caps() -> None:
    assert bound_json_line({"type": "x"}) == '{"type":"x"}'
    short = bound_json_line(
        {"type": "session", "blob": {"x": "y" * 100}},
        max_string_bytes=20,
        max_event_bytes=60,
    )
    assert json.loads(short)["truncated"] is True
    assert "blob" not in json.loads(short)
    fallback = bound_json_line(
        {"type": "t" * 40, "blob": {"x": 1}},
        max_string_bytes=40,
        max_event_bytes=20,
    )
    assert json.loads(fallback)["truncated"] is True


def test_parse_arguments_keeps_text_and_rejects_non_finite() -> None:
    assert parse_arguments({"a": 1}) == {"a": 1}
    assert parse_arguments(7) == 7
    assert parse_arguments("") == {}
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("not json") == "not json"
    assert parse_arguments('{"a": Infinity}') == '{"a": Infinity}'


def test_result_text_joins_only_text_blocks() -> None:
    assert result_text(None) == ""
    assert result_text("raw") == "raw"
    assert (
        result_text(
            [
                {"type": "text", "text": "a"},
                {"type": "image"},
                "bad",
                {"type": "text", "text": "b"},
            ]
        )
        == "ab"
    )


def test_add_usage_accumulates_and_reports_incomplete_samples() -> None:
    assert add_usage(None, None) == (None, False)
    assert add_usage({"a": 1}, None) == ({"a": 1}, True)
    assert add_usage({"a": 1}, "x") == ({"a": 1}, False)
    assert add_usage({"a": 1}, {"a": True}) == ({"a": 1}, False)
    assert add_usage({"a": 1}, {"a": "x"}) == ({"a": 1}, False)
    assert add_usage({"a": 1}, {"a": float("inf")}) == ({"a": 1}, False)
    assert add_usage({"a": 1}, {"a": 2}) == ({"a": 3}, True)


def test_projection_maps_committed_events_to_wire_events() -> None:
    lines: list[str] = []
    projection = HeadlessJsonProjection(lines.append, "s1", "/work")
    assert json.loads(lines[0]) == {
        "type": "session",
        "sessionId": "s1",
        "cwd": "/work",
    }

    projection.on_event("turn/start", {"turn": 1})
    projection.on_event("step/start", {"turn": 1, "step": 2})
    projection.on_event("assistant/attempt", {"stream": {"tokens": 1}})
    projection.on_event(
        "assistant/message", {"usage": {"tokens": 2}, "message": {"content": []}}
    )
    projection.on_event(
        "assistant/message",
        {
            "message": {
                "content": [
                    {"type": "reasoning", "text": "why"},
                    {"type": "text", "text": "answer"},
                    {"type": "image"},
                    "bad",
                ]
            }
        },
    )
    projection.on_event("step/end", {"turn": 1, "step": 2})
    projection.on_event("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    projection.on_event(
        "tool/call", {"callId": "c1", "name": "bash", "arguments": "{}"}
    )
    projection.on_event(
        "tool/result",
        {
            "message": {
                "toolCallId": "c1",
                "isError": True,
                "content": [{"type": "text", "text": "boom"}],
            },
            "error": {"code": "E"},
        },
        "append",
    )
    projection.on_event(
        "tool/result",
        {"message": {"toolCallId": "c1", "content": []}},
        "replace",
    )
    projection.on_event("tool/result", {"message": "not a dict"}, "append")
    projection.on_event("unknown/event", {})
    projection.finish("answer", {"kind": "completed"})

    events = [json.loads(line) for line in lines]
    kinds = [event["type"] for event in events]
    assert kinds == [
        "session",
        "status",
        "status",
        "thinking",
        "text",
        "status",
        "status",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert events[5]["usage"] == {"tokens": 3}
    assert events[8]["status"] == "error"
    assert events[-1] == {
        "type": "final",
        "sessionId": "s1",
        "reason": {"kind": "completed"},
        "text": "answer",
    }


def test_projection_handles_missing_message_fields() -> None:
    lines: list[str] = []
    projection = HeadlessJsonProjection(lines.append, "s2", "/work")
    projection.on_event("assistant/message", {"message": "bad"})
    projection.on_event("assistant/message", {"message": {"content": "not a list"}})
    projection.on_event("tool/result", {"message": {"toolCallId": "c"}}, "append")
    events = [json.loads(line) for line in lines]
    assert [event["type"] for event in events] == ["session", "tool_result"]
