"""Unit tests for session query filters."""

from __future__ import annotations

from magitech_harness.session_query.errors import SessionQueryError
from magitech_harness.session_query.filters import (
    compile_event_filter,
    compile_session_filter,
)


def test_compile_session_filter_id() -> None:
    pred = compile_session_filter([{"kind": "id", "values": ["s1", "s2"]}])
    assert pred({"header": {"id": "s1"}, "live": True, "persisted": False}) is True
    assert pred({"header": {"id": "s3"}, "live": True, "persisted": False}) is False


def test_compile_session_filter_id_empty_values() -> None:
    try:
        compile_session_filter([{"kind": "id", "values": []}])
        assert False, "Should have raised invalid_filter"
    except SessionQueryError:
        pass


def test_compile_session_filter_cwd() -> None:
    pred = compile_session_filter([{"kind": "cwd", "values": ["/tmp", None]}])
    assert pred({"header": {"cwd": "/tmp"}, "live": True, "persisted": False}) is True
    assert pred({"header": {"cwd": None}, "live": True, "persisted": False}) is True
    assert (
        pred({"header": {"cwd": "/other"}, "live": True, "persisted": False}) is False
    )


def test_compile_session_filter_created_at() -> None:
    pred = compile_session_filter([{"kind": "created-at", "gte": 1000, "lte": 2000}])
    assert (
        pred({"header": {"createdAt": 1500}, "live": True, "persisted": False}) is True
    )
    assert (
        pred({"header": {"createdAt": 500}, "live": True, "persisted": False}) is False
    )
    assert (
        pred({"header": {"createdAt": 2500}, "live": True, "persisted": False}) is False
    )


def test_compile_session_filter_created_at_inverted() -> None:
    try:
        compile_session_filter([{"kind": "created-at", "gte": 2000, "lte": 1000}])
        assert False, "Should have raised invalid_filter"
    except SessionQueryError:
        pass


def test_compile_session_filter_parent() -> None:
    pred = compile_session_filter([{"kind": "parent", "values": ["parent-1"]}])
    assert (
        pred(
            {"header": {"parentSession": "parent-1"}, "live": True, "persisted": False}
        )
        is True
    )
    assert (
        pred(
            {"header": {"parentSession": "parent-2"}, "live": True, "persisted": False}
        )
        is False
    )


def test_compile_session_filter_availability() -> None:
    pred = compile_session_filter([{"kind": "availability", "values": ["live"]}])
    assert pred({"header": {}, "live": True, "persisted": False}) is True
    assert pred({"header": {}, "live": False, "persisted": True}) is False
    assert pred({"header": {}, "live": True, "persisted": True}) is True

    pred_both = compile_session_filter([{"kind": "availability", "values": ["both"]}])
    assert pred_both({"header": {}, "live": True, "persisted": True}) is True
    assert pred_both({"header": {}, "live": True, "persisted": False}) is False


def test_compile_session_filter_anded() -> None:
    pred = compile_session_filter(
        [
            {"kind": "id", "values": ["s1"]},
            {"kind": "availability", "values": ["live"]},
        ]
    )
    assert pred({"header": {"id": "s1"}, "live": True, "persisted": False}) is True
    assert pred({"header": {"id": "s1"}, "live": False, "persisted": True}) is False
    assert pred({"header": {"id": "s2"}, "live": True, "persisted": False}) is False


def test_compile_event_filter_seq() -> None:
    pred = compile_event_filter([{"kind": "seq", "gte": 5, "lte": 10}])
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 7,
                "type": "t",
                "time": 1000,
                "surface": "current",
                "text": "",
            }
        )
        is True
    )
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 3,
                "type": "t",
                "time": 1000,
                "surface": "current",
                "text": "",
            }
        )
        is False
    )
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 15,
                "type": "t",
                "time": 1000,
                "surface": "current",
                "text": "",
            }
        )
        is False
    )


def test_compile_event_filter_type() -> None:
    pred = compile_event_filter(
        [{"kind": "type", "values": ["user/message", "system/message"]}]
    )
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 1,
                "type": "user/message",
                "time": 1000,
                "surface": "current",
                "text": "",
            }
        )
        is True
    )
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 2,
                "type": "assistant/message",
                "time": 1000,
                "surface": "current",
                "text": "",
            }
        )
        is False
    )


def test_compile_event_filter_surface() -> None:
    pred = compile_event_filter([{"kind": "surface", "values": ["shadowed"]}])
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 1,
                "type": "t",
                "time": 1000,
                "surface": "shadowed",
                "text": "",
            }
        )
        is True
    )
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 2,
                "type": "t",
                "time": 1000,
                "surface": "current",
                "text": "",
            }
        )
        is False
    )


def test_compile_event_filter_text() -> None:
    pred = compile_event_filter([{"kind": "text", "text": "hello"}])
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 1,
                "type": "t",
                "time": 1000,
                "surface": "current",
                "text": "hello world",
            }
        )
        is True
    )
    assert (
        pred(
            {
                "sessionId": "s1",
                "seq": 2,
                "type": "t",
                "time": 1000,
                "surface": "current",
                "text": "goodbye",
            }
        )
        is False
    )


def test_compile_event_filter_text_empty() -> None:
    try:
        compile_event_filter([{"kind": "text", "text": ""}])
        assert False, "Should have raised invalid_filter"
    except SessionQueryError:
        pass


def test_compile_event_filter_type_empty_values() -> None:
    try:
        compile_event_filter([{"kind": "type", "values": []}])
        assert False, "Should have raised invalid_filter"
    except SessionQueryError:
        pass


def test_event_matches_filter_case_insensitive() -> None:
    from magitech_harness.session.events import SessionEvent
    from magitech_harness.session_query.filters import event_matches_filter

    event = SessionEvent(
        id="s1:0",
        session_id="s1",
        type="user/message",
        timestamp=1000.0,
        payload={"message": {"content": [{"type": "text", "text": "Hello World"}]}},
        seq=0,
    )
    assert event_matches_filter(event, [{"kind": "text", "text": "hello"}]) is True
    assert event_matches_filter(event, [{"kind": "text", "text": "WORLD"}]) is True
    assert event_matches_filter(event, [{"kind": "text", "text": "foo"}]) is False
