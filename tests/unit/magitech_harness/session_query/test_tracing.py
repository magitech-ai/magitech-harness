"""Unit tests for session query tracing."""

from __future__ import annotations

from magitech_harness.session.events import SessionEvent
from magitech_harness.session_query.errors import SessionQueryError
from magitech_harness.session_query.tracing import trace_event, trace_session


def _make_session_record(session_id: str, parent: str | None = None) -> dict:
    return {
        "header": {"id": session_id, "parentSession": parent},
        "live": True,
        "persisted": False,
        "formatStatus": None,
    }


def test_trace_session_no_parent() -> None:
    """A session with no parent has empty ancestors and is its own root."""
    records = {"s1": _make_session_record("s1")}

    def get_record(sid: str):
        return records.get(sid)

    def get_parent_id(sid: str):
        rec = records.get(sid)
        return rec["header"]["parentSession"] if rec else None

    def get_children(sid: str):
        return []

    trace = trace_session("s1", get_record, get_parent_id, get_children)
    assert trace["target"]["header"]["id"] == "s1"
    assert trace["ancestors"] == []
    assert trace["descendants"] == []
    assert trace["complete"] is True
    assert trace["root"]["header"]["id"] == "s1"


def test_trace_session_with_parent_chain() -> None:
    """A session chain s3 -> s2 -> s1 traces all ancestors."""
    records = {
        "s1": _make_session_record("s1"),
        "s2": _make_session_record("s2", "s1"),
        "s3": _make_session_record("s3", "s2"),
    }

    def get_record(sid: str):
        return records.get(sid)

    def get_parent_id(sid: str):
        rec = records.get(sid)
        return rec["header"]["parentSession"] if rec else None

    def get_children(sid: str):
        return [r for r in records.values() if r["header"]["parentSession"] == sid]

    trace = trace_session("s3", get_record, get_parent_id, get_children)
    assert trace["target"]["header"]["id"] == "s3"
    assert len(trace["ancestors"]) == 2
    assert trace["ancestors"][0]["header"]["id"] == "s2"
    assert trace["ancestors"][1]["header"]["id"] == "s1"
    assert trace["complete"] is True
    assert trace["root"]["header"]["id"] == "s1"


def test_trace_session_incomplete_lineage() -> None:
    """When a parent is missing from the corpus, lineage is incomplete."""
    records = {
        "s1": _make_session_record("s1"),
        "s3": _make_session_record("s3", "s2"),  # s2 is missing
    }

    def get_record(sid: str):
        return records.get(sid)

    def get_parent_id(sid: str):
        rec = records.get(sid)
        return rec["header"]["parentSession"] if rec else None

    def get_children(sid: str):
        return []

    trace = trace_session("s3", get_record, get_parent_id, get_children)
    assert trace["complete"] is False
    assert trace["unresolvedParentId"] == "s2"


def test_trace_session_not_found() -> None:
    def get_record(sid: str):
        return None

    def get_parent_id(sid: str):
        return None

    def get_children(sid: str):
        return []

    try:
        trace_session("missing", get_record, get_parent_id, get_children)
        assert False, "Should have raised session_not_found"
    except SessionQueryError:
        pass


def test_trace_session_descendants() -> None:
    """Descendant trees are built recursively."""
    records = {
        "s1": _make_session_record("s1"),
        "s2": _make_session_record("s2", "s1"),
        "s3": _make_session_record("s3", "s1"),
        "s4": _make_session_record("s4", "s2"),
    }

    def get_record(sid: str):
        return records.get(sid)

    def get_parent_id(sid: str):
        rec = records.get(sid)
        return rec["header"]["parentSession"] if rec else None

    def get_children(sid: str):
        return [r for r in records.values() if r["header"]["parentSession"] == sid]

    trace = trace_session("s1", get_record, get_parent_id, get_children)
    assert len(trace["descendants"]) == 2
    child_ids = {d["session"]["header"]["id"] for d in trace["descendants"]}
    assert child_ids == {"s2", "s3"}
    # s2 has a child s4
    s2_desc = next(
        d for d in trace["descendants"] if d["session"]["header"]["id"] == "s2"
    )
    assert len(s2_desc["descendants"]) == 1
    assert s2_desc["descendants"][0]["session"]["header"]["id"] == "s4"


def test_trace_event_target_not_found() -> None:
    events = [
        SessionEvent(
            id="s1:0",
            session_id="s1",
            type="user/message",
            timestamp=1000.0,
            payload={"message": {"content": [{"type": "text", "text": "hi"}]}},
            seq=0,
        )
    ]

    def get_record(sid: str):
        return {"header": {"id": "s1"}, "live": True, "persisted": False}

    try:
        trace_event("s1", 5, events, get_record)
        assert False, "Should have raised invalid_lineage"
    except SessionQueryError:
        pass


def test_trace_event_no_replacements() -> None:
    """An event with no replacements traces cleanly."""
    events = [
        SessionEvent(
            id="s1:0",
            session_id="s1",
            type="user/message",
            timestamp=1000.0,
            payload={"message": {"content": [{"type": "text", "text": "hi"}]}},
            seq=0,
        )
    ]

    def get_record(sid: str):
        return {"header": {"id": "s1"}, "live": True, "persisted": False}

    trace = trace_event("s1", 0, events, get_record)
    assert trace["target"]["seq"] == 0
    assert trace["replacedBy"] is None
    assert trace["replacementChain"] == []
    assert trace["replacedEventSeqs"] == []
    assert trace["sourceEventSeqs"] == []
    assert trace["derivedEventSeqs"] == []
