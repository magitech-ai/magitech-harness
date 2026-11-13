"""Unit tests for crash-repair closers (openTurnClosers / interruptedTurnClosers)."""

import pytest

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.repair import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    ToolCallRecovery,
    interrupted_turn_closers,
    open_turn_closers,
)


def _event(
    seq: int,
    type: str,
    payload: dict | None = None,
    **kwargs: object,
) -> SessionEvent:
    return SessionEvent(
        id=f"s:{seq}",
        session_id="s",
        type=type,
        timestamp=1000.0 + seq,
        payload=payload or {},
        seq=seq,
        **kwargs,  # type: ignore[arg-type]
    )


def _assistant_with_call(
    seq: int, call_id: str = "c1", turn: int = 1, step: int = 1
) -> SessionEvent:
    return _event(
        seq,
        "assistant/message",
        {
            "turn": turn,
            "step": step,
            "message": {
                "id": f"a{seq}",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool-call",
                        "id": call_id,
                        "name": "search",
                        "arguments": "{}",
                    }
                ],
            },
            "stream": [],
        },
        surface_op="append",
    )


def _balanced_log() -> list[SessionEvent]:
    return [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2),
        _event(3, "tool/call", {"turn": 1, "step": 1, "callId": "c1"}),
        _event(
            4,
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "t4",
                    "role": "tool",
                    "toolCallId": "c1",
                    "isError": False,
                    "source": {"kind": "tool", "callId": "c1"},
                    "content": [{"type": "text", "text": "3"}],
                },
            },
            surface_op="append",
        ),
        _event(5, "step/end", {"turn": 1, "step": 1}),
        _event(6, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    ]


def test_closers_balanced_and_empty_logs() -> None:
    """Verifies balanced and empty logs need no synthetic closers."""
    assert open_turn_closers(_balanced_log(), "forked") == []
    assert open_turn_closers([], "forked") == []
    assert interrupted_turn_closers(_balanced_log()) == []
    assert interrupted_turn_closers([]) == []


def test_closers_open_turn_without_step() -> None:
    """Verifies an open turn closes with a forked turn/end only."""
    events = [_event(0, "turn/start", {"turn": 1})]
    closers = open_turn_closers(events, "forked")
    assert len(closers) == 1
    assert closers[0].type == "turn/end"
    assert closers[0].payload["reason"] == {"kind": "forked"}
    assert closers[0].seq == 1
    assert closers[0].timestamp == 1000.0


def test_closers_open_step_closes_step_then_turn() -> None:
    """Verifies an open step closes before its turn."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
    ]
    closers = open_turn_closers(events, "forked")
    assert [c.type for c in closers] == ["step/end", "turn/end"]
    assert closers[0].payload == {"turn": 1, "step": 1}
    assert closers[1].payload["reason"] == {"kind": "forked"}
    assert [c.seq for c in closers] == [2, 3]


def test_closers_unmatched_started_call_gets_unknown_outcome() -> None:
    """Verifies a started but unanswered call gets an unknown-outcome result."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2),
        _event(3, "tool/call", {"turn": 1, "step": 1, "callId": "c1"}),
    ]
    closers = open_turn_closers(events, "forked")
    assert [c.type for c in closers] == ["tool/result", "step/end", "turn/end"]
    result = closers[0]
    assert result.seq == 4
    assert result.surface_op == "append"
    assert result.source_event_seqs == [3]
    assert result.payload["message"]["id"] == "forked-tool-result-c1-4"
    message = result.payload["message"]
    assert message["isError"] is True
    assert result.payload["error"]["code"] == TOOL_OUTCOME_UNKNOWN
    assert "records this tool call starting" in message["content"][0]["text"]


def test_closers_unmatched_unstarted_call_gets_not_started() -> None:
    """Verifies a never-started call gets a not-started error result."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2),
    ]
    closers = open_turn_closers(events, "forked")
    result = closers[0]
    assert result.type == "tool/result"
    assert result.source_event_seqs is None
    assert result.payload["message"]["id"] == "forked-tool-result-c1-3"
    message = result.payload["message"]
    assert result.payload["error"]["code"] == TOOL_NOT_STARTED
    assert "has no record of this tool call starting" in message["content"][0]["text"]


def test_closers_matched_result_resolves_pending_call() -> None:
    """Verifies an append tool/result clears its pending call."""
    events = _balanced_log()[:-2]  # Drop step/end and turn/end.
    closers = open_turn_closers(events, "forked")
    assert [c.type for c in closers] == ["step/end", "turn/end"]


def test_closers_replace_result_does_not_resolve() -> None:
    """Verifies only an append tool/result resolves a pending call."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2),
        _event(3, "tool/call", {"turn": 1, "step": 1, "callId": "c1"}),
        _event(
            4,
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "t4",
                    "role": "tool",
                    "toolCallId": "c1",
                    "isError": False,
                    "source": {"kind": "tool", "callId": "c1"},
                    "content": [{"type": "text", "text": "3"}],
                },
            },
            surface_op={"op": "replace", "startSeq": 4, "endSeq": 4},
            source_event_seqs=[4],
        ),
    ]
    closers = open_turn_closers(events, "forked")
    assert closers[0].type == "tool/result"
    assert closers[0].payload["error"]["code"] == TOOL_OUTCOME_UNKNOWN


def test_closers_turn_end_clears_pending() -> None:
    """Verifies a turn/end clears pending calls from an earlier turn."""
    events = [
        _assistant_with_call(0),
        _event(1, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        _event(2, "turn/start", {"turn": 2}),
        _event(3, "step/start", {"turn": 2, "step": 1}),
    ]
    closers = open_turn_closers(events, "forked")
    assert [c.type for c in closers] == ["step/end", "turn/end"]


def test_closers_closed_steps_and_turns_preserved() -> None:
    """Verifies closed steps/turns — and failed calls already missing
    results — are left unchanged by the closers."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2),
        _event(3, "step/end", {"turn": 1, "step": 1}),
        _event(4, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        _event(5, "turn/start", {"turn": 2}),
        _event(6, "step/start", {"turn": 2, "step": 1}),
    ]
    closers = open_turn_closers(events, "forked")
    assert [c.type for c in closers] == ["step/end", "turn/end"]
    # The failed call from turn 1 (no result) is not resurrected.
    assert all(c.seq > 6 for c in closers)


def test_interrupted_closers_use_interrupted_cause() -> None:
    """Verifies the crash-recovery twin uses interrupted wording and ids."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2),
    ]
    closers = interrupted_turn_closers(events)
    result = closers[0]
    assert result.payload["message"]["id"] == "interrupted-tool-result-c1-3"
    assert result.payload["error"]["code"] == TOOL_NOT_STARTED
    assert (
        "interrupted before the Harness recorded it as started"
        in (result.payload["message"]["content"][0]["text"])
    )
    assert closers[-1].payload["reason"] == {"kind": "interrupted"}


def test_recovery_flat_tool_calls_payload() -> None:
    """Verifies the harness flat tool_calls payload shape is also tracked."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _event(
            2,
            "assistant/message",
            {
                "turn": 1,
                "step": 1,
                "tool_calls": [{"call_id": "flat1", "name": "search"}],
            },
            surface_op="append",
        ),
    ]
    closers = open_turn_closers(events, "forked")
    assert closers[0].payload["message"]["id"] == "forked-tool-result-flat1-3"
    assert closers[0].payload["error"]["code"] == TOOL_NOT_STARTED


def test_recovery_multiple_calls_preserve_assistant_order() -> None:
    """Verifies several unmatched calls close in assistant order."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2, "c1"),
        _assistant_with_call(3, "c2"),
    ]
    closers = open_turn_closers(events, "forked")
    assert [c.payload["message"]["id"] for c in closers[:2]] == [
        "forked-tool-result-c1-4",
        "forked-tool-result-c2-5",
    ]
    assert [c.seq for c in closers] == [4, 5, 6, 7]
    assert closers[2].type == "step/end"
    assert closers[3].type == "turn/end"


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "not-a-dict"},
        {"message": {"content": "not-a-list"}},
        {"message": {"content": [{"type": "text", "text": "no call"}]}},
        {"message": {"content": [{"type": "tool-call"}]}},
        {"tool_calls": [{"call_id": 5}]},
        {"tool_calls": ["not-a-dict"]},
        {"tool_calls": [{"name": "search"}]},
    ],
)
def test_recovery_malformed_assistant_payloads(payload: dict) -> None:
    """Verifies malformed assistant payloads register no pending calls."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _event(2, "assistant/message", payload, surface_op="append"),
    ]
    closers = open_turn_closers(events, "forked")
    assert [c.type for c in closers] == ["step/end", "turn/end"]


def test_recovery_unknown_call_and_result_ids() -> None:
    """Verifies tool/call and tool/result naming unknown calls change
    nothing: the real pending call still closes as not started."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2, "c1"),
        _event(3, "tool/call", {"turn": 1, "step": 1, "callId": "unknown"}),
        _event(
            4,
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "t4",
                    "role": "tool",
                    "toolCallId": "unknown",
                    "isError": False,
                    "source": {"kind": "tool", "callId": "unknown"},
                    "content": [{"type": "text", "text": "x"}],
                },
            },
            surface_op="append",
        ),
    ]
    closers = open_turn_closers(events, "forked")
    assert closers[0].type == "tool/result"
    assert closers[0].payload["error"]["code"] == TOOL_NOT_STARTED
    assert closers[0].payload["message"]["id"] == "forked-tool-result-c1-5"


def test_recovery_tool_call_id_fallback() -> None:
    """Verifies a tool/result naming its call only via message.toolCallId
    (no source block) still resolves the pending call."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2, "c1"),
        _event(
            3,
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "t3",
                    "role": "tool",
                    "toolCallId": "c1",
                    "isError": False,
                    "content": [{"type": "text", "text": "3"}],
                },
            },
            surface_op="append",
        ),
    ]
    closers = open_turn_closers(events, "forked")
    assert [c.type for c in closers] == ["step/end", "turn/end"]


def test_recovery_tool_call_without_any_id() -> None:
    """Verifies a tool/call naming no call id at all changes nothing."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2, "c1"),
        _event(3, "tool/call", {"turn": 1, "step": 1}),
    ]
    closers = open_turn_closers(events, "forked")
    assert closers[0].type == "tool/result"
    assert closers[0].payload["error"]["code"] == TOOL_NOT_STARTED
    assert closers[0].payload["message"]["id"] == "forked-tool-result-c1-4"


def test_recovery_non_string_source_call_id_falls_through() -> None:
    """Verifies a non-string source callId falls through to toolCallId."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2, "c1"),
        _event(
            3,
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "t3",
                    "role": "tool",
                    "toolCallId": "c1",
                    "isError": False,
                    "source": {"kind": "tool", "callId": 5},
                    "content": [{"type": "text", "text": "3"}],
                },
            },
            surface_op="append",
        ),
    ]
    closers = open_turn_closers(events, "forked")
    assert [c.type for c in closers] == ["step/end", "turn/end"]


def test_recovery_non_string_ids_yield_no_call() -> None:
    """Verifies non-string call ids everywhere yield no call identity."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "step/start", {"turn": 1, "step": 1}),
        _assistant_with_call(2, "c1"),
        _event(
            3,
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "t3",
                    "role": "tool",
                    "toolCallId": 7,
                    "isError": False,
                    "source": {"kind": "tool", "callId": 5},
                    "content": [{"type": "text", "text": "3"}],
                },
            },
            surface_op="append",
        ),
    ]
    closers = open_turn_closers(events, "forked")
    assert closers[0].type == "tool/result"
    assert closers[0].payload["message"]["id"] == "forked-tool-result-c1-4"


def test_recovery_results_empty_without_observation() -> None:
    """Verifies a fresh recovery with no observed events yields nothing."""
    assert ToolCallRecovery().results() == []
    assert ToolCallRecovery("forked").results() == []
