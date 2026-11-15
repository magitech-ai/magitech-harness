"""Unit tests for consumed-work fold semantics (1:1 parity)."""

from __future__ import annotations

from typing import Any

from magitech_harness.agent.consumed_work import ConsumedWork, fold_consumed_work
from magitech_harness.llm.message import create_user_message
from magitech_harness.session.store import Session


def _msg(text: str) -> dict[str, Any]:
    return create_user_message(
        {"content": [{"type": "text", "text": text}], "source": {"kind": "user"}}
    )


def _accept(session: Session, text: str) -> None:
    session.append(
        "agent/inbox/spliced",
        {"target": "next-turn", "start": 0, "inserted": [_msg(text)]},
    )


def _claim(session: Session) -> None:
    session.append(
        "agent/inbox/spliced",
        {"target": "next-turn", "start": 0, "removedCount": 1, "inserted": []},
    )


def _cancel_pending(session: Session) -> None:
    session.append(
        "agent/inbox/spliced",
        {
            "target": "next-turn",
            "start": 0,
            "removedCount": 1,
            "inserted": [],
            "outcome": "canceled",
        },
    )


def _stepped_turn(session: Session, turn: int, reason: dict[str, Any]) -> None:
    session.append("turn/start", {"turn": turn})
    _claim(session)
    session.append("step/start", {"turn": turn, "step": 1})
    session.append("step/end", {"turn": turn, "step": 1})
    session.append("turn/end", {"turn": turn, "reason": reason})


def test_empty_log_with_queued_work_reports_nothing() -> None:
    session = Session.create("empty")
    _accept(session, "queued")

    assert fold_consumed_work(session.events) == ConsumedWork(
        end=None, dropped_unrun=False
    )


def test_reports_latest_turn_that_entered_a_model_step() -> None:
    session = Session.create("stepped")
    _stepped_turn(session, 1, {"kind": "completed"})
    _stepped_turn(session, 2, {"kind": "max-tokens"})

    result = fold_consumed_work(session.events)
    assert result.end is not None
    assert result.end.payload == {"turn": 2, "reason": {"kind": "max-tokens"}}
    assert result.dropped_unrun is False


def test_reports_turn_that_claimed_input_then_failed_before_step() -> None:
    session = Session.create("failed-claim")
    _stepped_turn(session, 1, {"kind": "completed"})
    session.append("turn/start", {"turn": 2})
    _claim(session)
    session.append(
        "turn/end",
        {
            "turn": 2,
            "reason": {
                "kind": "error",
                "error": {"message": "ENOSPC", "code": "UNKNOWN"},
            },
        },
    )

    result = fold_consumed_work(session.events)
    assert result.end is not None
    assert result.end.payload["turn"] == 2


def test_reports_turn_that_claimed_input_then_aborted_before_step() -> None:
    session = Session.create("stopped-claim")
    _stepped_turn(session, 1, {"kind": "completed"})
    session.append("turn/start", {"turn": 2})
    _claim(session)
    session.append(
        "turn/end",
        {"turn": 2, "reason": {"kind": "aborted", "reason": {"kind": "user"}}},
    )

    result = fold_consumed_work(session.events)
    assert result.end is not None
    assert result.end.payload["turn"] == 2


def test_ignores_turns_without_claims() -> None:
    session = Session.create("no-claim")
    _stepped_turn(session, 1, {"kind": "completed"})
    session.append("turn/start", {"turn": 2})
    session.append(
        "turn/end",
        {"turn": 2, "reason": {"kind": "aborted", "reason": {"kind": "parent"}}},
    )
    session.append("turn/start", {"turn": 3})
    session.append(
        "turn/end",
        {
            "turn": 3,
            "reason": {"kind": "error", "error": {"message": "x", "code": "UNKNOWN"}},
        },
    )
    session.append("turn/start", {"turn": 4})
    session.append("turn/end", {"turn": 4, "reason": {"kind": "blocked"}})

    result = fold_consumed_work(session.events)
    assert result.end is not None
    assert result.end.payload["turn"] == 1


def test_reports_turn_whose_claimed_input_was_rejected_pre_step() -> None:
    session = Session.create("rejected-claim")
    _stepped_turn(session, 1, {"kind": "completed"})
    session.append("turn/start", {"turn": 2})
    _claim(session)
    session.append("turn/end", {"turn": 2, "reason": {"kind": "blocked"}})

    result = fold_consumed_work(session.events)
    assert result.end is not None
    assert result.end.payload["turn"] == 2


def test_ignores_claim_emptied_by_its_own_turn() -> None:
    session = Session.create("emptied-claim")
    _stepped_turn(session, 1, {"kind": "completed"})
    session.append("turn/start", {"turn": 2})
    _claim(session)
    session.append("turn/end", {"turn": 2, "reason": {"kind": "completed"}})

    result = fold_consumed_work(session.events)
    assert result.end is not None
    assert result.end.payload["turn"] == 1


def test_credits_orphan_claim_to_no_turn() -> None:
    session = Session.create("mid-turn-suffix")
    _stepped_turn(session, 1, {"kind": "completed"})
    _claim(session)
    session.append(
        "turn/end",
        {"turn": 2, "reason": {"kind": "aborted", "reason": {"kind": "user"}}},
    )

    result = fold_consumed_work(session.events)
    assert result.end is not None
    assert result.end.payload["turn"] == 1


def test_reports_dropped_unrun_work_after_last_accounting_turn() -> None:
    session = Session.create("dropped")
    _stepped_turn(session, 1, {"kind": "completed"})
    _accept(session, "never runs")
    _cancel_pending(session)

    expected_end = next(e for e in session.events if e.type == "turn/end")
    assert fold_consumed_work(session.events) == ConsumedWork(
        end=expected_end, dropped_unrun=True
    )


def test_keeps_replacement_pending_not_dropped() -> None:
    session = Session.create("replaced")
    _stepped_turn(session, 1, {"kind": "completed"})
    session.append(
        "agent/inbox/spliced",
        {
            "target": "next-turn",
            "start": 0,
            "removedCount": 1,
            "inserted": [_msg("rewritten")],
            "outcome": "canceled",
        },
    )

    assert fold_consumed_work(session.events).dropped_unrun is False


def test_later_accounting_turn_absorbs_earlier_drop() -> None:
    session = Session.create("absorbed")
    _stepped_turn(session, 1, {"kind": "completed"})
    _cancel_pending(session)
    _stepped_turn(session, 2, {"kind": "completed"})

    expected_end = [e for e in session.events if e.type == "turn/end"][-1]
    assert fold_consumed_work(session.events) == ConsumedWork(
        end=expected_end, dropped_unrun=False
    )


def test_unknown_claim_end_reason_still_accounts_for_work() -> None:
    session = Session.create("custom-end")
    session.append("turn/start", {"turn": 1})
    _claim(session)
    session.append("turn/end", {"turn": 1, "reason": {"kind": "interrupted"}})

    result = fold_consumed_work(session.events)
    assert result.end is not None
    assert result.end.payload["turn"] == 1
