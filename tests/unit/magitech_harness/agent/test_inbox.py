"""Unit tests for durable inbox projection and AgentInbox command facade."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest
from animas import Context

from magitech_harness.agent.dispatch import agent_events
from magitech_harness.agent.inbox import (
    AgentInbox,
    ReactLoopInbox,
    inbox_projection_definition,
)
from magitech_harness.llm.message import create_user_message
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session


def _user_msg(text: str) -> dict[str, Any]:
    return create_user_message(
        {"content": [{"type": "text", "text": text}], "source": {"kind": "user"}}
    )


class _StubAgent:
    def __init__(self, ctx: Context, session: Session) -> None:
        self.id = session.id
        self.session = session
        self.ctx = ctx
        self.inbox: AgentInbox | None = None


def _inbox_agent(raw_id: str) -> tuple[Context, Session, _StubAgent, AgentInbox]:
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    projection = ctx.sessionProjections.register(inbox_projection_definition)
    session = ctx.sessions.create(raw_id)
    agent = _StubAgent(ctx, session)
    inbox = AgentInbox(
        lambda: projection.state_of(session),
        session,
        agent_events(ctx, agent),  # type: ignore[arg-type]
    )
    agent.inbox = inbox
    assert inbox is not None
    assert ReactLoopInbox is AgentInbox
    return ctx, session, agent, inbox


def _reconstruct_persisted_error(raw_id: str, populate: Any) -> Exception:
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    session = ctx.sessions.create(raw_id)
    populate(session)
    projection = ctx.sessionProjections.register(inbox_projection_definition)
    agent = _StubAgent(ctx, session)
    inbox = AgentInbox(
        lambda: projection.state_of(session),
        session,
        agent_events(ctx, agent),  # type: ignore[arg-type]
    )
    with pytest.raises(Exception) as exc_info:
        _ = inbox.next_turn
    return exc_info.value


def test_retains_inbox_reads_after_registration_released() -> None:
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    projection = ctx.sessionProjections.register(inbox_projection_definition)
    session = ctx.sessions.create("inbox-projection")
    pending = _user_msg("pending")
    session.append(
        "agent/inbox/spliced",
        {"target": "next-turn", "start": 0, "inserted": [pending]},
    )
    agent = _StubAgent(ctx, session)
    dispatch = agent_events(ctx, agent)  # type: ignore[arg-type]
    first = AgentInbox(lambda: projection.state_of(session), session, dispatch)
    second = AgentInbox(lambda: projection.state_of(session), session, dispatch)

    assert first.next_turn == [pending]
    assert second.next_turn == [pending]
    assert first.has_pending is True
    assert ctx.sessionProjections.snapshot(session)["values"]["inbox"] == {
        "next-turn": [pending],
        "next-step": [],
    }
    row = ctx.sessionProjections.checkpoint(session)["inbox"]

    projection.dispose()
    assert ctx.sessionProjections.checkpoint(session)["inbox"] == row
    assert "inbox" not in ctx.sessionProjections.checkpoint(ctx.sessions.create())
    assert first.next_turn == [pending]
    first.clear()
    assert second.next_turn == []
    assert second.has_pending is False
    assert "inbox" not in ctx.sessionProjections.snapshot(session)["values"]


def test_rejects_invalid_durable_coordinates_and_duplicate_ids_on_reconstruction() -> (
    None
):
    out_of_range = _reconstruct_persisted_error(
        "invalid-inbox-range",
        lambda session: session.append(
            "agent/inbox/spliced",
            {"target": "next-turn", "start": 0, "removedCount": 1, "inserted": []},
        ),
    )
    assert str(out_of_range) == "invalid persisted inbox splice at session seq 0"
    assert isinstance(out_of_range.__cause__, ValueError)
    assert str(out_of_range.__cause__) == "invalid inbox splice"

    pending = _user_msg("duplicate")
    duplicate = _reconstruct_persisted_error(
        "invalid-inbox-duplicate",
        lambda session: (
            session.append(
                "agent/inbox/spliced",
                {"target": "next-turn", "start": 0, "inserted": [pending]},
            ),
            session.append(
                "agent/inbox/spliced",
                {"target": "next-step", "start": 0, "inserted": [pending]},
            ),
        ),
    )
    assert str(duplicate) == "invalid persisted inbox splice at session seq 1"
    assert isinstance(duplicate.__cause__, ValueError)
    assert str(duplicate.__cause__) == f'message "{pending["id"]}" is already pending'


def test_projects_inherited_inbox_events_in_forked_session() -> None:
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    projection = ctx.sessionProjections.register(inbox_projection_definition)
    parent = ctx.sessions.create("inbox-fork-parent")
    parent_agent = _StubAgent(ctx, parent)
    parent_inbox = AgentInbox(
        lambda: projection.state_of(parent),
        parent,
        agent_events(ctx, parent_agent),  # type: ignore[arg-type]
    )
    inherited = _user_msg("parent pending")
    parent_inbox.append("next-turn", inherited)
    child = ctx.sessions.fork(parent, child_session_id="inbox-fork-child")
    child_agent = _StubAgent(ctx, child)
    child_inbox = AgentInbox(
        lambda: projection.state_of(child),
        child,
        agent_events(ctx, child_agent),  # type: ignore[arg-type]
    )

    assert child.inherited_event_count == len(parent.events)
    assert child_inbox.next_turn == [inherited]

    own = _user_msg("child pending")
    child_inbox.append("next-turn", own)
    assert child_inbox.next_turn == [inherited, own]


def test_updates_projection_cell_before_session_observers_run() -> None:
    ctx, session, _agent, inbox = _inbox_agent("inbox-live-projection")
    pending = _user_msg("direct")
    observed: list[Any] = []

    def _on_event(subject: Session, event: Any) -> None:
        if subject is session and event.type == "agent/inbox/spliced":
            observed.extend(inbox.next_turn)

    ctx.on("session/event", _on_event)
    inbox.append("next-turn", pending)

    assert observed == [pending]
    assert ctx.sessionProjections.snapshot(session)["values"]["inbox"] == {
        "next-turn": [pending],
        "next-step": [],
    }


def test_replaces_pending_message_by_identity_across_both_lists() -> None:
    ctx, _session, agent, inbox = _inbox_agent("replace-inbox")
    inserted: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    ctx.on("agent/inbox/inserted", lambda p: inserted.append(p["message"]))
    ctx.on("agent/inbox/discarded", lambda p: discarded.append(p["message"]))

    original = _user_msg("original")
    next_step = _user_msg("step")
    replacement = _user_msg("replacement")
    edited_step = copy.deepcopy(dict(next_step))
    edited_step["content"] = [{"type": "text", "text": "edited step"}]

    inbox.append("next-turn", original)
    inbox.append("next-step", next_step)

    missing = _user_msg("missing")
    assert inbox.replace(missing["id"], replacement) is False
    assert inbox.replace(original["id"], replacement) is True
    assert inbox.replace(next_step["id"], edited_step) is True
    assert inbox.next_turn == [replacement]
    assert inbox.next_step == [edited_step]
    assert discarded == [original, next_step]
    assert inserted == [original, next_step, replacement, edited_step]
    assert agent.inbox is inbox

    with pytest.raises(
        ValueError, match=f'message "{replacement["id"]}" is already pending'
    ):
        inbox.replace(edited_step["id"], replacement)


def test_normalizes_splice_coordinates_rejects_duplicates_and_reports_missing_removals() -> (
    None
):
    _ctx, _session, _agent, inbox = _inbox_agent("splice-inbox")
    first = _user_msg("first")
    second = _user_msg("second")
    prefixed = _user_msg("prefixed")

    inbox.splice("next-turn", math.nan, math.nan, [first, second])
    assert inbox.next_turn == [first, second]
    assert inbox.splice("next-turn", -1, 1, []) == [second]
    inbox.prepend("next-turn", prefixed)
    assert inbox.next_turn == [prefixed, first]
    assert inbox.remove(second["id"]) is False
    assert inbox.remove(first["id"]) is True
    assert inbox.next_turn == [prefixed]
    assert inbox.splice("next-turn", 0, 0, []) == []

    with pytest.raises(
        ValueError, match=f'message "{prefixed["id"]}" is already pending'
    ):
        inbox.append("next-step", prefixed)


def test_clears_both_lists_as_durable_cancellations_and_claims_step_then_turn() -> None:
    ctx, session, _agent, inbox = _inbox_agent("clear-and-claim")
    discarded: list[dict[str, Any]] = []
    claimed_events: list[tuple[dict[str, Any], int]] = []
    ctx.on("agent/inbox/discarded", lambda p: discarded.append(p["message"]))
    ctx.on(
        "agent/inbox/claimed",
        lambda p: claimed_events.append((p["message"], p["turn"])),
    )

    turn_msg = _user_msg("turn")
    step_msg = _user_msg("step")
    inbox.append("next-turn", turn_msg)
    inbox.append("next-step", step_msg)
    before_clear = len(session.events)

    inbox.clear()
    assert inbox.next_turn == []
    assert inbox.next_step == []
    assert discarded == [step_msg, turn_msg]
    assert [e.payload for e in session.events[before_clear:]] == [
        {
            "target": "next-step",
            "start": 0,
            "removedCount": 1,
            "inserted": [],
            "outcome": "canceled",
        },
        {
            "target": "next-turn",
            "start": 0,
            "removedCount": 1,
            "inserted": [],
            "outcome": "canceled",
        },
    ]

    inbox.clear()
    assert len(session.events) == before_clear + 2

    # Test claim ordering: next-step then one item from next-turn
    step_a = _user_msg("step-a")
    turn_a = _user_msg("turn-a")
    turn_b = _user_msg("turn-b")
    inbox.append("next-step", step_a)
    inbox.append("next-turn", turn_a)
    inbox.append("next-turn", turn_b)

    batch = inbox.claim("next-turn", 4)
    assert batch == [step_a, turn_a]
    assert claimed_events == [(step_a, 4), (turn_a, 4)]
    assert inbox.next_step == []
    assert inbox.next_turn == [turn_b]

    # claim next-step only does not consume next-turn
    step_b = _user_msg("step-b")
    inbox.append("next-step", step_b)
    step_batch = inbox.claim("next-step", 4)
    assert step_batch == [step_b]
    assert inbox.next_turn == [turn_b]


def test_inbox_projection_ignores_unrelated_events_and_validates_wire_shape() -> None:
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    handle = ctx.sessionProjections.register(inbox_projection_definition)
    session = ctx.sessions.create("unrelated")
    initial = handle.state_of(session)
    session.append("turn/start", {"turn": 1})
    assert handle.state_of(session) is initial

    with pytest.raises(TypeError):
        inbox_projection_definition.state_schema("bad")
    with pytest.raises(TypeError):
        inbox_projection_definition.state_schema({"next-turn": "bad", "next-step": []})
    with pytest.raises(ValueError):
        inbox_projection_definition.state_schema(
            {"next-turn": [{"id": ""}], "next-step": []}
        )
    with pytest.raises(TypeError):
        inbox_projection_definition.state_schema({"next-turn": [123], "next-step": []})


def test_inbox_splice_clamps_infinity_and_negative_infinity() -> None:
    _ctx, _session, _agent, inbox = _inbox_agent("inbox-inf")
    first = _user_msg("first")
    second = _user_msg("second")
    tail = _user_msg("tail")
    head = _user_msg("head")
    inbox.splice("next-turn", float("inf"), 0, [first, second])
    inbox.splice("next-turn", float("inf"), 0, [tail])
    inbox.splice("next-turn", float("-inf"), float("-inf"), [head])
    assert inbox.next_turn == [head, first, second, tail]

    removed = inbox.splice("next-turn", 1, float("inf"), [])
    assert removed == [first, second, tail]
    assert inbox.next_turn == [head]

    inbox.append("next-turn", first)
    assert inbox.splice("next-turn", 1.7, 1.9, []) == [first]
    assert inbox.next_turn == [head]
