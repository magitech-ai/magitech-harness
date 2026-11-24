"""Unit tests for ApprovalService, its audit pair, policy and turn enclosure."""

import asyncio
from types import SimpleNamespace

import pytest
from animas import Context, EventMode

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.approval.service import (
    ALL_OUTCOMES,
    ALL_POLICIES,
    ApprovalService,
    calculate_argument_digest,
)
from magitech_harness.session.events import EventType
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.store import Session


def _agent(session: Session) -> SimpleNamespace:
    return SimpleNamespace(session=session)


def _session_with_turn(name: str) -> Session:
    session = Session.create(name)
    session.append(EventType.TURN_START, {"turn": 1})
    return session


def test_calculate_argument_digest_canonical_json() -> None:
    """Different key orderings produce identical sha256 digests."""
    d1 = {"b": 2, "a": 1, "nested": {"y": "val", "x": [1, 2]}}
    d2 = {"a": 1, "nested": {"x": [1, 2], "y": "val"}, "b": 2}
    assert calculate_argument_digest(d1) == calculate_argument_digest(d2)
    assert calculate_argument_digest(d1).startswith("sha256:")
    assert calculate_argument_digest({"a": 1}) != calculate_argument_digest(d1)


def test_vocabulary_is_closed() -> None:
    assert ALL_OUTCOMES == frozenset(
        {"allowed-once", "rejected", "cancelled", "unavailable"}
    )
    assert ALL_POLICIES == frozenset({"ask", "never"})


@pytest.mark.asyncio
async def test_request_outside_turn_fails_loud() -> None:
    """An approval ask between turns raises before appending any audit event."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = Session.create("no-turn")
    req = {"agent": _agent(session), "toolName": "bash"}

    with pytest.raises(RuntimeError, match="outside an open turn"):
        await service.request(req)
    assert not any(
        e.type in (EventType.APPROVAL_ASKED, EventType.APPROVAL_DECIDED)
        for e in session.events
    )


@pytest.mark.asyncio
async def test_audit_pair_appended_in_order() -> None:
    """Every ask appends approval/asked then approval/decided with the same id."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("audit")
    service.set_handler(lambda _r: asyncio.sleep(0, result="allowed-once"))  # type: ignore[arg-type]

    outcome = await service.request(
        {"agent": _agent(session), "toolName": "bash", "callId": "c1"}
    )

    assert outcome == "allowed-once"
    audit = [
        e
        for e in session.events
        if e.type in (EventType.APPROVAL_ASKED, EventType.APPROVAL_DECIDED)
    ]
    assert [e.type for e in audit] == [
        EventType.APPROVAL_ASKED,
        EventType.APPROVAL_DECIDED,
    ]
    assert audit[0].payload["id"] == audit[1].payload["id"]
    assert audit[0].payload["toolName"] == "bash"
    assert audit[0].payload["callId"] == "c1"
    assert audit[1].payload["outcome"] == "allowed-once"


@pytest.mark.asyncio
async def test_no_answerer_fails_closed() -> None:
    """With no answerer the outcome is 'unavailable' and the audit pair still lands."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("no-answerer")

    outcome = await service.request({"agent": _agent(session), "toolName": "bash"})
    assert outcome == "unavailable"


@pytest.mark.asyncio
async def test_never_policy_auto_rejects_without_asking() -> None:
    """The 'never' policy rejects deterministically, never calling an answerer."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx, {"policy": "never"})
    session = _session_with_turn("never")

    handler_called = []

    async def _handler(_r):
        handler_called.append(True)
        return "allowed-once"

    service.set_handler(_handler)
    outcome = await service.request({"agent": _agent(session), "toolName": "bash"})
    assert outcome == "rejected"
    assert handler_called == []


@pytest.mark.asyncio
async def test_grant_is_one_shot_not_cached() -> None:
    """A grant applies to one call only; the next identical call re-asks."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("one-shot")
    calls = []

    async def _handler(_r):
        calls.append(True)
        return "allowed-once" if len(calls) == 1 else "rejected"

    service.set_handler(_handler)
    req = {"agent": _agent(session), "toolName": "bash", "callId": "c1"}
    assert await service.request(req) == "allowed-once"
    assert await service.request(req) == "rejected"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_waterfall_answerer_and_normalization() -> None:
    """The approval/request waterfall is consulted; rogue returns fail closed."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("waterfall")

    ctx.on(
        "approval/request",
        lambda req, next_fn: "allowed-once",
        mode=EventMode.WATERFALL,
    )
    assert (
        await service.request({"agent": _agent(session), "toolName": "x"})
        == "allowed-once"
    )

    session2 = _session_with_turn("rogue")
    # A separate context for the rogue case to avoid stacking listeners.
    ctx2 = Context()
    SessionProjectionRegistry(ctx2)
    svc2 = ApprovalService(ctx2)
    ctx2.on(
        "approval/request",
        lambda req, next_fn: "not-a-valid-outcome",
        mode=EventMode.WATERFALL,
    )
    assert (
        await svc2.request({"agent": _agent(session2), "toolName": "x"})
        == "unavailable"
    )


@pytest.mark.asyncio
async def test_cancelled_signal_returns_cancelled() -> None:
    """An already-aborted signal yields 'cancelled' without asking."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("cancel")
    signal = AbortSignal()
    signal.abort("user")

    outcome = await service.request(
        {"agent": _agent(session), "toolName": "bash", "signal": signal}
    )
    assert outcome == "cancelled"


@pytest.mark.asyncio
async def test_set_policy_appends_event() -> None:
    """setPolicy appends an approval/policy record to the session."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("policy")
    agent = _agent(session)

    service.set_policy(agent, "never")
    assert any(
        e.type == EventType.APPROVAL_POLICY and e.payload["policy"] == "never"
        for e in session.events
    )
    assert service.effective_policy(session) == "never"


def test_validate_approval_state_rejects_bad_values() -> None:
    from magitech_harness.approval.service import _validate_approval_state

    with pytest.raises(TypeError):
        _validate_approval_state("not a dict")
    with pytest.raises(ValueError):
        _validate_approval_state({"policy": "bogus", "turn_open": False})
    with pytest.raises(TypeError):
        _validate_approval_state({"policy": None, "turn_open": "yes"})
    assert _validate_approval_state({"policy": "ask", "turn_open": True}) == {
        "policy": "ask",
        "turn_open": True,
    }


def test_invalid_default_policy_rejected() -> None:
    with pytest.raises(ValueError, match="ask|never"):
        ApprovalService(Context(), {"policy": "bogus"})


@pytest.mark.asyncio
async def test_set_policy_invalid_and_noop() -> None:
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("policy-noop")
    agent = _agent(session)
    with pytest.raises(ValueError):
        service.set_policy(agent, "bogus")  # type: ignore[arg-type]
    service.set_policy(agent, "never")
    before = len(session.events)
    service.set_policy(agent, "never")
    assert len(session.events) == before  # no duplicate event


@pytest.mark.asyncio
async def test_request_without_agent_rejected() -> None:
    ctx = Context()
    service = ApprovalService(ctx)
    with pytest.raises(ValueError, match="requires an agent"):
        await service.request({"toolName": "x"})


@pytest.mark.asyncio
async def test_state_without_projection_registry_is_permissive() -> None:
    ctx = Context()
    service = ApprovalService(ctx)  # no sessionProjections row
    session = _session_with_turn("no-registry")
    service.set_handler(lambda _r: asyncio.sleep(0, result="allowed-once"))  # type: ignore[arg-type]
    assert (
        await service.request({"agent": _agent(session), "toolName": "x"})
        == "allowed-once"
    )


@pytest.mark.asyncio
async def test_handler_throwing_fails_closed() -> None:
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("handler-throw")

    async def _bad(_r):
        raise RuntimeError("boom")

    service.set_handler(_bad)
    assert (
        await service.request({"agent": _agent(session), "toolName": "x"})
        == "unavailable"
    )


@pytest.mark.asyncio
async def test_waterfall_throwing_fails_closed() -> None:
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("wf-throw")

    def _bad(req, next_fn):
        raise RuntimeError("boom")

    ctx.on("approval/request", _bad, mode=EventMode.WATERFALL)
    assert (
        await service.request({"agent": _agent(session), "toolName": "x"})
        == "unavailable"
    )


@pytest.mark.asyncio
async def test_signal_abort_mid_request_returns_cancelled() -> None:
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("mid-cancel")
    signal = AbortSignal()

    async def _slow(_r):
        await asyncio.sleep(10)
        return "allowed-once"

    service.set_handler(_slow)
    task = asyncio.create_task(
        service.request({"agent": _agent(session), "toolName": "x", "signal": signal})
    )
    await asyncio.sleep(0.01)
    signal.abort("user")
    assert await task == "cancelled"


def test_apply_approval_state_ignores_unrelated_and_invalid_policy() -> None:
    from magitech_harness.approval.service import APPROVAL_STATE_PROJECTION
    from magitech_harness.session.events import SessionEvent

    state = {"policy": None, "turn_open": False}
    other = SessionEvent(id="e1", session_id="s", type="user/message", payload={})
    assert APPROVAL_STATE_PROJECTION.apply(state, other) is state
    invalid = SessionEvent(
        id="e2",
        session_id="s",
        type=EventType.APPROVAL_POLICY,
        payload={"policy": "bogus"},
    )
    assert APPROVAL_STATE_PROJECTION.apply(state, invalid) is state


@pytest.mark.asyncio
async def test_answer_wins_when_signal_aborts_late() -> None:
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("late-abort")
    signal = AbortSignal()
    service.set_handler(lambda _r: asyncio.sleep(0, result="allowed-once"))  # type: ignore[arg-type]
    outcome = await service.request(
        {"agent": _agent(session), "toolName": "x", "signal": signal}
    )
    signal.abort("late")
    assert outcome == "allowed-once"


@pytest.mark.asyncio
async def test_double_abort_is_idempotent() -> None:
    ctx = Context()
    SessionProjectionRegistry(ctx)
    service = ApprovalService(ctx)
    session = _session_with_turn("double-abort")
    signal = AbortSignal()

    async def _slow(_r):
        await asyncio.sleep(10)
        return "allowed-once"

    service.set_handler(_slow)
    task = asyncio.create_task(
        service.request({"agent": _agent(session), "toolName": "x", "signal": signal})
    )
    await asyncio.sleep(0.01)
    signal.abort("first")
    signal.abort("second")  # second abort sees cancelled already done
    assert await task == "cancelled"
