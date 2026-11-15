"""Unit tests for JobRegistry and LocalJobRegistry."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from anima import Context, Service

from magitech_harness.agent.cancellation import AbortController
from magitech_harness.jobs.local import LocalJobRegistry
from magitech_harness.jobs.service import JobRegistry
from magitech_harness.jobs.types import (
    TASK_WAIT_TIMEOUT,
    JobEvent,
    JobHandle,
    JobOutcome,
    JobOutputSourceRead,
    JobRunHooks,
)
from magitech_harness.scope import create_scope


@dataclass(eq=False)
class _FakeSession:
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fail_append: bool = False

    def append(self, event_type: str, data: dict[str, Any]) -> None:
        if self.fail_append:
            raise RuntimeError("disk full")
        self.events.append((event_type, data))


@dataclass(eq=False)
class _FakeAgent:
    id: str
    ctx: Context
    session: _FakeSession = field(default_factory=_FakeSession)


class _FakeAgents(Service):
    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "agents")
        self._store: dict[str, _FakeAgent] = {}

    def add(self, agent: _FakeAgent) -> None:
        self._store[agent.id] = agent

    def get(self, session_id: str) -> _FakeAgent | None:
        return self._store.get(session_id)


class _ManualHooks:
    def __init__(
        self,
        cancel_error: Exception | None = None,
        *,
        auto_settle_on_cancel: bool = True,
    ) -> None:
        self.done: asyncio.Future[JobOutcome] = (
            asyncio.get_running_loop().create_future()
        )
        self.cancel_calls: list[str | None] = []
        self._cancel_error = cancel_error
        self._auto_settle_on_cancel = auto_settle_on_cancel

    def cancel(self, reason: str | None = None) -> None:
        self.cancel_calls.append(reason)
        if self._cancel_error is not None:
            raise self._cancel_error
        if self._auto_settle_on_cancel and not self.done.done():
            self.done.set_result({"status": "killed"})


class _PullSource:
    def __init__(
        self,
        channel: str | None,
        fn: Any,
    ) -> None:
        self._channel = channel
        self._fn = fn

    @property
    def channel(self) -> str | None:
        return self._channel

    def read(self, from_byte: int) -> JobOutputSourceRead:
        return self._fn(from_byte)


async def test_abstract_registry_cannot_be_instantiated_and_bad_configs() -> None:
    root = Context()
    with pytest.raises(TypeError):
        JobRegistry(root)  # type: ignore[abstract]

    for bad_cfg in (
        {"maxConcurrentJobsPerOwner": 0},
        {"retainBytes": True},
        {"settledRetainBytes": -1},
        {"pumpPollMs": "10"},
    ):
        with pytest.raises(ValueError, match="invalid "):
            LocalJobRegistry(Context(), bad_cfg)


async def test_preflight_validation_and_owner_resolution() -> None:
    root = Context()
    reg = LocalJobRegistry(root, {"maxConcurrentJobsPerOwner": 1})

    def _run_noop(_h: JobHandle) -> JobRunHooks:
        return _ManualHooks()

    # No controller yet
    with pytest.raises(RuntimeError, match="no job controller"):
        reg.start({"kind": "bash", "label": "hi", "run": _run_noop})

    # Owned without agents service
    with pytest.raises(RuntimeError, match="requires the agent registry"):
        reg.start({"kind": "bash", "label": "hi", "owner": "s1", "run": _run_noop})

    agents = _FakeAgents(root)
    with pytest.raises(RuntimeError, match="has no live agent"):
        reg.start({"kind": "bash", "label": "hi", "owner": "s1", "run": _run_noop})

    detach = reg.attach_controller("ctrl")
    with pytest.raises(ValueError, match="invalid job kind"):
        reg.start({"kind": "", "label": "hi", "run": _run_noop})
    with pytest.raises(ValueError, match="invalid job label"):
        reg.start({"kind": "bash", "label": "", "run": _run_noop})
    for bad_limit in (0, -2, True):
        with pytest.raises(ValueError, match="invalid outputLimitBytes"):
            reg.start(
                {
                    "kind": "bash",
                    "label": "hi",
                    "outputLimitBytes": bad_limit,  # type: ignore[typeddict-item]
                    "run": _run_noop,
                }
            )

    def _run_throws(_h: JobHandle) -> JobRunHooks:
        raise RuntimeError("spawn failed")

    with pytest.raises(RuntimeError, match="spawn failed"):
        reg.start({"kind": "bash", "label": "hi", "run": _run_throws})
    assert reg.list() == []

    # Concurrent job limit
    j1 = reg.start(
        {"kind": "bash", "label": "first", "outputLimitBytes": 128, "run": _run_noop}
    )
    assert reg.get(j1)["outputLimitBytes"] == 128
    with pytest.raises(RuntimeError, match="background job limit reached"):
        reg.start({"kind": "bash", "label": "second", "run": _run_noop})

    # Unknown job access
    with pytest.raises(KeyError, match="unknown job"):
        reg.get("missing")

    # Foreign job access
    agent_ctx = create_scope(root, object()).ctx
    agent = _FakeAgent("s1", agent_ctx)
    agents.add(agent)
    j2 = reg.start({"kind": "bash", "label": "owned", "owner": "s1", "run": _run_noop})
    with pytest.raises(PermissionError, match="another session"):
        reg.get(j2, caller="s2")
    with pytest.raises(PermissionError, match="another session"):
        reg.get(j2, caller=None)
    assert [v["id"] for v in reg.list("s1")] == [j1, j2]
    assert [v["id"] for v in reg.list(None)] == [j1]

    detach()
    await root.fiber.dispose()


async def test_staged_writes_uuid_collision_ring_cursor_and_kill_merge() -> None:
    root = Context()
    reg = LocalJobRegistry(
        root, {"retainBytes": 10, "settledRetainBytes": 4, "pumpPollMs": 20}
    )
    agents = _FakeAgents(root)
    agent = _FakeAgent("s1", create_scope(root, object()).ctx)
    agents.add(agent)
    reg.attach_controller("ctrl")

    events: list[JobEvent] = []
    reg.events.subscribe({"owners": "all"}, events.append)

    hooks_list: list[_ManualHooks] = []
    handles: list[JobHandle] = []

    def _run_staged(h: JobHandle) -> JobRunHooks:
        handles.append(h)
        assert h.id.startswith("bash-")
        h.update_progress("init")
        h.append("pre-", {"channel": "stdout"})
        h.append("", None)
        hk = _ManualHooks(auto_settle_on_cancel=False)
        hooks_list.append(hk)
        return hk

    u1 = uuid.UUID("11111111111111111111111111111111")
    u2 = uuid.UUID("22222222222222222222222222222222")
    with patch("magitech_harness.jobs.local.uuid.uuid4", side_effect=[u1, u1, u2]):
        jid1 = reg.start(
            {
                "kind": "bash",
                "label": "staged",
                "owner": "s1",
                "toolCall": {"callId": "c1", "rootCallId": "r1"},
                "run": _run_staged,
            }
        )
        jid2 = reg.start({"kind": "bash", "label": "second", "run": _run_staged})

    assert jid1 == "bash-1111111111111111"
    assert jid2 == "bash-2222222222222222"
    assert reg.get(jid1, "s1")["progress"] == "init"
    assert [e[0] for e in agent.session.events] == ["job/registered"]

    # Invalid read_at offsets
    for bad_off in (-1, True):
        with pytest.raises(ValueError, match="invalid output read offset"):
            reg.read_at(jid1, bad_off, "s1")  # type: ignore[arg-type]

    # Removing live job fails
    with pytest.raises(RuntimeError, match="still running"):
        reg.remove(jid1, "s1")

    handles[0].update_progress("step2")
    handles[0].append("1234567890", {"channel": "stdout"})
    # Live retainBytes=10 dropped "pre-"
    non_consuming = reg.read_at(jid1, 0, "s1")
    assert non_consuming["lossy"] is True
    assert non_consuming["chunks"] == [
        {"at": 4, "text": "1234567890", "channel": "stdout"}
    ]
    # Live read before settlement returns no result yet and leaves ring live
    live_read = reg.read(jid1, "s1")
    assert "result" not in live_read
    assert live_read["job"]["status"] == "running"

    # Kill with reason merges on killed outcome
    assert reg.kill(jid1, "s1", "user stop") == "requested"
    hooks_list[0].done.set_result(
        {"status": "killed", "detail": "SIGTERM", "result": "partial"}
    )
    await asyncio.sleep(0)
    # Re-seed a byte via new append? Instead read2 checks settled trim on empty cursor!
    assert reg.get(jid1, "s1")["detail"] == "SIGTERM; user stop"
    assert "progress" not in reg.get(jid1, "s1")
    # Post-settlement append & progress drop safely
    handles[0].append("late")
    handles[0].update_progress("late")

    # First post-settlement read delivers result once & trims to settledRetainBytes=4
    r1 = reg.read(jid1, "s1")
    assert r1["result"] == "partial"
    r2 = reg.read(jid1, "s1")
    assert "result" not in r2
    assert r2["job"]["output"]["earliest"] == 10

    assert reg.kill(jid1, "s1") == "already-finished"
    reg.remove(jid1, "s1")
    assert any(ev["type"] == "removed" and ev["job"]["id"] == jid1 for ev in events)

    # Kill with reason without producer detail, vs completed outcome
    hooks_list[1].done.set_result({"status": "completed", "detail": "ok"})
    await reg.wait(jid2, 500)
    assert reg.get(jid2)["detail"] == "ok"

    # Kill without producer detail -> killReason alone
    hk3 = _ManualHooks(auto_settle_on_cancel=False)
    jid3 = reg.start({"kind": "bash", "label": "k3", "run": lambda _h: hk3})
    reg.kill(jid3, None, "user abort")
    # Second kill without reason preserves prior killReason
    reg.kill(jid3, None, None)
    hk3.done.set_result({"status": "killed"})
    await reg.wait(jid3, 500)
    assert reg.get(jid3)["detail"] == "user abort"

    await root.fiber.dispose()


async def test_wait_settlement_timeout_abort_and_pull_sources() -> None:
    root = Context()
    reg = LocalJobRegistry(root, {"pumpPollMs": 5})
    reg.attach_controller("ctrl")

    hk = _ManualHooks()
    reads = 0

    def _pull(from_byte: int) -> JobOutputSourceRead:
        nonlocal reads
        reads += 1
        if reads == 1:
            return {
                "text": "abc",
                "nextOffset": 3,
                "lossy": False,
                "spillPath": "/tmp/a.log",
            }
        if reads == 2:
            return {"text": "def", "nextOffset": 6, "lossy": False}
        raise RuntimeError("stream exploded")

    def _ok_pull(_from_byte: int) -> JobOutputSourceRead:
        return {"text": "", "nextOffset": 0, "lossy": False}

    events: list[JobEvent] = []
    reg.events.subscribe({"owners": "all"}, events.append)

    jid = reg.start(
        {
            "kind": "bash",
            "label": "pumped",
            "output": [
                _PullSource("stdout", _pull),
                _PullSource(None, _ok_pull),
            ],
            "run": lambda _h: hk,
        }
    )
    assert reg.get(jid)["output"]["spillPaths"] == ["/tmp/a.log"]

    for bad_t in (0, -1, True, float("inf")):
        with pytest.raises(ValueError, match="invalid wait timeout"):
            await reg.wait(jid, bad_t)  # type: ignore[arg-type]

    # Wait timeout returns live projection and records awaited=False when settled later
    timed = await reg.wait(jid, 15)
    assert timed["status"] == "running"
    assert "spillPaths" not in timed["output"]

    # Already aborted signal fails fast
    aborted_ctrl = AbortController()
    aborted_ctrl.abort("already")
    with pytest.raises(RuntimeError, match="wait aborted"):
        await reg.wait(jid, 1000, signal=aborted_ctrl.signal)

    # Aborting live wait rejects caller
    live_ctrl = AbortController()
    wait_task = asyncio.create_task(reg.wait(jid, 5000, signal=live_ctrl.signal))
    await asyncio.sleep(0)
    live_ctrl.abort("stop waiting")
    with pytest.raises(RuntimeError, match="wait aborted"):
        await wait_task

    # Live wait resolved by settlement sets awaited=True
    settled_wait = asyncio.create_task(reg.wait(jid, 5000))
    await asyncio.sleep(0)
    hk.done.set_result({"status": "completed"})
    done_view = await settled_wait
    assert done_view["status"] == "completed"
    settled_events = [e for e in events if e["type"] == "settled"]
    assert len(settled_events) == 1
    assert settled_events[0].get("awaited") is True

    # Wait on already-settled job returns immediately (even with aborted signal)
    assert (await reg.wait(jid, 10, signal=aborted_ctrl.signal))[
        "status"
    ] == "completed"
    # Same-tick settlement + abort does not crash _on_abort (synchronous unsub)
    hk_race = _ManualHooks()
    jid_race = reg.start({"kind": "bash", "label": "race", "run": lambda _h: hk_race})
    race_ctrl = AbortController()
    unsub_race = reg.events.subscribe(
        {"owners": "all"},
        lambda ev: (
            race_ctrl.abort("same-tick-after-settle")
            if ev["type"] == "settled" and ev["job"]["id"] == jid_race
            else None
        ),
    )
    wait_race = asyncio.create_task(reg.wait(jid_race, 5000, signal=race_ctrl.signal))
    await asyncio.sleep(0)
    hk_race.done.set_result({"status": "completed"})
    assert (await wait_race)["status"] == "completed"
    unsub_race()
    # Check Task wait constant symbol imported cleanly
    assert TASK_WAIT_TIMEOUT == "TASK_WAIT_TIMEOUT"

    await root.fiber.dispose()


async def test_teardown_containment_done_rejection_and_owner_disposal() -> None:
    root = Context()
    reg_fiber = root.use(LocalJobRegistry)
    await reg_fiber
    reg: LocalJobRegistry = root.jobs
    agents = _FakeAgents(root)

    # Scoped controller serves child scope and descendant, NOT sibling scope
    key_a = object()
    key_b = object()
    key_ac = object()
    scope_a = create_scope(root, key_a)
    scope_b = create_scope(root, key_b)
    scope_a_child = create_scope(scope_a.ctx, key_ac, {"parent": key_a})
    scope_a.ctx.jobs.attach_controller("scoped-ctrl")

    agent_a = _FakeAgent("sa", scope_a_child.ctx)
    agent_a.session.fail_append = True  # verify session.append error containment
    agent_b = _FakeAgent("sb", scope_b.ctx)
    agents.add(agent_a)
    agents.add(agent_b)

    with pytest.raises(RuntimeError, match="no job controller"):
        reg.start(
            {
                "kind": "bash",
                "label": "b",
                "owner": "sb",
                "run": lambda _h: _ManualHooks(),
            }
        )

    # Throwing teardown cancel force-fails record while late pump/done settle safely
    throwing_hooks = _ManualHooks(
        cancel_error=RuntimeError("unstoppable"), auto_settle_on_cancel=False
    )
    reg.start(
        {
            "kind": "bash",
            "label": "a",
            "owner": "sa",
            "output": [
                _PullSource(
                    "stdout",
                    lambda off: {
                        "text": "late-pump",
                        "nextOffset": off + 9,
                        "lossy": False,
                    },
                )
            ],
            "run": lambda _h: throwing_hooks,
        }
    )

    # Also test producer done rejection containment
    rej_hooks = _ManualHooks()
    jid_rej = reg.start(
        {"kind": "bash", "label": "rej", "owner": "sa", "run": lambda _h: rej_hooks}
    )
    rej_hooks.done.set_exception(RuntimeError("crash"))
    await reg.wait(jid_rej, 500, caller="sa")
    assert reg.get(jid_rej, "sa")["status"] == "failed"

    # Dispose owner scope_a cleanly (cancels throwing_hooks -> forced failed)
    await scope_a.dispose()
    assert reg.list("sa") == []
    # Late producer settlement after forced teardown settlement is ignored (first-wins)
    throwing_hooks.done.set_result({"status": "completed"})
    await asyncio.sleep(0)

    # Mount another live compliant job on scope_c with global controller and dispose only reg_fiber
    # so _dispose_all cancels live job and detaches cross-fiber owner cleanup cleanly.
    reg.attach_controller("global-ctrl")
    compliant_hooks = _ManualHooks()
    agent_c = _FakeAgent("sc", create_scope(root, object()).ctx)
    agents.add(agent_c)
    reg.start(
        {"kind": "bash", "label": "c", "owner": "sc", "run": lambda _h: compliant_hooks}
    )
    reg.start(
        {"kind": "bash", "label": "c-done", "owner": "sc", "run": lambda _h: rej_hooks}
    )

    await reg_fiber.dispose()

    # Dispose a registry with zero owned jobs (empty owner_cleanups branch)
    empty_fiber = root.use(LocalJobRegistry)
    await empty_fiber
    await empty_fiber.dispose()
