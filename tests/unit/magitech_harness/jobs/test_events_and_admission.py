"""Unit tests for JobEventHub and install_job_archive_admission."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from anima import Context

from magitech_harness.jobs.archive_admission import install_job_archive_admission
from magitech_harness.jobs.events import JobEventHub, JobLayer
from magitech_harness.jobs.types import JobEvent, JobView
from magitech_harness.scope import ScopedLayers, create_scope


@dataclass(eq=False)
class _Owner:
    id: str
    ctx: Context


def _sample_job(job_id: str, owner_id: str | None = None) -> JobView:
    view: JobView = {
        "id": job_id,
        "kind": "bash",
        "label": "echo hi",
        "status": "running",
        "startedAt": 100,
        "output": {"total": 0, "earliest": 0},
    }
    if owner_id is not None:
        view["owner"] = owner_id
    return view


def test_job_layer_is_empty_and_hub_routing() -> None:
    root = Context()
    layer = JobLayer()
    assert layer.is_empty() is True
    undo_ctrl = layer.controllers.append(object())
    assert layer.is_empty() is False
    undo_ctrl()
    assert layer.is_empty() is True

    warnings: list[str] = []
    layers: ScopedLayers[JobLayer] = ScopedLayers(
        lambda _scope: JobLayer(), lambda: None
    )
    hub = JobEventHub(layers, warnings.append)

    key_a = object()
    key_a_child = object()
    key_b = object()
    scope_a = create_scope(root, key_a)
    scope_a_child = create_scope(scope_a.ctx, key_a_child, {"parent": key_a})
    scope_b = create_scope(root, key_b)

    seen_all: list[str] = []
    seen_s1: list[str] = []
    seen_scope_global: list[str] = []
    seen_scope_a: list[str] = []
    seen_scope_b: list[str] = []

    unsub_all = hub.subscribe(root, None, lambda ev: seen_all.append(ev["job"]["id"]))
    hub.subscribe(root, {"owner": "s1"}, lambda ev: seen_s1.append(ev["job"]["id"]))
    hub.subscribe(
        root,
        {"owners": "scope"},
        lambda ev: seen_scope_global.append(ev["job"]["id"]),
    )
    hub.subscribe(
        scope_a.ctx,
        {"owners": "scope"},
        lambda ev: seen_scope_a.append(ev["job"]["id"]),
    )
    hub.subscribe(
        scope_b.ctx,
        {"owners": "scope"},
        lambda ev: seen_scope_b.append(ev["job"]["id"]),
    )
    layer_a = layers.peek(key_a)
    assert layer_a is not None
    assert layer_a.is_empty() is False

    def _bad_listener(_ev: JobEvent) -> None:
        raise RuntimeError("boom")

    hub.subscribe(root, {"owners": "all"}, _bad_listener)

    owner_s1 = _Owner("s1", scope_a_child.ctx)
    owner_s2 = _Owner("s2", scope_b.ctx)

    hub.emit({"type": "registered", "job": _sample_job("j-s1", "s1")}, owner_s1)
    hub.emit({"type": "registered", "job": _sample_job("j-s2", "s2")}, owner_s2)
    hub.emit({"type": "registered", "job": _sample_job("j-open")}, None)

    assert seen_all == ["j-s1", "j-s2", "j-open"]
    assert seen_s1 == ["j-s1", "j-open"]
    assert seen_scope_global == ["j-s1", "j-s2", "j-open"]
    assert seen_scope_a == ["j-s1"]
    assert seen_scope_b == ["j-s2"]
    assert len(warnings) == 3

    # Double unsubscribe / dispose safety
    unsub_all()
    unsub_all()
    unsub_unscoped = hub.subscribe(root, {"owners": "all"}, lambda _ev: None)
    unsub_unscoped()
    unsub_unscoped()
    hub.emit({"type": "registered", "job": _sample_job("j-after")}, None)
    assert seen_all == ["j-s1", "j-s2", "j-open"]


async def test_install_job_archive_admission() -> None:
    root = Context()
    registry = MagicMock()
    registry.list.side_effect = lambda caller: [
        _sample_job("j-run", "s1"),
        {**_sample_job("j-done", "s1"), "status": "completed"},
        _sample_job("j-unowned", None),
    ]

    kill_calls: list[tuple[str, str, str]] = []

    def _kill(job_id: str, caller: str, reason: str) -> str:
        kill_calls.append((job_id, caller, reason))
        if job_id == "j-fail":
            raise RuntimeError("cannot kill")
        return "requested"

    registry.kill.side_effect = _kill

    dispose = install_job_archive_admission(root, registry)

    # Downstream fallback handler on workspace/session-activity
    root.on("workspace/session-activity", lambda _req, _next: [{"kind": "other"}])

    activity = await root.waterfall("workspace/session-activity", {"sessionId": "s1"})
    assert activity == [
        {"kind": "job", "items": [{"id": "j-run", "label": "echo hi"}]},
        {"kind": "other"},
    ]

    # When no jobs match owner s2, returns downstream only
    empty_activity = await root.waterfall(
        "workspace/session-activity", {"sessionId": "s2"}
    )
    assert empty_activity == [{"kind": "other"}]

    # Stop contains exceptions across jobs
    registry.list.side_effect = lambda _caller: [
        _sample_job("j-fail", "s1"),
        _sample_job("j-ok", "s1"),
    ]
    await root.parallel("workspace/session-stop", {"sessionId": "s1"})
    assert kill_calls == [
        ("j-fail", "s1", "session archived"),
        ("j-ok", "s1", "session archived"),
    ]

    dispose()
