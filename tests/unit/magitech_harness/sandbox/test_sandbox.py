"""Tests for the bwrap-backed sandbox seam and policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.sandbox import escalation as esc
from magitech_harness.sandbox.escalation import SandboxEscalationError
from magitech_harness.sandbox.policy import SandboxPolicyService
from magitech_harness.sandbox.policy import apply as apply_policy
from magitech_harness.sandbox.roots import (
    additional_root,
    canonical_path,
    workspace_writable_roots,
    writable_roots,
)
from magitech_harness.sandbox.service import SandboxService, apply
from magitech_harness.sandbox.types import (
    SANDBOX_UNAVAILABLE,
    RunnerFailureRule,
    SandboxError,
    SandboxInvalidRootError,
    SandboxPreparationError,
    SandboxUnavailableError,
)
from magitech_harness.shell.bash_sandbox import (
    SandboxBashExecutor,
    classify_runner_failure,
    matches_signature,
)


def test_types_and_roots(tmp_path: Path) -> None:
    assert SANDBOX_UNAVAILABLE == "SANDBOX_UNAVAILABLE"
    assert "no sandbox backend" in str(SandboxUnavailableError("read-only"))
    assert "Runner failure: bad" in str(SandboxUnavailableError("read-only", "bad"))
    assert "additional writable directory" in str(SandboxInvalidRootError("x", "bad"))
    assert SandboxPreparationError("bad").code == "SANDBOX_PREPARATION_FAILED"
    assert canonical_path(str(tmp_path)) == str(tmp_path.resolve())

    directory = tmp_path / "writable"
    directory.mkdir()
    assert additional_root(str(directory)) == str(directory.resolve())
    with pytest.raises(SandboxInvalidRootError):
        additional_root(str(tmp_path / "missing"))
    file = tmp_path / "file.md"
    file.write_text("x", encoding="utf-8")
    with pytest.raises(SandboxInvalidRootError):
        additional_root(str(file))
    link = tmp_path / "link"
    try:
        link.symlink_to(directory, target_is_directory=True)
        with pytest.raises(SandboxInvalidRootError):
            additional_root(str(link))
    except OSError:
        pass

    policy = {"mode": "read-only", "workspaceRoot": str(tmp_path)}
    assert workspace_writable_roots(policy) == []
    policy["mode"] = "workspace-write"
    assert workspace_writable_roots(policy) == [str(tmp_path.resolve())]
    assert "/tmp" in writable_roots(policy)


def test_escalation_validation_and_markers() -> None:
    esc.validate_escalation_args(None, None)
    with pytest.raises(SandboxEscalationError):
        esc.validate_escalation_args("workspace-write", None)
    with pytest.raises(SandboxEscalationError):
        esc.validate_escalation_args(None, "because")
    with pytest.raises(SandboxEscalationError):
        esc.validate_escalation_args("workspace-write", "  ")
    assert esc.sandbox_denial_marker("read-only") == (
        "[sandbox: file access denied under read-only mode]"
    )
    assert "escalation available" in esc.escalation_hint_marker("command")
    assert "one-shot retry" in esc.sandbox_permissions_description("command")
    assert esc.WIDER_MODES["read-only"] == ("workspace-write", "danger-full-access")
    assert esc.ESCALATION_TARGETS == ("workspace-write", "danger-full-access")
    assert esc.escalation_display_reason("read-only", "why") == (
        "Allow this operation with read-only permissions: why"
    )


@pytest.mark.asyncio
async def test_escalation_approval() -> None:
    request = {
        "requestedMode": "workspace-write",
        "effectiveMode": "read-only",
        "justification": "need writes",
        "subject": "command",
    }
    agent = object()

    async def allow_scenario(outcome: str, **extra: Any) -> Any:
        seen: dict[str, Any] = {}

        def approver(payload: dict[str, Any]) -> dict[str, str]:
            seen.update(payload)
            return {"status": outcome}

        try:
            mode = await esc.approve_escalation(
                request, {"approver": approver, "agent": agent, **extra}
            )
            return mode
        except SandboxEscalationError as error:
            return error

    assert await allow_scenario("allowed-once") == "workspace-write"
    rejected = await allow_scenario("rejected")
    assert isinstance(rejected, SandboxEscalationError)
    unavailable = await allow_scenario("unavailable")
    assert isinstance(unavailable, SandboxEscalationError)
    same = await esc.approve_escalation(
        {**request, "requestedMode": "read-only"},
        {"approver": lambda _p: {"status": "rejected"}, "agent": agent},
    )
    assert same == "read-only"

    with pytest.raises(SandboxEscalationError):
        await esc.approve_escalation(request, {"agent": agent})
    assert (
        await esc.approve_escalation(
            {
                **request,
                "requestedMode": "danger-full-access",
                "effectiveMode": "danger-full-access",
            },
            {"approver": lambda _p: {"status": "rejected"}, "agent": agent},
        )
        == "danger-full-access"
    )
    with pytest.raises(SandboxEscalationError):
        await esc.approve_escalation(
            {
                **request,
                "requestedMode": "read-only",
                "effectiveMode": "workspace-write",
            },
            {"approver": lambda _p: {"status": "rejected"}, "agent": agent},
        )
    with pytest.raises(SandboxEscalationError):
        await esc.approve_escalation(
            request, {"approver": lambda _p: {"status": "weird"}, "agent": agent}
        )


def test_sandbox_service() -> None:
    ctx = Context()
    for config in (
        {"runnerFailureSignatures": ["x"]},
        {"runnerCommand": ["runner"]},
        {"runnerCommand": ["runner"], "runnerFailureSignatures": ["a\n b"]},
        {"probeTimeoutMs": 0},
    ):
        with pytest.raises(SandboxError):
            SandboxService(ctx, config)
    service = apply(ctx)
    assert isinstance(service, SandboxService)
    assert service.bwrap_path is not None

    policy = {"mode": "read-only", "workspaceRoot": "/tmp"}
    confined = service.confine(["bash", "-c", "echo hi"], policy)
    assert confined.argv[0] == service.bwrap_path
    assert confined.argv[-3:] == ["bash", "-c", "echo hi"]
    assert confined.enforcement == "full"
    with pytest.raises(SandboxError):
        service.confine(["bash", "-c", "x"], {**policy, "mode": "danger-full-access"})

    runner = SandboxService(
        ctx,
        {"runnerCommand": ["runner"], "runnerFailureSignatures": ["broken"]},
    )
    wrapped = runner.confine(["bash", "-c", "x"], policy)
    assert wrapped.argv[:2] == ["runner", "--"]

    class Aborted:
        aborted = True

        @staticmethod
        def throw_if_aborted() -> None:
            raise RuntimeError("aborted")

    with pytest.raises(RuntimeError):
        service.confine(["bash"], policy, Aborted())


@pytest.mark.asyncio
async def test_policy_service(tmp_path: Path) -> None:
    ctx = Context()
    for config in (
        {"mode": "bad", "workspaceRoot": "/tmp"},
        {"workspaceRoot": "relative"},
        {"workspaceRoot": "/tmp", "additionalWritableRoots": ["relative"]},
    ):
        with pytest.raises(SandboxError):
            SandboxPolicyService(ctx, config)

    service: SandboxPolicyService = apply_policy(
        ctx, {"mode": "workspace-write", "workspaceRoot": str(tmp_path)}
    )
    resolved = service.resolve()
    assert resolved["mode"] == "workspace-write"
    assert resolved["workspaceRoot"] == str(tmp_path)

    session = SimpleNamespace(
        session_id="s1",
        header={"cwd": str(tmp_path)},
        append=lambda event_type, payload: None,
    )
    assert service.resolve(session)["sessionId"] == "s1"
    assert service._render_context({"session": session}).startswith(
        "Current DSH file policy"
    )
    service.set_mode(session, "read-only")

    directory = tmp_path / "extra"
    directory.mkdir()
    dispose = service.register_writable_root(session, str(directory))
    assert str(directory) in service.resolve(session).get("additionalWritableRoots", [])
    dispose()
    with pytest.raises(SandboxError):
        service.resolve(SimpleNamespace(session_id="bad", header={"cwd": "relative"}))


def test_sandbox_fact_helpers() -> None:
    assert (
        matches_signature(0, "read-only file system", ("read-only file system",))
        is False
    )
    assert (
        matches_signature(1, "Read-Only File System", ("read-only file system",))
        is True
    )
    rule = RunnerFailureRule(("fatal",), (1,), ("info",))
    assert (
        classify_runner_failure(1, "INFO\nFATAL breakdown", (rule,))
        == "fatal breakdown"
    )
    assert classify_runner_failure(2, "fatal", (rule,)) is None
    assert classify_runner_failure(1, "info", (rule,)) is None


@pytest.mark.asyncio
async def test_real_bwrap_enforcement(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path / "sandbox")
    await ctx.flush()
    result = await ctx.shell.run("echo confined", cwd=str(tmp_path))
    assert result.stdout.strip() == "confined"
    assert result.sandbox is not None
    assert result.sandbox["enforcement"] == "full"
    assert result.sandbox["denied"] is False

    denied = await ctx.shell.run("touch /etc/mth-sandbox-forbidden", cwd=str(tmp_path))
    assert denied.sandbox is not None
    assert denied.sandbox["denied"] is True


@pytest.mark.asyncio
async def test_escalation_edge_paths() -> None:
    request = {
        "requestedMode": "workspace-write",
        "effectiveMode": "read-only",
        "justification": "why",
        "subject": "command",
    }
    with pytest.raises(SandboxEscalationError):
        await esc.approve_escalation(request, {"approver": lambda _p: {"status": "ok"}})

    async def async_approve(_payload: Any) -> dict[str, str]:
        return {"status": "allowed-once"}

    assert (
        await esc.approve_escalation(
            request, {"approver": async_approve, "agent": object()}
        )
        == "workspace-write"
    )

    class Aborted:
        def throw_if_aborted(self) -> None:
            raise RuntimeError("aborted")

    with pytest.raises(RuntimeError):
        await esc.approve_escalation(
            request,
            {
                "approver": lambda _p: {"status": "cancelled"},
                "agent": object(),
                "signal": Aborted(),
            },
        )
    with pytest.raises(SandboxEscalationError):
        await esc.approve_escalation(
            request,
            {"approver": lambda _p: {"status": "cancelled"}, "agent": object()},
        )


@pytest.mark.asyncio
async def test_policy_edge_paths(tmp_path: Path) -> None:
    bare = Context()
    service = SandboxPolicyService(bare, {"workspaceRoot": str(tmp_path)})
    with pytest.raises(RuntimeError):
        await service.init()
    with pytest.raises(TypeError):
        service._validate_state("bad")
    assert service._validate_state(None) is None
    assert service._validate_state("read-only") == "read-only"
    event = SimpleNamespace(type="sandbox/mode", payload={"mode": "workspace-write"})
    assert service._apply_event(None, event) == "workspace-write"
    assert (
        service._apply_event("read-only", SimpleNamespace(type="other", payload={}))
        == "read-only"
    )
    with pytest.raises(SandboxError):
        service.register_writable_root(object(), "relative")
    with pytest.raises(SandboxError):
        service.set_mode(object(), "bogus")

    service2 = SandboxPolicyService(
        bare, {"mode": "workspace-write", "workspaceRoot": str(tmp_path)}
    )
    bare.provide("workingDirectory", SimpleNamespace(get=lambda s: None))
    session = SimpleNamespace(session_id="s", header={})
    assert service2.resolve(session)["workspaceRoot"] == str(tmp_path)
    assert service2.resolve(SimpleNamespace(session_id="x", header={}))[
        "workspaceRoot"
    ] == str(tmp_path)
    bare.provide(
        "sessionProjections",
        SimpleNamespace(
            register=lambda d: SimpleNamespace(
                dispose=lambda: None, state_of=lambda s: None
            )
        ),
    )
    service3 = SandboxPolicyService(
        bare, {"mode": "workspace-write", "workspaceRoot": str(tmp_path)}
    )
    await service3.init()
    assert service3._override(None) is None
    assert service3.resolve(session)["workspaceRoot"] == str(tmp_path)


def test_roots_and_service_edges(monkeypatch: Any, tmp_path: Path) -> None:
    from pathlib import Path as P

    original = P.resolve

    def failing(self: P, *args: Any, **kwargs: Any) -> P:
        raise OSError("nope")

    monkeypatch.setattr(P, "resolve", failing)
    assert canonical_path("x") == "x"
    monkeypatch.setattr(P, "resolve", original)

    directory = tmp_path / "extra"
    directory.mkdir()
    policy = {
        "mode": "workspace-write",
        "workspaceRoot": str(tmp_path),
        "additionalWritableRoots": [str(directory)],
    }
    assert str(directory.resolve()) in workspace_writable_roots(policy)
    assert writable_roots({"mode": "read-only", "workspaceRoot": "/"}) == []

    service = SandboxService(Context())
    service.bwrap_path = None
    with pytest.raises(SandboxUnavailableError):
        service.confine(["bash"], {"mode": "read-only", "workspaceRoot": "/"})


@pytest.mark.asyncio
async def test_bash_sandbox_full_access_and_facts(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "full")
    await ctx.flush()
    executor = ctx.get("shell", None, strict=False)
    assert isinstance(executor, SandboxBashExecutor)
    assert executor.sandbox_mode == "workspace-write"

    spec = executor.resolve({"command": "echo hi"})
    spec["sandboxPolicy"] = {
        "mode": "danger-full-access",
        "workspaceRoot": str(tmp_path),
    }
    process = await executor.execute(spec)
    result = await process.result()
    assert result["sandbox"] == {"mode": "danger-full-access", "denied": False}

    bare = Context()
    no_policy = SandboxBashExecutor.__new__(SandboxBashExecutor)
    no_policy.ctx = bare
    assert no_policy.sandbox_mode is None


@pytest.mark.asyncio
async def test_policy_context_and_root_fallback(tmp_path: Path) -> None:
    ctx = Context()
    sections: list[Any] = []
    ctx.provide(
        "sessionProjections",
        SimpleNamespace(
            register=lambda d: SimpleNamespace(
                dispose=lambda: None, state_of=lambda s: None
            )
        ),
    )
    ctx.provide(
        "systemPrompt",
        SimpleNamespace(
            context=lambda section: sections.append(section),
            get_context_order=lambda name: 110,
        ),
    )
    service = SandboxPolicyService(
        ctx, {"mode": "workspace-write", "workspaceRoot": str(tmp_path)}
    )
    dispose = await service.init()
    assert sections[0]["name"] == "sandbox:policy"
    assert sections[0]["text"]({"session": None}).startswith("Current DSH file policy")
    session = SimpleNamespace(session_id="s", header={})
    assert service.resolve(session)["workspaceRoot"] == str(tmp_path)
    dispose()


@pytest.mark.asyncio
async def test_bash_sandbox_runner_failure(tmp_path: Path, monkeypatch: Any) -> None:
    from magitech_harness.sandbox.types import ConfinedArgv, RunnerFailureRule

    ctx = create_harness(storage_dir=tmp_path / "runner-fail")
    await ctx.flush()
    executor = ctx.get("shell", None, strict=False)
    spec = executor.resolve({"command": "echo broken >&2; exit 1"})
    policy = {"mode": "read-only", "workspaceRoot": str(tmp_path)}
    spec["sandboxPolicy"] = policy
    from magitech_harness.sandbox.service import SandboxService

    original = SandboxService.confine

    def confine(self: Any, argv: Any, _policy: Any, signal: Any = None) -> ConfinedArgv:
        result = original(self, argv, _policy, signal)
        return ConfinedArgv(
            result.argv,
            result.enforcement,
            result.denial_signatures,
            (RunnerFailureRule(("broken",)),),
        )

    monkeypatch.setattr(SandboxService, "confine", confine)
    process = await executor.execute(spec)
    with pytest.raises(SandboxUnavailableError):
        await process.result()

    # No workspace root from workingDirectory or header -> deployment fallback.
    policy_service = ctx.sandboxPolicy
    probe_session = ctx.sessions.create("policy-fallback")
    assert policy_service.resolve(probe_session)["workspaceRoot"]
