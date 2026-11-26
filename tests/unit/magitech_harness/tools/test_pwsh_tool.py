"""Tests for the model-facing PowerShell tool."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness.tools.pwsh_tool import (
    PWSH_DESCRIPTION,
    PWSH_SECTION,
    apply,
    handle_pwsh,
)

PWSH_BIN = shutil.which("pwsh") or "/usr/bin/pwsh"


@pytest.mark.asyncio
async def test_pwsh_execution(tmp_path: Any) -> None:
    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    result = await handle_pwsh(
        {"description": "add", "command": "20+22"}, ctx=ctx, spill_dir=tmp_path
    )
    assert "42" in result

    result = await handle_pwsh(
        {"command": "Write-Output 'hi'; Write-Error 'bad'"}, ctx=ctx, spill_dir=tmp_path
    )
    assert "hi" in result and "[stderr]" in result

    result = await handle_pwsh({"command": "exit 7"}, ctx=ctx, spill_dir=tmp_path)
    assert "[exit code: 7]" in result

    workdir = tmp_path / "work"
    workdir.mkdir()
    result = await handle_pwsh(
        {"command": "(Get-Location).Path", "workdir": str(workdir)},
        ctx=ctx,
        spill_dir=tmp_path,
    )
    assert str(workdir) in result


@pytest.mark.asyncio
async def test_pwsh_validation_and_errors(tmp_path: Any) -> None:
    assert "command" in await handle_pwsh({})
    assert "command" in await handle_pwsh({"command": "   "})
    assert "timeoutMs" in await handle_pwsh({"command": "1", "timeoutMs": "x"})
    assert "Working directory" in await handle_pwsh(
        {"command": "1", "workdir": str(tmp_path / "missing")}
    )
    assert "1" in await handle_pwsh({"command": "1"}, ctx=None)

    old = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = "/nonexistent"
        result = await handle_pwsh({"command": "1"}, ctx=None)
        assert "could not start PowerShell" in result
    finally:
        os.environ["PATH"] = old


@pytest.mark.asyncio
async def test_pwsh_timeout_and_spill(tmp_path: Any) -> None:
    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    result = await handle_pwsh(
        {"command": "Start-Sleep -Seconds 2", "timeoutMs": 50},
        ctx=ctx,
        spill_dir=tmp_path,
    )
    assert "timed out" in result

    result = await handle_pwsh(
        {"command": "'x' * 100000", "command_timeout": 1},
        ctx=ctx,
        spill_dir=tmp_path,
        max_output_bytes=10,
    )
    assert "truncated" in result


@pytest.mark.asyncio
async def test_pwsh_apply() -> None:

    ctx = Context()
    registered: list[Any] = []
    ctx.provide("tools", SimpleNamespace(register=registered.append))
    ctx.provide("pwshConfig", PWSH_BIN)
    apply(ctx)
    assert registered[0]["name"] == "pwsh"
    assert "42" in await registered[0]["execute"](
        {"description": "add", "command": "20+22"}
    )
    assert PWSH_DESCRIPTION and PWSH_SECTION


@pytest.mark.asyncio
async def test_pwsh_spill_failure_and_prompt(monkeypatch: Any, tmp_path: Any) -> None:

    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    original = Path.write_bytes

    def fail_write(self: Path, data: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    result = await handle_pwsh(
        {"command": "'x' * 100000"},
        ctx=ctx,
        spill_dir=tmp_path,
        max_output_bytes=10,
    )
    assert "unavailable" in result
    monkeypatch.setattr(Path, "write_bytes", original)

    registered: list[Any] = []
    sections: list[Any] = []
    ctx2 = Context()
    ctx2.provide("tools", SimpleNamespace(register=registered.append))
    ctx2.provide(
        "systemPrompt",
        SimpleNamespace(
            section=sections.append,
            get_section_order=lambda name: 1000,
        ),
    )
    apply(ctx2)
    assert sections[0]["name"] == "tool:pwsh"


@pytest.mark.asyncio
async def test_pwsh_abort_and_config_keys(tmp_path: Any) -> None:
    from magitech_harness.agent.cancellation import AbortSignal

    ctx = Context()
    ctx.provide("pwshPath", PWSH_BIN)
    assert "42" in await handle_pwsh(
        {"description": "add", "command": "20+22"}, ctx=ctx
    )

    signal = AbortSignal()
    signal.abort()
    assert isinstance(await handle_pwsh({"command": "1", "signal": signal}), str)

    class LateSignal:
        def __init__(self) -> None:
            self.callback: Any = None

        def add_listener(self, callback: Any) -> Any:
            self.callback = callback
            return lambda: None

    late = LateSignal()
    result = await _abort_during_run(late)
    assert result


async def _abort_during_run(signal: Any) -> str:
    import asyncio

    task = asyncio.create_task(
        handle_pwsh({"command": "Start-Sleep -Seconds 2", "signal": signal})
    )
    await asyncio.sleep(0.05)
    if signal.callback is not None:
        signal.callback()
    return await task


@pytest.mark.asyncio
async def test_pwsh_spill_mkdir_failure(monkeypatch: Any, tmp_path: Any) -> None:
    from pathlib import Path as P

    original = P.mkdir

    def fail_mkdir(self: P, *args: Any, **kwargs: Any) -> None:
        raise OSError("no space")

    monkeypatch.setattr(P, "mkdir", fail_mkdir)
    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    result = await handle_pwsh(
        {"command": "'x' * 100000"},
        ctx=ctx,
        spill_dir=tmp_path,
        max_output_bytes=10,
    )
    assert "unavailable" in result
    monkeypatch.setattr(P, "mkdir", original)


@pytest.mark.asyncio
async def test_pwsh_sandbox_seam(tmp_path: Any) -> None:
    from magitech_harness.tools.pwsh_tool import handle_pwsh

    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    calls: list[Any] = []

    class Sandbox:
        def confine(self, argv: list[str], policy: Any, signal: Any = None) -> Any:
            calls.append(policy)
            from types import SimpleNamespace as NS

            return NS(argv=argv, enforcement="full")

    class Policy:
        def resolve(self, session: Any) -> dict[str, Any]:
            return {"mode": "read-only", "workspaceRoot": str(tmp_path)}

    ctx.provide("sandbox", Sandbox())
    ctx.provide("sandboxPolicy", Policy())
    result = await handle_pwsh(
        {"description": "add", "command": "20+22"},
        ctx=ctx,
        exec_view={"agent": SimpleNamespace(session=SimpleNamespace())},
    )
    assert "42" in result
    assert calls[0]["mode"] == "read-only"


@pytest.mark.asyncio
async def test_pwsh_config_key_and_full_access_policy(tmp_path: Any) -> None:
    from magitech_harness.tools.pwsh_tool import handle_pwsh

    ctx = Context()
    ctx.provide("pwshPath", PWSH_BIN)

    class Policy:
        def resolve(self, session: Any) -> dict[str, Any]:
            return {"mode": "danger-full-access", "workspaceRoot": str(tmp_path)}

    class Sandbox:
        def confine(self, *args: Any) -> Any:
            raise AssertionError("full access must not be confined")

    ctx.provide("sandbox", Sandbox())
    ctx.provide("sandboxPolicy", Policy())
    out = await handle_pwsh(
        {"description": "add", "command": "20+22"},
        ctx=ctx,
        exec_view={"agent": SimpleNamespace(session=SimpleNamespace())},
    )
    assert "42" in out


@pytest.mark.asyncio
async def test_pwsh_abort_and_kill_oserror(monkeypatch: Any) -> None:
    import asyncio

    from magitech_harness.tools.pwsh_tool import handle_pwsh

    class LateSignal:
        def __init__(self) -> None:
            self.callback: Any = None

        def add_listener(self, callback: Any) -> Any:
            self.callback = callback
            return lambda: None

    late = LateSignal()

    async def run_late() -> str:
        task = asyncio.create_task(
            handle_pwsh({"command": "Start-Sleep -Seconds 2", "signal": late})
        )
        await asyncio.sleep(0.05)
        if late.callback is not None:
            late.callback()
        return await task

    assert "aborted" in (await run_late()).lower()

    class KillFails:
        def kill(self) -> None:
            raise OSError("gone")

        async def wait(self) -> int:
            return -1

    real_create = asyncio.create_subprocess_exec

    async def fake_create(*args: Any, **kwargs: Any) -> Any:
        process = await real_create(*args, **kwargs)
        process.kill = KillFails().kill
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    out = await handle_pwsh({"command": "Start-Sleep -Seconds 2", "timeoutMs": 30})
    assert "timed out" in out


@pytest.mark.asyncio
async def test_pwsh_remaining_branches() -> None:
    from magitech_harness.tools.pwsh_tool import handle_pwsh

    # No configured key -> resolve from PATH.
    assert "42" in await handle_pwsh({"description": "add", "command": "20+22"})

    class DoubleSignal:
        def __init__(self) -> None:
            self.callback: Any = None

        def add_listener(self, callback: Any) -> Any:
            self.callback = callback
            return lambda: None

    signal = DoubleSignal()

    async def call_twice() -> str:
        task = asyncio.create_task(handle_pwsh({"command": "20+22", "signal": signal}))
        await asyncio.sleep(0.01)
        if signal.callback is not None:
            signal.callback()
            signal.callback()
        return await task

    result = await call_twice()
    assert "42" in result or "aborted" in result.lower()


@pytest.mark.asyncio
async def test_pwsh_bare_ctx_and_quiet_signal() -> None:
    from magitech_harness.tools.pwsh_tool import handle_pwsh

    out = await handle_pwsh({"description": "add", "command": "20+22"}, ctx=Context())
    assert "42" in out

    class QuietSignal:
        def add_listener(self, callback: Any) -> Any:
            return lambda: None

    out = await handle_pwsh(
        {"description": "add", "command": "20+22", "signal": QuietSignal()}
    )
    assert "42" in out


@pytest.mark.asyncio
async def test_pwsh_abort_kill_oserror_and_denial_marker(monkeypatch: Any) -> None:
    import asyncio

    from magitech_harness.tools.pwsh_tool import handle_pwsh

    class FailingKill:
        def kill(self) -> None:
            raise OSError("gone")

        async def wait(self) -> int:
            return -1

    real_create = asyncio.create_subprocess_exec

    async def fake_create(*args: Any, **kwargs: Any) -> Any:
        process = await real_create(*args, **kwargs)
        process.kill = FailingKill().kill
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    class Signal:
        def __init__(self) -> None:
            self.callback: Any = None

        def add_listener(self, callback: Any) -> Any:
            self.callback = callback
            return lambda: None

    signal = Signal()

    async def run() -> str:
        task = asyncio.create_task(
            handle_pwsh({"command": "Start-Sleep -Seconds 2", "signal": signal})
        )
        await asyncio.sleep(0.02)
        if signal.callback is not None:
            signal.callback()
        return await task

    assert "aborted" in (await run()).lower()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", real_create)

    class Policy:
        def resolve(self, session: Any) -> dict[str, Any]:
            return {"mode": "read-only", "workspaceRoot": "/tmp"}

    class Sandbox:
        def confine(self, argv: list[str], policy: Any, signal: Any = None) -> Any:
            from types import SimpleNamespace as NS

            return NS(
                argv=argv,
                denial_signatures=("read-only file system", "permission denied"),
            )

    ctx = Context()
    ctx.provide("sandbox", Sandbox())
    ctx.provide("sandboxPolicy", Policy())
    out = await handle_pwsh(
        {"command": "touch /etc/x; exit 1"},
        ctx=ctx,
        exec_view={"agent": SimpleNamespace(session=SimpleNamespace())},
    )
    assert "[sandbox: file access denied under read-only mode]" in out
    assert "[sandbox: escalation available" in out, (
        "denial must be followed by the same-turn escalation hint"
    )
    assert out.rstrip().endswith("[exit code: 1]")


@pytest.mark.asyncio
async def test_pwsh_denial_signatures_no_match() -> None:
    from magitech_harness.tools.pwsh_tool import handle_pwsh

    class Policy:
        def resolve(self, session: Any) -> dict[str, Any]:
            return {"mode": "read-only", "workspaceRoot": "/tmp"}

    class Sandbox:
        def confine(self, argv: list[str], policy: Any, signal: Any = None) -> Any:
            from types import SimpleNamespace as NS

            return NS(argv=argv, denial_signatures=("read-only file system",))

    ctx = Context()
    ctx.provide("sandbox", Sandbox())
    ctx.provide("sandboxPolicy", Policy())
    out = await handle_pwsh(
        {"command": "'unrelated failure' | Write-Error; exit 4"},
        ctx=ctx,
        exec_view={"agent": SimpleNamespace(session=SimpleNamespace())},
    )
    assert "[exit code: 4]" in out
    assert "[sandbox: file access denied" not in out


@pytest.mark.asyncio
async def test_pwsh_marker_order_single_timeout_and_cap(tmp_path: Any) -> None:
    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    result = await handle_pwsh(
        {"command": "Write-Output hi; exit 3"}, ctx=ctx, spill_dir=tmp_path
    )
    assert result.rstrip().endswith("[exit code: 3]")
    from magitech_harness.agent.cancellation import AbortSignal

    with_signal = await handle_pwsh(
        {"command": "exit 3", "signal": AbortSignal()}, ctx=ctx, spill_dir=tmp_path
    )
    assert with_signal.rstrip().endswith("[exit code: 3]")

    timed = await handle_pwsh(
        {"command": "Start-Sleep -Seconds 2", "timeoutMs": 50},
        ctx=ctx,
        spill_dir=tmp_path,
    )
    assert timed == "[timed out after 50ms]"

    registered: list[Any] = []
    capped = Context()
    capped.provide("tools", SimpleNamespace(register=registered.append))
    capped.provide("pwshConfig", PWSH_BIN)
    capped.provide("shell", SimpleNamespace(config={"maxTimeoutMs": 30}))
    apply(capped)
    out = await registered[0]["execute"](
        {
            "description": "sleep",
            "command": "Start-Sleep -Seconds 2",
            "timeoutMs": 1000,
        }
    )
    assert out == "[timed out after 30ms]"


@pytest.mark.asyncio
async def test_pwsh_signal_marker(monkeypatch: Any) -> None:
    class Signalled:
        def __init__(self, code: int) -> None:
            self.returncode = code

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def spawn(code: int) -> Any:
        async def create(*_args: Any, **_kwargs: Any) -> Any:
            return Signalled(code)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        return await handle_pwsh({"command": "1"})

    assert "[killed by signal: SIGTERM]" in await spawn(-15)
    assert "[killed by signal: 999999]" in await spawn(-999999)
