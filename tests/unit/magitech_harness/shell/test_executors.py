"""Unit tests for ShellExecutor, LocalBashExecutor, and LocalPwshExecutor."""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from anima import Context

from magitech_harness.agent.cancellation import MAX_TIMER_DELAY_MS, AbortController
from magitech_harness.shell import (
    PWSH_ENCODING_PREAMBLE as ENCODING_PREAMBLE,
)
from magitech_harness.shell import (
    LocalBashExecutor,
    LocalPwshExecutor,
    ShellExecutor,
    assert_serviceable_bash_config,
    assert_serviceable_pwsh_config,
    candidate_pwsh_paths,
    resolve_pwsh_path,
)
from magitech_harness.shell.bash_local import ENV_OVERRIDES as BASH_ENV
from magitech_harness.shell.pwsh_local import ENV_OVERRIDES as PWSH_ENV
from magitech_harness.shell.pwsh_local import _candidate_exists
from magitech_harness.subprocess import LocalSubprocessRuntime


async def test_config_validation_and_executor_resolve(tmp_path: Path) -> None:
    valid = {
        "timeoutMs": 1000,
        "maxTimeoutMs": 5000,
        "maxOutputBytes": 1024,
        "maxSpillBytes": 2048,
        "graceMs": 500,
    }
    assert_serviceable_bash_config(valid)
    assert_serviceable_pwsh_config(valid)

    for bad_val in (0, -1, True, "5", math.nan, math.inf):
        with pytest.raises(ValueError, match="positive finite number"):
            assert_serviceable_bash_config({**valid, "timeoutMs": bad_val})
    with pytest.raises(ValueError, match="no greater than"):
        assert_serviceable_bash_config({**valid, "graceMs": MAX_TIMER_DELAY_MS + 1})

    root = Context()
    f1 = root.use(LocalSubprocessRuntime)
    f2 = root.use(
        LocalBashExecutor,
        {"cwd": str(tmp_path), "timeoutMs": 1000, "maxTimeoutMs": 2000},
    )
    await f1
    await f2

    exe: LocalBashExecutor = root.shell
    assert exe.sandbox_mode is None
    assert isinstance(exe, ShellExecutor)

    sig = AbortController().signal
    spec = exe.resolve(
        {
            "command": "echo hi",
            "timeoutMs": 9999,
            "stdin": "payload",
            "env": {"FOO": "bar"},
            "magitechEnv": {"MAGITECH_TEST": "1"},
            "signal": sig,
            "sandboxPolicy": {"mode": "ro"},
        }
    )
    assert spec["workdir"] == str(tmp_path)
    assert spec["timeoutMs"] == 2000
    assert spec["stdin"] == "payload"
    assert spec["env"] == {"FOO": "bar"}
    assert spec["magitechEnv"] == {"MAGITECH_TEST": "1"}
    assert spec["signal"] is sig
    assert spec["sandboxPolicy"] == {"mode": "ro"}

    with pytest.raises(ValueError, match="positive finite number"):
        exe.resolve({"command": "echo hi", "stdoutMaxBytes": 0})

    await f2.dispose()
    await f1.dispose()


async def test_bash_local_live_execution_streaming_kill_timeout_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAGITECH_LEAKED_SECRET", "top-secret")
    root = Context()
    f1 = root.use(LocalSubprocessRuntime)
    is_win = sys.platform == "win32"
    executor_cls = LocalPwshExecutor if is_win else LocalBashExecutor
    f2 = root.use(
        executor_cls,
        {
            "cwd": str(tmp_path),
            "timeoutMs": 5000,
            "maxOutputBytes": 200,
            "maxSpillBytes": 1000,
            "graceMs": 100,
        },
    )
    await f1
    await f2
    exe: ShellExecutor = root.shell

    # 1. Foreground with stdin, env precedence, magitechEnv beating env, and ShellExecutor.run()
    scrub_cmd = (
        "[Console]::Out.Write(\"$env:NO_COLOR|$env:CUSTOM_E|$(if ($env:MAGITECH_LEAKED_SECRET) { $env:MAGITECH_LEAKED_SECRET } else { 'scrubbed' })\")"
        if is_win
        else 'printf \'%s|%s|%s\' "$NO_COLOR" "$CUSTOM_E" "${MAGITECH_LEAKED_SECRET:-scrubbed}"'
    )
    run_res = await exe.run(
        scrub_cmd,
        cwd=tmp_path,
        timeout_seconds=10.0 if is_win else 2.0,
        max_output_bytes=1024,
    )
    assert run_res.exit_code == 0
    assert run_res.stdout == "1||scrubbed"
    assert run_res.timed_out is False
    assert run_res.aborted is False

    # 2. Spec with stdin, env, magitechEnv, missing trailing newline + stderr, and on_started + sandbox memoization
    stdin_cmd = (
        "$line = [Console]::In.ReadLine(); "
        '[Console]::Out.Write("out:${line}:${env:CUSTOM}:${env:MAGITECH_OWN}"); '
        '[Console]::Error.Write("errline")'
        if is_win
        else 'read -r line; printf \'out:%s:%s:%s\' "$line" "$CUSTOM" "$MAGITECH_OWN"; printf \'errline\' >&2'
    )
    spec = exe.resolve(
        {
            "command": stdin_cmd,
            "stdin": "inval",
            "env": {"CUSTOM": "user", "MAGITECH_OWN": "spoof"},
            "magitechEnv": {"MAGITECH_OWN": "trusted"},
        }
    )
    started: list[Any] = []
    if is_win:
        assert isinstance(exe, LocalPwshExecutor)
        proc = await exe.execute_argv(spec, exe.argv(spec), on_started=started.append)
    else:
        assert isinstance(exe, LocalBashExecutor)
        proc = await exe.execute_argv(
            spec, ["bash", "-c", spec["command"]], on_started=started.append
        )
    assert started == [proc]
    proc.sandbox = {"mode": "read-only", "denied": False}
    fg = await proc.result()
    # Second result() call returns memoized task
    assert (await proc.result()) is fg
    assert fg["exitCode"] == 0
    assert fg["stdout"]["text"] == "out:inval:user:trusted"
    assert fg["stderr"]["text"] == "errline"
    assert fg["sandbox"] == {"mode": "read-only", "denied": False}
    assert proc.observed["stderr"].read_from(0)["text"] == "errline"
    read1 = proc.read_output()
    assert "out:inval:user:trusted\n[stderr]\nerrline" == read1["delta"]
    assert proc.kill() is False

    # 3. Spill paths & lossy read_output + observed streams + kill live process
    py_cmd = (
        f"& '{sys.executable}' -c 'import sys; sys.stdout.write(''a''*500); sys.stderr.write(''b''*500)'"
        if is_win
        else 'python3 -c \'import sys; sys.stdout.write("a"*500); sys.stderr.write("b"*500)\''
    )
    big_spec = exe.resolve({"command": py_cmd, "onExpiry": "none"})
    big_proc = await exe.execute(big_spec)
    await big_proc.done
    big_out = big_proc.read_output()
    assert big_out["lossy"] is True
    assert "stdoutSpillPath" in big_out
    assert "stderrSpillPath" in big_out
    assert "spillPath" in (await big_proc.result())["stdout"]

    # 4. Timeout kill classification via onExpiry="kill"
    sleep_cmd = "Start-Sleep -Seconds 10" if is_win else "sleep 5"
    to_spec = exe.resolve(
        {"command": sleep_cmd, "timeoutMs": 300 if is_win else 60, "onExpiry": "kill"}
    )
    to_proc = await exe.execute(to_spec)
    to_res = await to_proc.result()
    assert to_res["timedOut"] is True
    assert to_res["aborted"] is False
    assert to_proc.status == "killed"

    # 5. Manual kill returns True once then False; ShellExecutor.run exitCode=None -> -1
    kill_spec = exe.resolve({"command": sleep_cmd, "onExpiry": "none"})
    kill_proc = await exe.execute(kill_spec)
    assert kill_proc.kill() is True
    assert kill_proc.kill() is False
    await kill_proc.done
    assert kill_proc.status == "killed"

    sig = AbortController()
    asyncio.get_running_loop().call_later(0.05, sig.abort, "stop")
    aborted_run = await exe.run(sleep_cmd, signal=sig.signal)
    assert aborted_run.aborted is True
    assert aborted_run.exit_code == -1

    await f2.dispose()
    await f1.dispose()


async def test_execute_argv_prep_and_provider_failures(tmp_path: Path) -> None:
    root = Context()
    fake_sub = MagicMock()
    root.provide("subprocess", fake_sub)
    exe = LocalBashExecutor(root, {"cwd": str(tmp_path)})

    # 1. Async preparation succeeding
    ok_proc_done = asyncio.get_running_loop().create_future()
    ok_proc_done.set_result({"exitCode": 0, "signal": None})
    reader = MagicMock()
    reader.read_from.return_value = {"text": "", "nextOffset": 0, "lossy": False}
    fake_handle = SimpleNamespace(
        done=ok_proc_done,
        collected={"stdout": reader, "stderr": reader},
        terminate=MagicMock(),
    )
    fake_sub.spawn.return_value = fake_handle

    async def _prep_ok(_sig: Any) -> list[str]:
        return ["echo", "ok"]

    spec_none = exe.resolve({"command": "echo ok", "onExpiry": "none"})
    p_bash = await exe.execute(spec_none)
    await p_bash.done
    assert p_bash.status == "completed"
    p_ok = await exe.execute_argv(spec_none, _prep_ok)
    await p_ok.done
    assert p_ok.status == "completed"

    # 2. Preparation aborted during prepare -> raises without spawning
    ctrl = AbortController()

    async def _prep_slow(_sig: Any) -> list[str]:
        await asyncio.sleep(1)
        return ["echo", "never"]

    ctrl.abort(RuntimeError("user cancelled"))
    with pytest.raises(RuntimeError, match="user cancelled"):
        await exe.execute_argv(
            exe.resolve({"command": "x", "signal": ctrl.signal}), _prep_slow
        )

    # Pre-aborted with non-BaseException reason & prep finishing synchronously before checking aborted
    ctrl2 = AbortController()
    ctrl2.abort("plain-string-reason")

    async def _prep_fast(_sig: Any) -> list[str]:
        return ["echo"]

    with pytest.raises(RuntimeError, match="aborted"):
        await exe.execute_argv(
            exe.resolve({"command": "x", "onExpiry": "none", "signal": ctrl2.signal}),
            _prep_fast,
        )

    # Mid-flight abort with non-exception reason cancels pending task cleanly
    ctrl3 = AbortController()
    asyncio.get_running_loop().call_later(0.02, ctrl3.abort, "mid")
    with pytest.raises(RuntimeError, match="aborted"):
        await exe.execute_argv(
            exe.resolve({"command": "x", "onExpiry": "none", "signal": ctrl3.signal}),
            _prep_slow,
        )

    # 3. Preparation timeout with onExpiry="kill" returns settled killed handle with no spawn
    spec_to = exe.resolve({"command": "x", "timeoutMs": 30, "onExpiry": "kill"})
    p_prep_to = await exe.execute_argv(spec_to, _prep_slow)
    res_to = await p_prep_to.result()
    assert res_to["timedOut"] is True
    assert p_prep_to.status == "killed"
    assert res_to["stdout"] == {"text": "", "truncated": False}

    # 4. Sync spawn throw -> kill() before done sees running is None; settled killed + observed/read_output failure note
    fake_sub.spawn.side_effect = RuntimeError("enoent")
    p_sync_err = await exe.execute(spec_none)
    assert p_sync_err.kill() is True
    await p_sync_err.done
    assert p_sync_err.status == "killed"
    obs1 = p_sync_err.observed["stderr"].read_from(0)
    assert "subprocess failed before reporting an outcome: enoent" in obs1["text"]
    obs_partial = p_sync_err.observed["stderr"].read_from(11)
    assert obs_partial["text"].startswith("failed")
    assert "enoent" in p_sync_err.read_output()["delta"]
    assert p_sync_err.read_output()["delta"] == ""
    assert p_sync_err.kill() is False
    with pytest.raises(RuntimeError, match="enoent"):
        await p_sync_err.result()

    # 5. Async running.done rejection WITHOUT kill -> provider failure
    bad_done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    bad_done.set_exception(OSError("spawn crashed"))
    fake_sub.spawn.side_effect = None
    fake_sub.spawn.return_value = SimpleNamespace(
        done=bad_done,
        collected={"stdout": reader, "stderr": reader},
        terminate=MagicMock(),
    )
    p_async_err = await exe.execute(spec_none)
    await p_async_err.done
    with pytest.raises(OSError, match="spawn crashed"):
        await p_async_err.result()

    # 6. Async running.done rejection AFTER proc.kill() -> NOT provider failure!
    kill_done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    fake_sub.spawn.return_value = SimpleNamespace(
        done=kill_done,
        collected={"stdout": reader, "stderr": reader},
        terminate=MagicMock(),
    )
    p_killed_rej = await exe.execute(spec_none)
    assert p_killed_rej.kill() is True
    kill_done.set_exception(RuntimeError("terminated early"))
    res_killed = await p_killed_rej.result()
    assert res_killed["timedOut"] is False


async def test_pwsh_path_resolution_and_executor(tmp_path: Path) -> None:
    paths = candidate_pwsh_paths(
        {
            "ProgramFiles": r"D:\Apps",
            "SystemRoot": r"D:\Win",
            "PATH": ' ; "C:\\Tools\\pwsh" ; D:\\Extra ',
        }
    )
    assert paths == [
        r"D:\Apps\PowerShell\7\pwsh.exe",
        r"C:\Tools\pwsh\pwsh.exe",
        r"D:\Extra\pwsh.exe",
        r"D:\Win\System32\WindowsPowerShell\v1.0\powershell.exe",
    ]
    assert resolve_pwsh_path("/custom/pwsh") == "/custom/pwsh"
    assert resolve_pwsh_path(None, {}, "linux") == "pwsh"
    assert (
        resolve_pwsh_path(
            None,
            {"ProgramFiles": "/nonexistent", "SystemRoot": "/nonexistent"},
            "win32",
        )
        == "pwsh"
    )

    # Match existing file via _candidate_exists & simulated win32 candidate probe
    from unittest.mock import patch

    tool_dir = tmp_path / "pwshbin"
    tool_dir.mkdir()
    exe_file = tool_dir / "pwsh.exe"
    exe_file.write_text("", encoding="utf-8")
    assert _candidate_exists(str(exe_file)) is True
    assert _candidate_exists(str(tool_dir)) is False
    assert _candidate_exists(str(tmp_path / "missing.exe")) is False
    expected_probe = f"{tool_dir}\\pwsh.exe"
    with patch(
        "magitech_harness.shell.pwsh_local._candidate_exists",
        side_effect=lambda p: p == expected_probe,
    ):
        assert (
            resolve_pwsh_path(
                None,
                {
                    "ProgramFiles": str(tmp_path / "nope"),
                    "SystemRoot": str(tmp_path / "nope2"),
                    "PATH": str(tool_dir),
                },
                "win32",
            )
            == expected_probe
        )

    root = Context()
    fake_sub = MagicMock()
    reader = MagicMock()
    reader.read_from.return_value = {"text": "", "nextOffset": 0, "lossy": False}
    ok_done = asyncio.get_running_loop().create_future()
    ok_done.set_result({"exitCode": 0, "signal": None})
    fake_sub.spawn.return_value = SimpleNamespace(
        done=ok_done,
        collected={"stdout": reader, "stderr": reader},
        terminate=MagicMock(),
    )
    root.provide("subprocess", fake_sub)

    pwsh = LocalPwshExecutor(root, {"pwshPath": "/opt/pwsh"})
    assert "TERM" not in PWSH_ENV
    assert "TERM" in BASH_ENV
    assert pwsh.pwsh_path == "/opt/pwsh"
    pwsh.config["pwshPath"] = "/opt/pwsh2"
    assert pwsh.pwsh_path == "/opt/pwsh2"
    assert pwsh.pwsh_path == "/opt/pwsh2"

    spec = pwsh.resolve({"command": "Write-Output 1"})
    proc = await pwsh.execute(spec)
    await proc.done
    spawned_spec = fake_sub.spawn.call_args.args[0]
    assert spawned_spec["argv"] == [
        "/opt/pwsh2",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        f"{ENCODING_PREAMBLE}Write-Output 1",
    ]
