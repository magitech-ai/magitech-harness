"""Unit tests for SubprocessRuntime, env scrubbing, and LocalSubprocessRuntime."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest
from anima import Context

from magitech_harness.agent.cancellation import AbortController
from magitech_harness.subprocess.local import (
    LocalSubprocessRuntime,
    signal_tree,
    taskkill_process_tree,
    validate_subprocess_spec,
)
from magitech_harness.subprocess.service import (
    SubprocessExecutableNotFoundError,
    SubprocessRuntime,
    child_env,
    scrubbed_parent_env,
)


def test_env_scrubbing_and_spec_validation() -> None:
    """Scrubs harness/provider env vars by default and validates spawn specs."""
    base = {
        "PATH": "/usr/bin",
        "MAGITECH_SECRET": "x",
        "MTH_KEY": "y",
        "OPENAI_API_KEY": "z",
        "USER": "alice",
    }
    scrubbed = scrubbed_parent_env(base)
    assert scrubbed == {"PATH": "/usr/bin", "USER": "alice"}
    assert child_env({"OPENAI_API_KEY": "override"}, base) == {
        "PATH": "/usr/bin",
        "USER": "alice",
        "OPENAI_API_KEY": "override",
    }

    with pytest.raises(ValueError, match="positive finite"):
        validate_subprocess_spec(
            {
                "argv": ["echo"],
                "cwd": ".",
                "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
                "graceMs": 0,
            }
        )

    aborted = AbortController()
    aborted.abort("stop-now")
    with pytest.raises(RuntimeError, match="aborted before spawn: stop-now"):
        validate_subprocess_spec(
            {
                "argv": ["echo"],
                "cwd": ".",
                "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
                "graceMs": 500,
                "signal": aborted.signal,
            }
        )

    with pytest.raises(ValueError, match="non-empty program"):
        validate_subprocess_spec(
            {
                "argv": [""],
                "cwd": ".",
                "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
                "graceMs": 500,
            }
        )


@pytest.mark.asyncio
async def test_resolve_executable_and_abstract_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolves absolute/PATH binaries, checks win32 PATHEXT, and rejects invalid inputs."""
    ctx = Context()
    ctx.use(LocalSubprocessRuntime, {"spillDir": str(tmp_path / "spills")})
    await ctx.flush()
    rt: SubprocessRuntime = ctx.subprocess

    exe = await rt.resolveExecutable(sys.executable)
    assert exe == sys.executable

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    script = fake_bin / "mytool"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    non_exec = fake_bin / "noexec"
    non_exec.write_text("hi", encoding="utf-8")
    non_exec.chmod(0o644)
    (fake_bin / "subdir").mkdir()

    env_path = f"{fake_bin}{os.pathsep}{os.pathsep}"
    assert await rt.resolve_executable("mytool", {"PATH": env_path}) == str(script)

    with pytest.raises(ValueError, match="must be non-empty"):
        await rt.resolve_executable("")
    with pytest.raises(ValueError, match="relative path"):
        await rt.resolve_executable("./mytool")
    with pytest.raises(SubprocessExecutableNotFoundError):
        await rt.resolve_executable("subdir", {"PATH": env_path})
    with pytest.raises(SubprocessExecutableNotFoundError):
        await rt.resolve_executable("noexec", {"PATH": env_path})
    with pytest.raises(SubprocessExecutableNotFoundError):
        await rt.resolve_executable(str(tmp_path / "missing"))
    with pytest.raises(SubprocessExecutableNotFoundError):
        await rt.resolve_executable(
            "never_found", {"PATH": env_path}, AbortController().signal
        )

    # Case-insensitive env lookup & Win32 PATHEXT resolution
    import magitech_harness.subprocess.local as local_mod

    monkeypatch.delenv("PATH", raising=False)
    win_exe = fake_bin / "winapp.EXE"
    win_exe.write_text("x", encoding="utf-8")
    monkeypatch.setattr(local_mod.sys, "platform", "win32")
    assert await rt.resolve_executable(
        "winapp", {"Path": str(fake_bin), "PATHEXT": ".EXE"}
    ) == str(win_exe)
    assert await rt.resolve_executable("winapp.EXE", {"Path": str(fake_bin)}) == str(
        win_exe
    )
    assert local_mod._env_lookup_ci({"OTHER": "1"}, "PATH") is None
    monkeypatch.undo()

    # Path error OSError skipped & pre/mid-abort raised
    ctrl = AbortController()
    ctrl.abort("canceled")
    with pytest.raises(Exception, match="canceled"):
        await rt.resolve_executable("mytool", {"PATH": env_path}, ctrl.signal)

    monkeypatch.setattr(
        local_mod.Path,
        "is_file",
        lambda _s: (_ for _ in ()).throw(OSError("stat fail")),
    )
    with pytest.raises(SubprocessExecutableNotFoundError):
        await rt.resolve_executable("mytool", {"PATH": env_path})
    ctrl2 = AbortController()
    orig_throw = ctrl2.signal.throw_if_aborted
    calls = 0

    def _abort_after_first() -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            ctrl2.abort("mid-resolve")
        orig_throw()

    monkeypatch.setattr(ctrl2.signal, "throw_if_aborted", _abort_after_first)
    with pytest.raises(Exception, match="mid-resolve"):
        await rt.resolve_executable("mytool", {"PATH": env_path}, ctrl2.signal)
    monkeypatch.undo()

    # Dispose a context with no live subprocesses (targets empty branch)
    empty_ctx = Context()
    empty_ctx.use(LocalSubprocessRuntime)
    await empty_ctx.flush()
    await empty_ctx.fiber.dispose()

    # Check post-loop abort right before SubprocessExecutableNotFoundError
    ctrl3 = AbortController()
    orig_t3 = ctrl3.signal.throw_if_aborted
    calls3 = 0

    def _abort_after_scan() -> None:
        nonlocal calls3
        calls3 += 1
        if calls3 >= 2:
            ctrl3.abort("end-abort")
        orig_t3()

    monkeypatch.setattr(ctrl3.signal, "throw_if_aborted", _abort_after_scan)
    with pytest.raises(Exception, match="end-abort"):
        await rt.resolve_executable(str(tmp_path / "absent"), None, ctrl3.signal)
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_local_subprocess_spawn_collect_pipe_terminate_and_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawns commands with bytes stdin, collect+spill, pipes, escalation, and dispose."""
    ctx = Context()
    ctx.use(LocalSubprocessRuntime, {"spillDir": str(tmp_path / "spills")})
    await ctx.flush()
    rt: LocalSubprocessRuntime = ctx.subprocess

    # Batch bytes stdin -> collect stdout/stderr with spill
    h1 = rt.spawn(
        {
            "argv": [
                sys.executable,
                "-c",
                "import sys; d = sys.stdin.read(); sys.stdout.write(d.upper()); sys.stderr.write('err');",
            ],
            "cwd": str(tmp_path),
            "stdio": {
                "stdin": {"kind": "bytes", "data": b"hello"},
                "stdout": {
                    "kind": "collect",
                    "maxBytes": 3,
                    "spill": {"maxBytes": 100},
                },
                "stderr": {"kind": "collect", "maxBytes": 16},
            },
            "graceMs": 500,
            "env": {"CUSTOM_VAR": "1"},
        }
    )
    assert await h1.wait_for_exit() is True
    assert await h1.wait_for_exit() is True  # already exited fast path
    outcome1 = await h1.done
    h1._force_kill()  # safety no-op once process exited

    # Live wait_for_exit with non-aborting signal completes normally when child exits
    h_sig_ok = rt.spawn(
        {
            "argv": [sys.executable, "-c", "import time; time.sleep(0.03)"],
            "cwd": str(tmp_path),
            "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
            "graceMs": 200,
        }
    )
    assert await h_sig_ok.wait_for_exit(AbortController().signal) is True
    await h_sig_ok.done
    assert outcome1 == {"exitCode": 0, "signal": None}
    out1 = h1.collected["stdout"].finalize()
    assert out1["text"] == "LLO"
    assert out1["truncated"] is True
    assert "spillPath" in out1
    assert h1.collected["stderr"].finalize()["text"] == "err"
    assert h1.stdin is None and h1.stdout is None and h1.stderr is None

    # Pipe mode + inherit stdout/stderr fast check
    h_pipe = rt.spawn(
        {
            "argv": [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write(sys.stdin.readline().strip()); sys.stderr.write('e');",
            ],
            "cwd": str(tmp_path),
            "stdio": {"stdin": "pipe", "stdout": "pipe", "stderr": "pipe"},
            "graceMs": 500,
        }
    )
    assert h_pipe.pid is not None
    assert h_pipe.stdin is not None
    assert h_pipe.stdout is not None
    assert h_pipe.stderr is not None
    h_pipe.stdin.write(b"ping\n")
    h_pipe.stdin.close()
    assert await h_pipe.done == {"exitCode": 0, "signal": None}
    assert await h_pipe.stdout.read() == b"ping"
    assert await h_pipe.stderr.read() == b"e"

    # Nonexistent executable rejects done future cleanly
    h_bad = rt.spawn(
        {
            "argv": [str(tmp_path / "no-such-bin")],
            "cwd": str(tmp_path),
            "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
            "graceMs": 100,
        }
    )
    h_bad.terminate()  # exercises pre-spawn terminate flag before launch fails
    with pytest.raises(OSError):
        await h_bad.done

    # Wait until child signals SIGTERM handler is installed, then abort
    ready_file = tmp_path / "sig_ready.flag"
    abort_sig = AbortController()
    h_sleep = rt.spawn(
        {
            "argv": [
                sys.executable,
                "-c",
                (
                    "import pathlib, signal, time; "
                    "signal.signal(signal.SIGTERM, lambda *_: None); "
                    f"pathlib.Path({str(ready_file)!r}).write_text('ok'); "
                    "time.sleep(5)"
                ),
            ],
            "cwd": str(tmp_path),
            "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
            "graceMs": 80,
            "signal": abort_sig.signal,
        }
    )
    while not ready_file.exists():
        await asyncio.sleep(0.005)
    # Test wait_for_exit abort before exit returns False
    wait_abort = AbortController()
    wait_task = asyncio.create_task(h_sleep.wait_for_exit(wait_abort.signal))
    await asyncio.sleep(0.01)
    wait_abort.abort(RuntimeError("wait stopped"))
    assert await wait_task is False
    assert await h_sleep.wait_for_exit(wait_abort.signal) is False  # already aborted

    pre_aborted_wait = AbortController()
    pre_aborted_wait.abort("plain-stop")
    assert await h_sleep.wait_for_exit(pre_aborted_wait.signal) is False

    abort_sig.abort("kill-now")
    sleep_outcome = await h_sleep.done
    assert sleep_outcome["exitCode"] is None
    assert sleep_outcome["signal"] == "SIGKILL"
    await h_sleep.wait_for_exit()
    h_sleep.terminate()  # idempotent after exit
    h_sleep.terminate_for_host_exit()

    # Early terminate() on valid command right after spawn() before create_subprocess_exec completes
    h_early = rt.spawn(
        {
            "argv": [sys.executable, "-c", "import time; time.sleep(5)"],
            "cwd": str(tmp_path),
            "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
            "graceMs": 100,
        }
    )
    h_early.terminate()
    await h_early.done

    # Detached grandchild holds stdout pipe past parent exit -> pump timeout cancels pending pump
    h_drain = rt.spawn(
        {
            "argv": [
                sys.executable,
                "-c",
                "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1)'])",
            ],
            "cwd": str(tmp_path),
            "stdio": {
                "stdin": "ignore",
                "stdout": {"kind": "collect", "maxBytes": 64},
                "stderr": "inherit",
            },
            "graceMs": 50,
        }
    )
    await h_drain.done
    signal_tree(h_drain.pid, signal.SIGKILL)

    # Stdin write OSError (simulated EPIPE on closed writer) + win32 creationflags branch
    import magitech_harness.subprocess.local as local_mod

    orig_start_popen = local_mod._start_popen

    def _mock_win32_popen(argv: Any, kwargs: Any) -> Any:
        assert "creationflags" in kwargs
        del kwargs["creationflags"]
        kwargs["start_new_session"] = True
        return orig_start_popen(argv, kwargs)

    monkeypatch.setattr(local_mod.sys, "platform", "win32")
    monkeypatch.setattr(local_mod, "_start_popen", _mock_win32_popen)
    h_win = rt.spawn(
        {
            "argv": [
                sys.executable,
                "-c",
                "import sys; print('win-out'); sys.stderr.write('win-err')",
            ],
            "cwd": str(tmp_path),
            "stdio": {
                "stdin": {"kind": "bytes", "data": b"data"},
                "stdout": {"kind": "collect", "maxBytes": 64},
                "stderr": {"kind": "collect", "maxBytes": 64},
            },
            "graceMs": 100,
        }
    )
    assert h_win.pid is not None
    await h_win.done
    assert "win-out" in h_win.collected["stdout"].finalize()["text"]
    assert h_win.collected["stderr"].finalize()["text"] == "win-err"
    monkeypatch.undo()

    # Patch stdin.write to raise OSError during launch via mock Popen
    class _StubProc:
        pid = 999999
        returncode = 0

        class _BadStdin:
            def write(self, _d: bytes) -> int:
                raise BrokenPipeError("epipe")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                raise OSError("close fail")

        stdin = _BadStdin()
        stdout = None
        stderr = None

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(local_mod, "_start_popen", lambda _argv, _kw: _StubProc())
    h_epipe = rt.spawn(
        {
            "argv": [sys.executable],
            "cwd": str(tmp_path),
            "stdio": {
                "stdin": {"kind": "bytes", "data": b"hi"},
                "stdout": "inherit",
                "stderr": "inherit",
            },
            "graceMs": 100,
        }
    )
    assert await h_epipe.done == {"exitCode": 0, "signal": None}
    monkeypatch.undo()

    # Dispose live processes via context teardown + terminate_for_host_exit
    h_live1 = rt.spawn(
        {
            "argv": [sys.executable, "-c", "import time; time.sleep(5)"],
            "cwd": str(tmp_path),
            "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
            "graceMs": 200,
        }
    )
    assert h_live1.pid is not None
    h_live1.terminate_for_host_exit()
    await h_live1.done

    h_live2 = rt.spawn(
        {
            "argv": [sys.executable, "-c", "import time; time.sleep(5)"],
            "cwd": str(tmp_path),
            "stdio": {"stdin": "ignore", "stdout": "inherit", "stderr": "inherit"},
            "graceMs": 200,
        }
    )
    assert h_live2.pid is not None
    await ctx.fiber.dispose()
    res2 = await h_live2.done
    assert res2["signal"] == "SIGTERM"


def test_signal_helpers_and_platform_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers signal_tree, taskkill_process_tree, _decode_returncode, and win32 guards."""
    import magitech_harness.subprocess.local as local_mod

    assert local_mod._decode_returncode(None) == {"exitCode": None, "signal": None}
    assert local_mod._decode_returncode(-99999) == {
        "exitCode": None,
        "signal": "SIG99999",
    }

    signal_tree(None, signal.SIGTERM)
    taskkill_process_tree(None)

    called: list[Any] = []
    monkeypatch.setattr(
        local_mod.py_subprocess,
        "run",
        lambda args, **_kw: called.append(args),
    )
    taskkill_process_tree(1234)
    assert called == [["taskkill", "/PID", "1234", "/T", "/F"]]

    monkeypatch.setattr(
        local_mod.py_subprocess,
        "run",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("no taskkill")),
    )
    taskkill_process_tree(1234)  # swallows OSError

    monkeypatch.setattr(local_mod.sys, "platform", "win32")
    signal_tree(1234, signal.SIGTERM)
    monkeypatch.undo()

    # POSIX fallback from os.killpg failure -> os.kill -> final OSError swallow
    monkeypatch.setattr(
        local_mod.os,
        "killpg",
        lambda *_a: (_ for _ in ()).throw(OSError("no group")),
    )
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        local_mod.os,
        "kill",
        lambda pid, sig: kill_calls.append((pid, sig)),
    )
    signal_tree(42, signal.SIGTERM)
    assert kill_calls == [(42, signal.SIGTERM)]

    monkeypatch.setattr(
        local_mod.os,
        "kill",
        lambda *_a: (_ for _ in ()).throw(ProcessLookupError("gone")),
    )
    signal_tree(42, signal.SIGTERM)
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_env_scrubbing_pattern_win32_stream_pumps_and_pgrp_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers SENSITIVE_ENV_PATTERN, win32 child_env, _PipeStreamWriter, _StreamPump, and pgrp."""
    import magitech_harness.subprocess.local as local_mod
    import magitech_harness.subprocess.service as service_mod
    from magitech_harness.subprocess import child_env, scrubbed_parent_env

    # SENSITIVE_ENV_PATTERN + MAGITECH_* scrubbing, while preserving non-secret vars
    scrubbed = scrubbed_parent_env(
        {
            "MY_CUSTOM_KEY": "secret",
            "DB_PASSWORD": "pw",
            "AUTH_SECRET": "sec",
            "github_token": "tok",
            "MAGITECH_HOME": "/tmp/mth",
            "magitech_debug": "1",
            "OPENAI_BASE_URL": "https://example.invalid/v1",
            "PATH": "/bin",
        }
    )
    assert "OPENAI_BASE_URL" in scrubbed
    assert "PATH" in scrubbed
    for dropped in (
        "MY_CUSTOM_KEY",
        "DB_PASSWORD",
        "AUTH_SECRET",
        "github_token",
        "MAGITECH_HOME",
        "magitech_debug",
    ):
        assert dropped not in scrubbed

    monkeypatch.setattr(service_mod.sys, "platform", "win32")
    win_env = child_env(
        overrides={"path": "C:\\bin"},
        base_env={"Path": "C:\\old", "KEEP": "1"},
    )
    assert win_env == {"KEEP": "1", "path": "C:\\bin"}
    monkeypatch.undo()

    # _PipeStreamWriter full contract + OSError paths
    class _MemPipe:
        def __init__(self, fail_flush: bool = False, fail_close: bool = False) -> None:
            self.chunks: list[bytes] = []
            self.fail_flush = fail_flush
            self.fail_close = fail_close

        def write(self, data: bytes) -> int:
            self.chunks.append(data)
            return len(data)

        def writelines(self, lines: Any) -> None:
            self.chunks.extend(lines)

        def flush(self) -> None:
            if self.fail_flush:
                raise OSError("flush fail")

        def close(self) -> None:
            if self.fail_close:
                raise OSError("close fail")

    p_ok = _MemPipe()
    writer = local_mod._PipeStreamWriter(p_ok)  # type: ignore[arg-type]
    assert writer.can_write_eof() is True
    assert writer.is_closing() is False
    writer.writelines([b"a", b"b"])
    await writer.drain()
    writer.write_eof()
    assert writer.is_closing() is True
    writer.close()  # idempotent second close
    await writer.wait_closed()
    assert p_ok.chunks == [b"a", b"b"]

    writer_err = local_mod._PipeStreamWriter(
        _MemPipe(fail_flush=True, fail_close=True)  # type: ignore[arg-type]
    )
    writer_err.close()

    # _close_pipe error swallow
    local_mod._close_pipe(_MemPipe(fail_close=True))  # type: ignore[arg-type]

    # _StreamPump edge cases: BlockingIOError, OSError in _on_readable, win32 threaded error
    loop = asyncio.get_running_loop()
    r_fd, w_fd = os.pipe()
    os.write(w_fd, b"x")
    r_file = os.fdopen(r_fd, "rb", buffering=0)
    seen: list[bytes] = []
    eof_count = 0

    def _on_eof() -> None:
        nonlocal eof_count
        eof_count += 1

    pump = local_mod._StreamPump(loop, r_file, seen.append, _on_eof)
    monkeypatch.setattr(
        local_mod.os,
        "read",
        lambda *_a: (_ for _ in ()).throw(BlockingIOError()),
    )
    pump._on_readable()
    monkeypatch.setattr(
        local_mod.os,
        "read",
        lambda *_a: (_ for _ in ()).throw(OSError("bad read")),
    )
    pump._on_readable()
    pump.stop()  # idempotent when already closed
    monkeypatch.undo()
    os.close(w_fd)
    assert eof_count == 1

    class _ErrReadPipe:
        def read(self, _n: int) -> bytes:
            raise OSError("thread read err")

        def close(self) -> None:
            pass

    monkeypatch.setattr(local_mod.sys, "platform", "win32")
    pump_w = local_mod._StreamPump(
        loop,
        _ErrReadPipe(),
        seen.append,
        _on_eof,  # type: ignore[arg-type]
    )
    await pump_w.done.wait()

    class _OneChunkPipe:
        def read(self, _n: int) -> bytes:
            return b"once"

        def close(self) -> None:
            pass

    pump_w2 = local_mod._StreamPump(
        loop,
        _OneChunkPipe(),
        lambda _c: pump_w2.stop(),
        _on_eof,  # type: ignore[arg-type]
    )
    await pump_w2.done.wait()
    monkeypatch.undo()

    # _abort_error helper
    ctrl_none = AbortController()
    ctrl_none.abort(None)
    assert str(local_mod._abort_error(ctrl_none.signal)) == "aborted"
    ctrl_str = AbortController()
    ctrl_str.abort("stop now")
    assert str(local_mod._abort_error(ctrl_str.signal)) == "aborted: stop now"
    ctrl_exc = AbortController()
    err = RuntimeError("boom")
    ctrl_exc.abort(err)
    assert local_mod._abort_error(ctrl_exc.signal) is err

    # _process_group_has_live_members & _linux_process_group_has_live_members branches
    assert local_mod._linux_process_group_has_live_members(os.getpgrp()) is True
    assert local_mod._process_group_has_live_members(os.getpgrp()) is True

    monkeypatch.setattr(local_mod.sys, "platform", "win32")
    assert local_mod._process_group_has_live_members(123) is False
    monkeypatch.undo()

    monkeypatch.setattr(
        local_mod.os,
        "killpg",
        lambda *_a: (_ for _ in ()).throw(PermissionError("perm")),
    )
    assert local_mod._process_group_has_live_members(123) is True
    monkeypatch.setattr(
        local_mod.os,
        "killpg",
        lambda *_a: (_ for _ in ()).throw(OSError("other")),
    )
    assert local_mod._process_group_has_live_members(123) is False
    monkeypatch.setattr(local_mod.os, "killpg", lambda *_a: None)
    monkeypatch.setattr(local_mod.sys, "platform", "linux")
    monkeypatch.setattr(
        local_mod, "_linux_process_group_has_live_members", lambda _p: False
    )
    assert local_mod._process_group_has_live_members(123) is False
    monkeypatch.undo()

    # Mock /proc entries for _linux_process_group_has_live_members edge branches
    fake_proc = tmp_path / "proc"
    fake_proc.mkdir()
    (fake_proc / "notpid").mkdir()
    (fake_proc / "10").mkdir()  # no stat file -> OSError skip
    (fake_proc / "11").mkdir()
    (fake_proc / "11" / "stat").write_text("bad_format", encoding="ascii")
    (fake_proc / "12").mkdir()
    (fake_proc / "12" / "stat").write_text("12 (sh) S 1", encoding="ascii")
    (fake_proc / "13").mkdir()
    (fake_proc / "13" / "stat").write_text("13 (sh) S 1 bad 1", encoding="ascii")
    (fake_proc / "14").mkdir()
    (fake_proc / "14" / "stat").write_text("14 (sh) Z 1 999 999", encoding="ascii")
    (fake_proc / "15").mkdir()
    (fake_proc / "15" / "stat").write_text("15 (sh) Z 1 777 777", encoding="ascii")

    orig_path = local_mod.Path

    def _patched_path(arg: Any) -> Path:
        if arg == "/proc":
            return fake_proc
        if isinstance(arg, str) and arg.startswith("/proc/"):
            return fake_proc / arg.removeprefix("/proc/")
        return orig_path(arg)

    monkeypatch.setattr(local_mod, "Path", _patched_path)
    assert local_mod._linux_process_group_has_live_members(777) is False
    assert local_mod._linux_process_group_has_live_members(888) is None
    fake_proc.rmdir() if False else None
    monkeypatch.setattr(
        local_mod,
        "Path",
        lambda arg: tmp_path / "no-such-proc" if arg == "/proc" else orig_path(arg),
    )
    assert local_mod._linux_process_group_has_live_members(777) is None
    monkeypatch.undo()
