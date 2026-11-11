"""Unit tests for LocalShellService and ShellResult."""

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from anima import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.shell.service import (
    LocalShellService,
    ShellResult,
    _kill_process_tree,
    _read_bounded_stream,
)
from magitech_harness.tools.service import ToolService


@pytest.mark.asyncio
async def test_shell_service_run(tmp_path: Path) -> None:
    """Verifies shell execution returns code, stdout, and stderr, respecting cwd."""
    ctx = Context()
    ctx.use(LocalShellService)
    await ctx.flush()

    assert ctx.has("shell")

    # Command with both stdout and stderr
    code, stdout, stderr = await ctx.shell.run(
        "echo 'hello stdout' && echo 'error stderr' >&2"
    )
    assert code == 0
    assert "hello stdout" in stdout
    assert "error stderr" in stderr

    # Command respecting cwd (python prints the native cwd on every platform;
    # the Git Bash `pwd` on Windows prints a POSIX-style path instead)
    code, stdout, _stderr = await ctx.shell.run(
        'python3 -c "import os; print(os.getcwd())"', cwd=tmp_path
    )
    assert code == 0
    assert str(tmp_path.resolve()) in stdout

    # Failing command
    code, _stdout, _stderr = await ctx.shell.run("false")
    assert code != 0


@pytest.mark.asyncio
async def test_shell_service_does_not_register_tools() -> None:
    """Verifies LocalShellService strictly acts as an execution backend with zero domain tools."""
    ctx = Context()
    ctx.use(ToolService)
    ctx.use(LocalShellService)
    await ctx.flush()

    assert ctx.has("shell")
    assert ctx.has("tools")
    # Strictly zero built-in domain tools in core harness
    assert ctx.tools.get("bash") is None


@pytest.mark.asyncio
async def test_shell_result_attributes_and_indexing() -> None:
    """Verifies ShellResult supports structured attributes and tuple indexing/unpacking."""
    res = ShellResult(
        exit_code=0,
        stdout="out",
        stderr="err",
        timed_out=False,
        aborted=False,
        bytes_read=6,
    )
    assert res.exit_code == 0
    assert res.stdout == "out"
    assert res.stderr == "err"
    assert not res.timed_out
    assert not res.aborted
    assert res.bytes_read == 6

    # Test indexing
    assert res[0] == 0
    assert res[1] == "out"
    assert res[2] == "err"

    # Test unpacking
    code, out, err = res
    assert (code, out, err) == (0, "out", "err")


@pytest.mark.asyncio
async def test_shell_bounded_reading() -> None:
    """Verifies bounded reading truncates large streams retaining head and tail."""
    ctx = Context()
    ctx.use(LocalShellService)
    await ctx.flush()

    # Generate 1000 lines of numbers
    cmd = "python3 -c \"for i in range(1000): print(f'line {i}')\""
    res = await ctx.shell.run(cmd, max_output_bytes=200)

    assert res.exit_code == 0
    assert "line 0" in res.stdout
    assert "line 999" in res.stdout
    assert "[truncated" in res.stdout
    assert res.bytes_read > 200


@pytest.mark.asyncio
async def test_read_bounded_stream_none() -> None:
    """Verifies _read_bounded_stream handles None stream safely."""
    content, total = await _read_bounded_stream(None, 1024)
    assert content == ""
    assert total == 0


@pytest.mark.asyncio
async def test_shell_timeout_and_tree_kill() -> None:
    """Verifies command execution timeout terminates child process tree."""
    ctx = Context()
    ctx.use(LocalShellService)
    await ctx.flush()

    res = await ctx.shell.run("sleep 5", timeout_seconds=0.1)
    assert res.timed_out
    assert res.exit_code != 0


@pytest.mark.asyncio
async def test_shell_abort_signal_pre_aborted() -> None:
    """Verifies shell execution exits immediately if AbortSignal is pre-aborted."""
    ctx = Context()
    ctx.use(LocalShellService)
    await ctx.flush()

    sig = AbortSignal()
    sig.abort("Pre-cancelled")

    res = await ctx.shell.run("sleep 5", signal=sig)
    assert res.aborted
    assert res.exit_code == -1
    assert res.stdout == ""


@pytest.mark.asyncio
async def test_shell_abort_signal_dynamic() -> None:
    """Verifies shell execution aborts mid-flight when signal triggers."""
    ctx = Context()
    ctx.use(LocalShellService)
    await ctx.flush()

    sig = AbortSignal()

    async def _abort_later() -> None:
        await asyncio.sleep(0.1)
        sig.abort("User cancelled")

    task = asyncio.create_task(_abort_later())
    res = await ctx.shell.run("sleep 5", signal=sig)
    await task

    assert res.aborted
    assert res.exit_code != 0


def test_kill_process_tree_posix_success() -> None:
    """Verifies POSIX tree killing resolves the group and signals it."""
    fake_signal = SimpleNamespace(SIGKILL=9)
    with (
        patch("sys.platform", "linux"),
        patch("os.getpgid", create=True, return_value=1234),
        patch("os.killpg", create=True) as mock_killpg,
        patch("magitech_harness.shell.service.sys_signal", fake_signal),
    ):
        _kill_process_tree(9999)
        mock_killpg.assert_called_once_with(1234, 9)


def test_kill_process_tree_posix_exceptions() -> None:
    """Verifies POSIX tree killing handles ProcessLookupError, PermissionError, and generic Exception."""
    with (
        patch("sys.platform", "linux"),
        patch("os.getpgid", create=True) as mock_getpgid,
    ):
        mock_getpgid.side_effect = ProcessLookupError()
        _kill_process_tree(999999)

        mock_getpgid.side_effect = PermissionError()
        _kill_process_tree(999999)

        mock_getpgid.side_effect = OSError("Other OS error")
        _kill_process_tree(999999)


@pytest.mark.asyncio
async def test_read_bounded_stream_exact_and_small_tail() -> None:
    """Verifies bounded reader handles stream with tail <= half and total > max."""

    # max_bytes = 10, half = 5. Feed 8 bytes (head=5, tail=3), then 3 bytes (tail=3+3=6 > 5 -> 5)
    # Then test total <= max (e.g. 10 bytes)
    class FakeStream:
        def __init__(self, chunks: list[bytes]) -> None:
            self.chunks = list(chunks)

        async def read(self, _n: int) -> bytes:
            if self.chunks:
                return self.chunks.pop(0)
            return b""

    # Tail not exceeding half
    stream1 = FakeStream([b"12345", b"67"])
    content1, total1 = await _read_bounded_stream(stream1, 10)  # type: ignore[arg-type]
    assert content1 == "1234567"
    assert total1 == 7

    # Tail exceeding half and truncating
    stream2 = FakeStream([b"12345", b"67890", b"abcde"])
    content2, total2 = await _read_bounded_stream(stream2, 6)  # type: ignore[arg-type]
    assert "[truncated" in content2
    assert total2 == 15


@pytest.mark.asyncio
async def test_shell_run_returncode_none_and_no_pid() -> None:
    """Verifies fallback when proc.returncode is None and proc.pid is None."""
    ctx = Context()
    ctx.use(LocalShellService)
    await ctx.flush()

    class FakeProc:
        def __init__(self, pid: Any = None) -> None:
            self.pid = pid
            self.returncode = None
            self._waited = asyncio.Event()

        async def wait(self) -> int:
            await asyncio.sleep(0.05)
            return 0

        @property
        def stdout(self) -> Any:
            return None

        @property
        def stderr(self) -> Any:
            return None

    sig = AbortSignal()
    with patch("asyncio.create_subprocess_shell", return_value=FakeProc()):
        res = await ctx.shell.run("dummy", signal=sig)
        assert res.exit_code == -1

    # Trigger abort with no pid
    with patch("asyncio.create_subprocess_shell", return_value=FakeProc(pid=None)):
        sig2 = AbortSignal()

        async def _trigger() -> None:
            await asyncio.sleep(0.01)
            sig2.abort()

        asyncio.create_task(_trigger())
        res2 = await ctx.shell.run("dummy", signal=sig2)
        assert res2.aborted

    # Timeout with no pid
    class FakeTimeoutProc(FakeProc):
        async def wait(self) -> int:
            await asyncio.sleep(1)
            return 0

    with patch(
        "asyncio.create_subprocess_shell", return_value=FakeTimeoutProc(pid=None)
    ):
        res_timeout = await ctx.shell.run("dummy", timeout_seconds=0.01)
        assert res_timeout.timed_out


def test_kill_process_tree_windows() -> None:
    """Verifies Windows process tree killing invokes taskkill."""
    with (
        patch("sys.platform", "win32"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        _kill_process_tree(12345)
        mock_run.assert_called_once_with(
            ["taskkill", "/F", "/T", "/PID", "12345"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        # Verify exception tolerance
        mock_run.side_effect = OSError("taskkill failed")
        _kill_process_tree(12345)
