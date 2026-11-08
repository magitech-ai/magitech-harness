"""LocalShellService (ctx.shell): Process and command execution provider.

Features cross-platform process tree termination, in-memory bounded reading,
and cooperative abort signal integration.
"""

from __future__ import annotations

import asyncio
import os
import signal as sys_signal
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anima import Context, Service

from magitech_harness.agent.cancellation import AbortSignal


@dataclass(frozen=True)
class ShellResult:
    """Structured execution result from a shell command."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    aborted: bool = False
    bytes_read: int = 0

    def __iter__(self) -> Iterator[Any]:
        """Support 3-element tuple unpacking for backwards compatibility."""
        return iter((self.exit_code, self.stdout, self.stderr))

    def __getitem__(self, index: int) -> Any:
        """Support 3-element tuple indexing for backwards compatibility."""
        return (self.exit_code, self.stdout, self.stderr)[index]


def _kill_process_tree(pid: int) -> None:
    """Terminates an entire process tree across POSIX and Windows."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            pass
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, sys_signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


async def _read_bounded_stream(
    stream: asyncio.StreamReader | None,
    max_bytes: int,
) -> tuple[str, int]:
    """Reads an asyncio stream up to EOF, retaining head and tail within max_bytes budget."""
    if stream is None:
        return "", 0

    effective_max = max(1, max_bytes)
    half = effective_max // 2
    head = bytearray()
    tail = bytearray()
    total_bytes = 0

    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        total_bytes += len(chunk)
        if len(head) < half:
            needed = half - len(head)
            head.extend(chunk[:needed])
            remaining = chunk[needed:]
        else:
            remaining = chunk

        if remaining:
            tail.extend(remaining)
            if len(tail) > half:
                tail = tail[-half:]

    if total_bytes <= effective_max:
        content = (head + tail).decode(errors="replace")
        return content, total_bytes

    truncated_bytes = total_bytes - len(head) - len(tail)
    marker = f"\n... [truncated {truncated_bytes} bytes] ...\n".encode()
    content = (head + marker + tail).decode(errors="replace")
    return content, total_bytes


class LocalShellService(Service):
    """Service providing shell command execution backend."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "shell")

    async def run(
        self,
        command: str,
        cwd: Path | str | None = None,
        *,
        timeout_seconds: float | None = None,
        signal: AbortSignal | None = None,
        max_output_bytes: int = 64 * 1024,
    ) -> ShellResult:
        """Execute a shell command asynchronously with bounded output and tree termination."""
        if signal is not None and signal.aborted:
            return ShellResult(
                exit_code=-1,
                stdout="",
                stderr="",
                timed_out=False,
                aborted=True,
                bytes_read=0,
            )

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,
        )

        aborted = False
        timed_out = False

        def _on_abort() -> None:
            nonlocal aborted
            aborted = True
            if proc.pid:
                _kill_process_tree(proc.pid)

        remove_listener = None
        if signal is not None:
            remove_listener = signal.add_listener(_on_abort)

        stdout_task = asyncio.create_task(
            _read_bounded_stream(proc.stdout, max_output_bytes)
        )
        stderr_task = asyncio.create_task(
            _read_bounded_stream(proc.stderr, max_output_bytes)
        )

        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
                except TimeoutError:
                    timed_out = True
                    if proc.pid:
                        _kill_process_tree(proc.pid)
                    await proc.wait()
            else:
                await proc.wait()
        finally:
            if remove_listener is not None:
                remove_listener()

        stdout_str, stdout_bytes = await stdout_task
        stderr_str, stderr_bytes = await stderr_task
        total_bytes = stdout_bytes + stderr_bytes

        exit_code = proc.returncode if proc.returncode is not None else -1

        return ShellResult(
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
            timed_out=timed_out,
            aborted=aborted,
            bytes_read=total_bytes,
        )
