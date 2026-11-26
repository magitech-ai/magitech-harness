"""Abstract shell executor seam (``ctx.shell``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from animas import Context, Service

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.shell.types import (
    ShellExecRequest,
    ShellExecSpec,
    ShellExecution,
    ShellResult,
)


class ShellExecutor(Service, ABC):
    """Abstract ``ctx.shell`` service for resolved command execution."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "shell")

    @property
    def sandbox_mode(self) -> str | None:
        """Default sandbox mode applied by this executor, or None."""
        return None

    @abstractmethod
    def resolve(self, request: ShellExecRequest) -> ShellExecSpec:
        """Fill defaults and clamp limits into a full :class:`ShellExecSpec`."""

    @abstractmethod
    async def execute(self, spec: ShellExecSpec) -> ShellExecution:
        """Prepare and start *spec*, returning a live :class:`ShellExecution`."""

    async def run(
        self,
        command: str,
        cwd: Path | str | None = None,
        *,
        timeout_seconds: float | None = None,
        signal: AbortSignal | None = None,
        max_output_bytes: int = 64 * 1024,
    ) -> ShellResult:
        """Execute *command* to completion and project to :class:`ShellResult`."""
        request: ShellExecRequest = {
            "command": command,
            "stdoutMaxBytes": max_output_bytes,
        }
        if cwd is not None:
            request["workdir"] = str(cwd)
        if signal is not None:
            request["signal"] = signal
        if timeout_seconds is not None and timeout_seconds > 0:
            request["timeoutMs"] = timeout_seconds * 1000.0
            request["onExpiry"] = "kill"
        else:
            request["onExpiry"] = "none"

        spec = self.resolve(request)
        proc = await self.execute(spec)
        res = await proc.result()
        stdout = res["stdout"]["text"]
        stderr = res["stderr"]["text"]
        exit_code = res["exitCode"] if res["exitCode"] is not None else -1
        stdout_data = res["stdout"]
        stderr_data = res["stderr"]
        return ShellResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=res["timedOut"],
            aborted=res["aborted"],
            bytes_read=len(stdout.encode("utf-8")) + len(stderr.encode("utf-8")),
            truncated=bool(
                stdout_data.get("truncated") or stderr_data.get("truncated")
            ),
            stdout_spill_path=stdout_data.get("spillPath"),
            stderr_spill_path=stderr_data.get("spillPath"),
            sandbox=res.get("sandbox"),
        )


__all__ = ["ShellExecutor"]
