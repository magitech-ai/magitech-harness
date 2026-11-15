"""Shell execution types, results, and handle contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Protocol, TypedDict

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.subprocess.types import (
    MAGITECH_ENV_PREFIX,
    CollectedOutput,
    CollectedOutputReader,
    OutputReadSlice,
)

ShellExpiryPolicy = Literal["kill", "none"]
ShellProcessStatus = Literal["running", "completed", "killed"]


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


class ShellObservedStreams(TypedDict):
    """Non-consuming offset readers over captured stdout/stderr streams."""

    stdout: CollectedOutputReader
    stderr: CollectedOutputReader


class ShellSandboxInfo(TypedDict):
    """Sandbox execution facts when handled by a sandboxing executor."""

    mode: str
    denied: bool
    enforcement: NotRequired[str]
    runnerFailed: NotRequired[bool]


class ShellExecRequest(TypedDict):
    """Caller request before executor defaults and caps are applied."""

    command: str
    workdir: NotRequired[str | None]
    timeoutMs: NotRequired[float | None]
    onExpiry: NotRequired[ShellExpiryPolicy | None]
    stdoutMaxBytes: NotRequired[int | None]
    signal: NotRequired[AbortSignal | None]
    stdin: NotRequired[str | None]
    env: NotRequired[Mapping[str, str] | None]
    magitechEnv: NotRequired[Mapping[str, str] | None]
    sandboxPolicy: NotRequired[Any]


class ShellExecSpec(TypedDict):
    """Resolved execution specification produced by :meth:`ShellExecutor.resolve`."""

    command: str
    workdir: str
    timeoutMs: float
    onExpiry: ShellExpiryPolicy
    stdoutMaxBytes: int
    signal: NotRequired[AbortSignal]
    stdin: NotRequired[str]
    env: NotRequired[Mapping[str, str]]
    magitechEnv: NotRequired[Mapping[str, str]]
    sandboxPolicy: Any


class ShellProcessRead(TypedDict):
    """Consuming incremental output delta and loss/spill metadata."""

    delta: str
    lossy: bool
    stdoutSpillPath: NotRequired[str]
    stderrSpillPath: NotRequired[str]


class ShellRunResult(TypedDict):
    """Foreground execution outcome projection."""

    exitCode: int | None
    signal: str | None
    timedOut: bool
    aborted: bool
    timeoutMs: float
    stdout: CollectedOutput
    stderr: CollectedOutput
    sandbox: NotRequired[ShellSandboxInfo]


class ShellProcess(Protocol):
    """Background handle returned after command launch."""

    status: ShellProcessStatus
    exit_code: int | None
    signal: str | None
    sandbox: ShellSandboxInfo | None

    @property
    def done(self) -> asyncio.Future[None]:
        """Future resolving when the process settles (never rejects)."""

    @property
    def observed(self) -> ShellObservedStreams:
        """Non-consuming stream readers."""

    def read_output(self) -> ShellProcessRead:
        """Read incremental output since the prior consuming call."""

    def kill(self) -> bool:
        """Terminate process tree if running; returns True once when transitioned."""


class ShellExecution(ShellProcess, Protocol):
    """Unified process handle plus memoized foreground ``result()`` projection."""

    def result(self) -> asyncio.Future[ShellRunResult]:
        """Return settled foreground result or raise on infrastructure spawn error."""


__all__ = [
    "MAGITECH_ENV_PREFIX",
    "CollectedOutput",
    "CollectedOutputReader",
    "OutputReadSlice",
    "ShellExecRequest",
    "ShellExecSpec",
    "ShellExecution",
    "ShellExpiryPolicy",
    "ShellObservedStreams",
    "ShellProcess",
    "ShellProcessRead",
    "ShellProcessStatus",
    "ShellResult",
    "ShellRunResult",
    "ShellSandboxInfo",
]
