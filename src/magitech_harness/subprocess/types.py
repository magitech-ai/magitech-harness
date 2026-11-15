"""Shared subprocess types, environment constants, and protocol contracts."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from typing import Literal, NotRequired, Protocol, TypedDict

from magitech_harness.agent.cancellation import AbortSignal

MAGITECH_ENV_PREFIX = "MAGITECH_"

SENSITIVE_ENV_PATTERN = re.compile(
    r"^(?:"
    r"MAGITECH_|MTH_|ANTHROPIC_|OPENAI_|OPENROUTER_|GEMINI_|GOOGLE_|"
    r"AZURE_OPENAI_|COHERE_|MISTRAL_|TOGETHER_|GROQ_|FIREWORKS_|"
    r"REPLICATE_|PERPLEXITY_|XAI_|AI21_|HUGGINGFACE_|HF_|VERTEX_"
    r").+$"
)


class SubprocessSpillSpec(TypedDict):
    """Spill cap configuration when a stream exceeds its in-memory tail window."""

    maxBytes: int


class SubprocessCollect(TypedDict):
    """Stream collection mode retaining a bounded tail with optional spill file."""

    kind: Literal["collect"]
    maxBytes: int
    spill: NotRequired[SubprocessSpillSpec]


SubprocessOutputMode = Literal["pipe", "inherit"] | SubprocessCollect


class SubprocessStdinBytes(TypedDict):
    """One-shot byte payload written to child stdin then closed."""

    kind: Literal["bytes"]
    data: bytes


SubprocessStdinMode = Literal["pipe", "ignore"] | SubprocessStdinBytes


class SubprocessStdioSpec(TypedDict):
    """Per-stream stdio wiring specification."""

    stdin: SubprocessStdinMode
    stdout: SubprocessOutputMode
    stderr: SubprocessOutputMode


class SubprocessSpawnSpec(TypedDict):
    """Request passed to :meth:`SubprocessRuntime.spawn`."""

    argv: Sequence[str]
    cwd: str
    stdio: SubprocessStdioSpec
    graceMs: float
    env: NotRequired[Mapping[str, str] | None]
    signal: NotRequired[AbortSignal | None]


class SubprocessOutcome(TypedDict):
    """Terminal exit classification for a spawned process."""

    exitCode: int | None
    signal: str | None


class CollectedOutput(TypedDict):
    """Final collected output after settlement and spill seal."""

    text: str
    truncated: bool
    spillPath: NotRequired[str]


class OutputReadSlice(TypedDict):
    """Incremental read window in whole-stream byte coordinates."""

    text: str
    nextOffset: int
    lossy: bool
    spillPath: NotRequired[str]


class CollectedOutputReader(Protocol):
    """Live & final reader interface backed by an in-memory tail + optional spill."""

    def read_from(self, from_byte: int) -> OutputReadSlice:
        """Read bytes emitted since *from_byte*."""

    def snapshot(self) -> tuple[bytes, int]:
        """Return ``(tail_bytes, total_bytes_seen)`` without sealing."""

    def seal(self) -> None:
        """Seal any open spill descriptor."""

    def finalize(self) -> CollectedOutput:
        """Seal spill and return final collected output summary."""


class SubprocessCollected(TypedDict, total=False):
    """Optional collected stream readers attached to a subprocess handle."""

    stdout: CollectedOutputReader
    stderr: CollectedOutputReader


class SubprocessHandle(Protocol):
    """Lifecycle handle returned by :meth:`SubprocessRuntime.spawn`."""

    @property
    def pid(self) -> int | None:
        """OS process ID if launch succeeded."""

    @property
    def stdin(self) -> asyncio.StreamWriter | None:
        """Exposed writer when ``stdio.stdin == 'pipe'``."""

    @property
    def stdout(self) -> asyncio.StreamReader | None:
        """Exposed reader when ``stdio.stdout == 'pipe'``."""

    @property
    def stderr(self) -> asyncio.StreamReader | None:
        """Exposed reader when ``stdio.stderr == 'pipe'``."""

    @property
    def collected(self) -> SubprocessCollected:
        """Collected output readers for streams configured with ``collect``."""

    @property
    def done(self) -> asyncio.Future[SubprocessOutcome]:
        """Future resolving with exit outcome or rejecting on spawn/abort error."""

    def terminate(self) -> None:
        """Start SIGTERM -> grace -> SIGKILL escalation for the spawned tree."""

    def terminate_for_host_exit(self) -> None:
        """Best-effort immediate tree termination during synchronous host exit."""

    async def wait_for_exit(self, signal: AbortSignal | None = None) -> bool:
        """Wait until the managed process tree has exited or *signal* aborts."""


__all__ = [
    "MAGITECH_ENV_PREFIX",
    "SENSITIVE_ENV_PATTERN",
    "CollectedOutput",
    "CollectedOutputReader",
    "OutputReadSlice",
    "SubprocessCollect",
    "SubprocessCollected",
    "SubprocessHandle",
    "SubprocessOutcome",
    "SubprocessOutputMode",
    "SubprocessSpawnSpec",
    "SubprocessSpillSpec",
    "SubprocessStdinBytes",
    "SubprocessStdinMode",
    "SubprocessStdioSpec",
]
