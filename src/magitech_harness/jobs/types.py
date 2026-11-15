"""Background job types, constants, and lifecycle contracts (ctx.jobs)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, NotRequired, Protocol, TypedDict

TASK_WAIT_TIMEOUT = "TASK_WAIT_TIMEOUT"

DEFAULT_MAX_CONCURRENT_JOBS_PER_OWNER = 10
DEFAULT_RETAIN_BYTES = 262_144
DEFAULT_SETTLED_RETAIN_BYTES = 16_384
DEFAULT_PUMP_POLL_MS = 150

JobStatus = Literal["running", "stopping", "completed", "killed", "failed"]
JobSettleCause = Literal["producer", "kill", "teardown"]


def is_terminal_status(status: JobStatus) -> bool:
    """Return True for terminal job states (completed, killed, failed)."""
    return status in ("completed", "killed", "failed")


class JobToolCallRef(TypedDict):
    """Source tool call metadata for durable session lifecycle records."""

    callId: str
    rootCallId: str


class JobOutcome(TypedDict):
    """Terminal result returned by a producer's ``done`` awaitable."""

    status: Literal["completed", "killed", "failed"]
    detail: NotRequired[str]
    result: NotRequired[Any]


class JobAppendOptions(TypedDict, total=False):
    """Per-append stream channel and gap metadata."""

    channel: str
    gapBefore: Literal[True]


class JobOutputChunk(TypedDict):
    """Single ring output chunk."""

    at: int
    text: str
    channel: NotRequired[str]
    gapBefore: NotRequired[Literal[True]]


class JobOutputRead(TypedDict):
    """Byte-offset read result over a job's output ring."""

    chunks: list[JobOutputChunk]
    next: int
    lossy: bool


class JobOutputSummary(TypedDict):
    """Observable byte-window and spill summary attached to a :class:`JobView`."""

    total: int
    earliest: int
    spillPaths: NotRequired[list[str]]


class JobView(TypedDict):
    """Serializable point-in-time snapshot of a tracked background job."""

    id: str
    kind: str
    label: str
    owner: NotRequired[str]
    outputLimitBytes: NotRequired[int]
    status: JobStatus
    progress: NotRequired[str]
    detail: NotRequired[str]
    startedAt: int
    finishedAt: NotRequired[int]
    output: JobOutputSummary


class JobRead(TypedDict):
    """Cursor-advancing incremental read paired with optional terminal result."""

    chunks: list[JobOutputChunk]
    lossy: bool
    result: NotRequired[Any]
    job: JobView


class JobOutputSourceRead(TypedDict):
    """Pull result returned by an attached :class:`JobOutputSource`."""

    text: str
    nextOffset: int
    lossy: NotRequired[bool]
    spillPath: NotRequired[str]


class JobOutputSource(Protocol):
    """Pull output source drained periodically by the registry pump."""

    @property
    def channel(self) -> str | None:
        """Optional stream label (e.g., ``stdout``, ``stderr``)."""

    def read(self, from_byte: int) -> JobOutputSourceRead:
        """Read incremental output starting from *from_byte*."""


class JobHandle(Protocol):
    """Producer-side handle passed to :meth:`JobSpec.run`."""

    @property
    def id(self) -> str:
        """Unique assigned job identifier."""

    def append(self, text: str, options: JobAppendOptions | None = None) -> None:
        """Append output text chunk to the job's bounded ring."""

    def update_progress(self, line: str) -> None:
        """Update single-line progress status displayed on :class:`JobView`."""


class JobRunHooks(Protocol):
    """Lifecycle hooks returned synchronously by :attr:`JobSpec` runner."""

    @property
    def done(self) -> Awaitable[JobOutcome]:
        """Awaitable resolving with terminal :class:`JobOutcome`."""

    def cancel(self, reason: str | None = None) -> None:
        """Initiate cooperative or forced job cancellation."""


class JobSpec(TypedDict):
    """Specification passed to :meth:`JobRegistry.start`."""

    kind: str
    label: str
    run: Callable[[JobHandle], JobRunHooks]
    owner: NotRequired[str | None]
    outputLimitBytes: NotRequired[int]
    toolCall: NotRequired[JobToolCallRef]
    output: NotRequired[Sequence[JobOutputSource]]


class JobEvent(TypedDict):
    """Structured event emitted through ``ctx.jobs.events``."""

    type: Literal[
        "registered",
        "stopping",
        "settled",
        "progress",
        "output",
        "removed",
    ]
    job: NotRequired[JobView]
    id: NotRequired[str]
    owner: NotRequired[str]
    total: NotRequired[int]
    cause: NotRequired[JobSettleCause]
    awaited: NotRequired[bool]


class JobEventFilter(TypedDict, total=False):
    """Optional subscription filter for :meth:`JobEvents.subscribe`."""

    owner: str
    owners: Literal["all", "scope"]


class JobEvents(Protocol):
    """Event stream interface on ``ctx.jobs.events``."""

    def subscribe(
        self,
        filter_spec: JobEventFilter | None,
        listener: Callable[[JobEvent], Any],
    ) -> Callable[[], Awaitable[None] | None]:
        """Register *listener* bound to the accessing context's lifecycle."""


class JobRegisteredData(TypedDict):
    """Session event payload for ``job/registered``."""

    jobId: str
    owner: str
    kind: str
    label: str
    startedAt: int
    toolCall: NotRequired[JobToolCallRef]


class JobSettledData(TypedDict):
    """Session event payload for ``job/settled``."""

    jobId: str
    owner: str
    kind: str
    label: str
    startedAt: int
    finishedAt: int
    status: JobStatus
    cause: JobSettleCause
    producerReported: bool
    detail: NotRequired[str]
    toolCall: NotRequired[JobToolCallRef]


__all__ = [
    "DEFAULT_MAX_CONCURRENT_JOBS_PER_OWNER",
    "DEFAULT_PUMP_POLL_MS",
    "DEFAULT_RETAIN_BYTES",
    "DEFAULT_SETTLED_RETAIN_BYTES",
    "TASK_WAIT_TIMEOUT",
    "JobAppendOptions",
    "JobEvent",
    "JobEventFilter",
    "JobEvents",
    "JobHandle",
    "JobOutcome",
    "JobOutputChunk",
    "JobOutputRead",
    "JobOutputSource",
    "JobOutputSourceRead",
    "JobOutputSummary",
    "JobRead",
    "JobRegisteredData",
    "JobRunHooks",
    "JobSettleCause",
    "JobSettledData",
    "JobSpec",
    "JobStatus",
    "JobToolCallRef",
    "JobView",
    "is_terminal_status",
]
