"""Background job registry seam and process-local provider."""

from __future__ import annotations

from magitech_harness.jobs.archive_admission import install_job_archive_admission
from magitech_harness.jobs.events import JobEventHub, JobLayer
from magitech_harness.jobs.local import LocalJobRegistry
from magitech_harness.jobs.pump import PumpHandle, PumpSink, start_pump
from magitech_harness.jobs.ring import OutputRing
from magitech_harness.jobs.service import JobRegistry
from magitech_harness.jobs.types import (
    DEFAULT_MAX_CONCURRENT_JOBS_PER_OWNER,
    DEFAULT_PUMP_POLL_MS,
    DEFAULT_RETAIN_BYTES,
    DEFAULT_SETTLED_RETAIN_BYTES,
    TASK_WAIT_TIMEOUT,
    JobAppendOptions,
    JobEvent,
    JobEventFilter,
    JobEvents,
    JobHandle,
    JobOutcome,
    JobOutputChunk,
    JobOutputRead,
    JobOutputSource,
    JobOutputSourceRead,
    JobOutputSummary,
    JobRead,
    JobRegisteredData,
    JobRunHooks,
    JobSettleCause,
    JobSettledData,
    JobSpec,
    JobStatus,
    JobToolCallRef,
    JobView,
    is_terminal_status,
)

__all__ = [
    "DEFAULT_MAX_CONCURRENT_JOBS_PER_OWNER",
    "DEFAULT_PUMP_POLL_MS",
    "DEFAULT_RETAIN_BYTES",
    "DEFAULT_SETTLED_RETAIN_BYTES",
    "TASK_WAIT_TIMEOUT",
    "JobAppendOptions",
    "JobEvent",
    "JobEventFilter",
    "JobEventHub",
    "JobEvents",
    "JobHandle",
    "JobLayer",
    "JobOutcome",
    "JobOutputChunk",
    "JobOutputRead",
    "JobOutputSource",
    "JobOutputSourceRead",
    "JobOutputSummary",
    "JobRead",
    "JobRegisteredData",
    "JobRegistry",
    "JobRunHooks",
    "JobSettleCause",
    "JobSettledData",
    "JobSpec",
    "JobStatus",
    "JobToolCallRef",
    "JobView",
    "LocalJobRegistry",
    "OutputRing",
    "PumpHandle",
    "PumpSink",
    "install_job_archive_admission",
    "is_terminal_status",
    "start_pump",
]
