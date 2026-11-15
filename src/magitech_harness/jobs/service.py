"""Abstract background job registry service definition (``ctx.jobs``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Literal

from anima import Context, Service

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.jobs.archive_admission import install_job_archive_admission
from magitech_harness.jobs.types import (
    JobEvents,
    JobOutputRead,
    JobRead,
    JobSpec,
    JobView,
)


class JobRegistry(Service, ABC):
    """Abstract ``ctx.jobs`` service contract for background work tracking."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "jobs")
        install_job_archive_admission(ctx, self)

    @property
    @abstractmethod
    def events(self) -> JobEvents:
        """Filtered job lifecycle and output event stream."""

    @abstractmethod
    def start(self, spec: JobSpec) -> str:
        """Preflight access/limits, start *spec*, and atomically register its id."""

    @abstractmethod
    def list(self, caller: str | None = None) -> list[JobView]:
        """List unowned and *caller*-owned jobs in registration order."""

    @abstractmethod
    def get(self, job_id: str, caller: str | None = None) -> JobView:
        """Return a fresh snapshot of *job_id* for *caller* without mutating cursors."""

    @abstractmethod
    def read(self, job_id: str, caller: str | None = None) -> JobRead:
        """Consume ring chunks from the model cursor and return terminal result once."""

    @abstractmethod
    def read_at(
        self, job_id: str, from_offset: int, caller: str | None = None
    ) -> JobOutputRead:
        """Read retained ring chunks overlapping ``[from_offset, total)`` non-destructively."""

    @abstractmethod
    def kill(
        self,
        job_id: str,
        caller: str | None = None,
        reason: str | None = None,
    ) -> Literal["requested", "already-finished"]:
        """Cancel a live job or report ``already-finished`` for terminal jobs."""

    @abstractmethod
    async def wait(
        self,
        job_id: str,
        timeout_ms: float,
        caller: str | None = None,
        signal: AbortSignal | None = None,
    ) -> JobView:
        """Wait for settlement, *timeout_ms*, or caller abort without cancelling work."""

    @abstractmethod
    def remove(self, job_id: str, caller: str | None = None) -> None:
        """Remove a settled job record and announce its removal."""

    @abstractmethod
    def attach_controller(self, name: str) -> Callable[[], Awaitable[None] | None]:
        """Attach an effect-scoped controller serving its registering context scope."""


__all__ = ["JobRegistry"]
