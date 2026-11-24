"""Archive-admission seam connecting active background jobs to Workspace checks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from animas import Context

from magitech_harness.jobs.types import JobView

if TYPE_CHECKING:
    from magitech_harness.jobs.service import JobRegistry


def _running_jobs(registry: JobRegistry, owner: str) -> list[JobView]:
    return [
        job
        for job in registry.list(owner)
        if job.get("owner") == owner and job["status"] in ("running", "stopping")
    ]


def install_job_archive_admission(
    ctx: Context, registry: JobRegistry
) -> Callable[[], None]:
    """Answer Workspace activity and stop hooks using the abstract JobRegistry."""

    async def _on_activity(
        request: dict[str, Any], next_fn: Callable[[], Awaitable[list[Any]]]
    ) -> list[Any]:
        session_id = request["sessionId"]
        jobs = _running_jobs(registry, session_id)
        rest = await next_fn()
        if not jobs:
            return rest
        own: dict[str, Any] = {
            "kind": "job",
            "items": [{"id": job["id"], "label": job["label"]} for job in jobs],
        }
        return [own, *rest]

    def _on_stop(request: dict[str, Any]) -> None:
        session_id = request["sessionId"]
        for job in _running_jobs(registry, session_id):
            try:
                registry.kill(job["id"], session_id, "session archived")
            except Exception as exc:  # noqa: BLE001 - isolate failing producer cancels
                ctx.logger.warning(
                    f'jobs: killing "{job["id"]}" for an archived Session failed: {exc}'
                )

    dispose_activity = ctx.on("workspace/session-activity", _on_activity)
    dispose_stop = ctx.on("workspace/session-stop", _on_stop)

    def _dispose() -> None:
        dispose_activity()
        dispose_stop()

    return _dispose


__all__ = ["install_job_archive_admission"]
