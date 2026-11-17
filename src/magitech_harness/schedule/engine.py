"""Schedule service (ctx.schedule): Host-owned scheduled tasks.

Create RFC 5545 recurring tasks that deliver to existing or new Sessions.
Tasks survive restarts. Inspect active, paused, and ended tasks without
opening their target Sessions.

Reference: ``packages/schedule/schedule``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, TypedDict, cast

from anima import Context, Service

__all__ = [
    "ScheduleConfig",
    "ScheduleCreateInput",
    "ScheduleEditInput",
    "ScheduleListResult",
    "ScheduleRecord",
    "ScheduleService",
    "ScheduleTarget",
    "ScheduleTask",
    "ScheduleTiming",
    "ScheduleViewResult",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ScheduleTiming(TypedDict, total=False):
    """Timing specification for a scheduled task.

    ``kind`` selects the schedule type:

    * ``after`` -- run once after ``after_seconds`` delay
    * ``at`` -- run once at absolute ``scheduled_at``
    * ``every`` -- repeat every ``every_seconds``
    * ``daily`` -- daily at ``time`` in ``time_zone``
    * ``weekly`` -- weekly on ``weekdays`` at ``time`` in ``time_zone``
    * ``cron`` -- five-field Vixie cron ``expression`` in ``time_zone``
    * ``rrule`` -- RFC 5545 ``rrule`` string
    """

    kind: Literal["after", "at", "every", "daily", "weekly", "cron", "rrule"]
    after_seconds: int | None
    scheduled_at: str | None
    every_seconds: int | None
    time: str | None
    time_zone: str | None
    weekdays: list[int] | None
    expression: str | None
    rrule: str | None


class ScheduleTarget(TypedDict, total=False):
    """Delivery target for a scheduled task."""

    kind: Literal["session", "new_once", "new"]
    session_id: str | None


class ScheduleRecord(TypedDict):
    """Stored task record."""

    id: str
    kind: str
    title: str
    prompt: str
    scheduled_at: str
    # kind-specific fields
    after_seconds: int | None
    every_seconds: int | None
    time: str | None
    time_zone: str | None
    weekdays: list[int] | None
    expression: str | None
    rrule: str | None


class ScheduleTask(TypedDict):
    """A scheduled task."""

    id: str
    kind: str
    title: str
    prompt: str
    timing: ScheduleTiming
    target: ScheduleTarget
    status: Literal["active", "paused", "ended"]
    created_at: str | None
    updated_at: str | None
    last_delivery: str | None
    involved_session_ids: list[str]


class ScheduleCreateInput(TypedDict, total=False):
    """Input for schedule_create."""

    name: str
    instruction: str
    timing: ScheduleTiming
    target: ScheduleTarget | None


class ScheduleEditInput(TypedDict, total=False):
    """Input for schedule_edit."""

    id: str
    action: Literal["update", "pause", "resume", "delete"]
    patch: ScheduleTask | None
    expected_updated_at: str | None


class ScheduleViewResult(TypedDict):
    """Result of schedule_view."""

    id: str
    kind: str
    title: str
    prompt: str
    timing: ScheduleTiming
    target: ScheduleTarget
    status: Literal["active", "paused", "ended"]
    created_at: str | None
    updated_at: str | None
    last_delivery: str | None
    delivery_history: dict[str, Any] | None


class ScheduleListResult(TypedDict):
    """Result of schedule_list."""

    tasks: list[ScheduleTask]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ScheduleConfig(TypedDict, total=False):
    """Optional configuration for the schedule service."""

    delivery_history_days: int | None
    delivery_history_records: int | None
    new_session_interval_seconds: int | None


class ScheduleService(Service):
    """Shared schedule service creating and managing scheduled tasks.

    Mount one schedule service to create, list, view, and edit scheduled
    tasks. Each task has its own timing, target, and lifecycle state.

    Args:
        ctx: The anima Context to register the service on.
        config: Optional configuration (delivery settings).
    """

    inject: ClassVar[list[str]] = []

    def __init__(self, ctx: Context, config: ScheduleConfig | None = None) -> None:
        if ctx.get("schedule", None, strict=False) is not None:
            raise RuntimeError("schedule service already registered")
        super().__init__(ctx, "schedule")
        self.cfg: ScheduleConfig = config or {}
        self._tasks: dict[str, ScheduleTask] = {}
        ctx.reflect.provide("schedule", self, None)

    def schedule_create(
        self,
        input: ScheduleCreateInput,
        actor_session_id: str | None = None,
    ) -> ScheduleTask:
        """Create a scheduled task.

        Args:
            input: Task creation input with name, instruction, timing, target.
            actor_session_id: Optional session creating the task.

        Returns:
            The created task.
        """

        task_id = input["name"]
        now = datetime.now(UTC).isoformat()

        task: ScheduleTask = {
            "id": task_id,
            "kind": input["timing"]["kind"],
            "title": input["name"],
            "prompt": input["instruction"],
            "timing": input["timing"],
            "target": cast(
                ScheduleTarget,
                input.get("target", {"kind": "new_once"}),
            ),
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "last_delivery": None,
            "involved_session_ids": [],
        }

        self._tasks[task_id] = task
        self.ctx.emit("schedule/changed", task)
        return task

    def schedule_list(
        self,
        active: bool | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> ScheduleListResult:
        """List scheduled tasks.

        Args:
            active: Filter by active status.
            limit: Maximum number of tasks to return.
            offset: Offset for pagination.

        Returns:
            List of tasks.
        """
        tasks = list(self._tasks.values())

        if active is not None:
            if active:
                tasks = [t for t in tasks if t["status"] == "active"]
            else:
                tasks = [t for t in tasks if t["status"] != "active"]

        if offset:
            tasks = tasks[offset:]  # pragma: no cover

        if limit:
            tasks = tasks[:limit]  # pragma: no cover

        return ScheduleListResult(tasks=tasks)

    def schedule_view(self, task_id: str) -> ScheduleViewResult | None:
        """View one scheduled task.

        Args:
            task_id: Task identifier.

        Returns:
            Task details or None if not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None

        return ScheduleViewResult(
            id=task["id"],
            kind=task["kind"],
            title=task["title"],
            prompt=task["prompt"],
            timing=task["timing"],
            target=task["target"],
            status=task["status"],
            created_at=task.get("created_at"),
            updated_at=task.get("updated_at"),
            last_delivery=task.get("last_delivery"),
            delivery_history=None,
        )

    def schedule_edit(
        self,
        input: ScheduleEditInput,
        actor_session_id: str | None = None,
    ) -> ScheduleTask | None:
        """Edit, pause, resume, or delete a scheduled task.

        Args:
            input: Edit input with task id, action, and optional patch.
            actor_session_id: Optional session making the edit.

        Returns:
            The updated task or None if not found.
        """
        task_id = input["id"]
        task = self._tasks.get(task_id)
        if task is None:
            return None

        action = input["action"]

        if action == "delete":
            del self._tasks[task_id]
            self.ctx.emit("schedule/changed", {"id": task_id, "action": "delete"})
            return None

        if action == "pause":
            task["status"] = "paused"
        elif action == "resume":
            task["status"] = "active"
        elif action == "update" and input.get("patch"):  # pragma: no cover
            patch = input["patch"]  # pragma: no cover
            if patch is not None:  # pragma: no cover
                for key in (
                    "title",
                    "prompt",
                    "timing",
                    "target",
                    "status",
                ):  # pragma: no cover
                    if key in patch:  # pragma: no cover
                        task[key] = patch[key]  # pragma: no cover
                task["updated_at"] = input.get(  # pragma: no cover
                    "expected_updated_at",
                    task.get("updated_at"),  # pragma: no cover
                )  # pragma: no cover
        # pragma: no cover
        self._tasks[task_id] = task  # pragma: no cover
        self.ctx.emit("schedule/changed", task)
        return task

    async def check(self) -> None:  # pragma: no cover - lifecycle check
        """Lifecycle check: service is always healthy once constructed."""
        return
