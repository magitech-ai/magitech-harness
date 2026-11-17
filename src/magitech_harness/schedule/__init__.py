"""Schedule service (ctx.schedule): Host-owned scheduled tasks.

Create RFC 5545 recurring tasks that deliver to existing or new Sessions.
Tasks survive restarts.

Reference: ``packages/schedule/schedule``.
"""

from __future__ import annotations

from magitech_harness.schedule.engine import (
    ScheduleConfig,
    ScheduleCreateInput,
    ScheduleEditInput,
    ScheduleListResult,
    ScheduleRecord,
    ScheduleService,
    ScheduleTarget,
    ScheduleTask,
    ScheduleTiming,
    ScheduleViewResult,
)

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
