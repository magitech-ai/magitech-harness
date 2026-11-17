"""Unit tests for schedule service."""

from __future__ import annotations

from magitech_harness.schedule.engine import (
    ScheduleService,
)


def test_service_creation() -> None:
    """ScheduleService registers itself as ctx.schedule."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)
    assert ctx.get("schedule", None, strict=False) is service


def test_service_duplicate_raises() -> None:
    """Creating ScheduleService twice raises RuntimeError."""
    from anima import Context

    ctx = Context()
    ScheduleService(ctx)
    try:
        ScheduleService(ctx)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_schedule_create() -> None:
    """schedule_create() creates a task."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)

    task = service.schedule_create(
        {
            "name": "test-task",
            "instruction": "Check deployment status",
            "timing": {
                "kind": "daily",
                "time": "09:00:00.000",
                "time_zone": "Asia/Shanghai",
            },
            "target": {"kind": "session", "session_id": "existing-session"},
        }
    )

    assert task["id"] == "test-task"
    assert task["title"] == "test-task"
    assert task["prompt"] == "Check deployment status"
    assert task["status"] == "active"
    assert task["timing"]["kind"] == "daily"
    assert task["target"]["kind"] == "session"


def test_schedule_list() -> None:
    """schedule_list() returns all tasks."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)

    service.schedule_create(
        {
            "name": "task-1",
            "instruction": "First task",
            "timing": {"kind": "at", "scheduled_at": "2026-01-01T09:00:00.000Z"},
            "target": {"kind": "new_once"},
        }
    )
    service.schedule_create(
        {
            "name": "task-2",
            "instruction": "Second task",
            "timing": {"kind": "every", "every_seconds": 3600},
            "target": {"kind": "new"},
        }
    )

    result = service.schedule_list()
    assert len(result["tasks"]) == 2
    assert result["tasks"][0]["id"] == "task-1"
    assert result["tasks"][1]["id"] == "task-2"


def test_schedule_list_filter_active() -> None:
    """schedule_list() with active filter."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)

    service.schedule_create(
        {
            "name": "active-task",
            "instruction": "Active task",
            "timing": {"kind": "at", "scheduled_at": "2026-01-01T09:00:00.000Z"},
            "target": {"kind": "new_once"},
        }
    )
    service.schedule_create(
        {
            "name": "paused-task",
            "instruction": "Paused task",
            "timing": {"kind": "at", "scheduled_at": "2026-01-01T09:00:00.000Z"},
            "target": {"kind": "new_once"},
        }
    )

    # Pause one task
    service.schedule_edit({"id": "paused-task", "action": "pause", "patch": None})

    active = service.schedule_list(active=True)
    assert len(active["tasks"]) == 1
    assert active["tasks"][0]["id"] == "active-task"

    inactive = service.schedule_list(active=False)
    assert len(inactive["tasks"]) == 1
    assert inactive["tasks"][0]["id"] == "paused-task"


def test_schedule_view() -> None:
    """schedule_view() returns task details."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)

    service.schedule_create(
        {
            "name": "view-task",
            "instruction": "View me",
            "timing": {"kind": "at", "scheduled_at": "2026-01-01T09:00:00.000Z"},
            "target": {"kind": "new_once"},
        }
    )

    result = service.schedule_view("view-task")
    assert result is not None
    assert result["id"] == "view-task"
    assert result["title"] == "view-task"
    assert result["prompt"] == "View me"
    assert result["status"] == "active"

    # Non-existent task
    assert service.schedule_view("non-existent") is None


def test_schedule_edit_update() -> None:
    """schedule_edit() update action changes task fields."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)

    service.schedule_create(
        {
            "name": "edit-task",
            "instruction": "Original instruction",
            "timing": {"kind": "at", "scheduled_at": "2026-01-01T09:00:00.000Z"},
            "target": {"kind": "new_once"},
        }
    )

    updated = service.schedule_edit(
        {
            "id": "edit-task",
            "action": "update",
            "patch": {
                "id": "edit-task",
                "kind": "at",
                "title": "Updated name",
                "prompt": "Updated instruction",
                "timing": {"kind": "at", "scheduled_at": "2026-02-01T09:00:00.000Z"},
                "target": {"kind": "new_once"},
                "status": "active",
                "created_at": "2026-01-01T00:00:00.000Z",
                "updated_at": "2026-01-02T00:00:00.000Z",
                "last_delivery": None,
                "involved_session_ids": [],
            },
        }
    )

    assert updated is not None
    assert updated["title"] == "Updated name"
    assert updated["prompt"] == "Updated instruction"


def test_schedule_edit_pause_resume() -> None:
    """schedule_edit() pause and resume actions."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)

    service.schedule_create(
        {
            "name": "pause-resume-task",
            "instruction": "Pause and resume",
            "timing": {"kind": "at", "scheduled_at": "2026-01-01T09:00:00.000Z"},
            "target": {"kind": "new_once"},
        }
    )

    # Pause
    paused = service.schedule_edit(
        {"id": "pause-resume-task", "action": "pause", "patch": None}
    )
    assert paused is not None
    assert paused["status"] == "paused"

    # Resume
    resumed = service.schedule_edit(
        {"id": "pause-resume-task", "action": "resume", "patch": None}
    )
    assert resumed is not None
    assert resumed["status"] == "active"


def test_schedule_edit_delete() -> None:
    """schedule_edit() delete action removes task."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)

    service.schedule_create(
        {
            "name": "delete-task",
            "instruction": "Delete me",
            "timing": {"kind": "at", "scheduled_at": "2026-01-01T09:00:00.000Z"},
            "target": {"kind": "new_once"},
        }
    )

    assert service.schedule_view("delete-task") is not None

    deleted = service.schedule_edit(
        {"id": "delete-task", "action": "delete", "patch": None}
    )
    assert deleted is None
    assert service.schedule_view("delete-task") is None


def test_schedule_edit_nonexistent() -> None:
    """schedule_edit() on non-existent task returns None."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)

    result = service.schedule_edit(
        {"id": "non-existent", "action": "pause", "patch": None}
    )
    assert result is None


def test_schedule_config_default() -> None:
    """ScheduleService with no config uses empty config."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx)
    assert service.cfg == {}


def test_schedule_config_with_values() -> None:
    """OtelService with config uses provided values."""
    from anima import Context

    ctx = Context()
    service = ScheduleService(ctx, {"delivery_history_days": 30})
    assert service.cfg["delivery_history_days"] == 30
