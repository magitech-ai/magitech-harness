"""Tests for the todo_write tool and todos projection unit.

Covers the DSH @deepseek-ai/dsh-tool-todo reference behaviour:
whole-list replacement, status validation, unique content, single vs
parallel in_progress, turn/start clearing, and session event emission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.session.events import EventType, ToolCallData
from magitech_harness.tools.todo import (
    TodoArgsError,
    _apply_todos,
    _validate_todo_state,
    _validate_todos,
)

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_validate_todos_empty_list() -> None:
    assert _validate_todos([], allow_parallel=True) == []


def test_validate_todos_single_item() -> None:
    result = _validate_todos(
        [{"content": "write code", "status": "pending"}], allow_parallel=False
    )
    assert result == [{"content": "write code", "status": "pending"}]


def test_validate_todos_content_must_be_non_empty_string() -> None:
    with pytest.raises(TodoArgsError, match="content must be a non-empty string"):
        _validate_todos([{"content": "", "status": "pending"}], allow_parallel=False)


def test_validate_todos_content_is_trimmed() -> None:
    result = _validate_todos(
        [{"content": "  write code  ", "status": "pending"}], allow_parallel=False
    )
    assert result == [{"content": "write code", "status": "pending"}]


def test_validate_todos_content_must_be_unique() -> None:
    with pytest.raises(TodoArgsError, match="duplicates"):
        _validate_todos(
            [
                {"content": "x", "status": "pending"},
                {"content": "x", "status": "completed"},
            ],
            allow_parallel=False,
        )


def test_validate_todos_status_must_be_valid() -> None:
    with pytest.raises(TodoArgsError, match="status must be one of"):
        _validate_todos([{"content": "x", "status": "unknown"}], allow_parallel=False)


def test_validate_todos_single_in_progress_only() -> None:
    with pytest.raises(TodoArgsError, match="at most one task may be in_progress"):
        _validate_todos(
            [
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ],
            allow_parallel=False,
        )


def test_validate_todos_parallel_allows_multiple_in_progress() -> None:
    result = _validate_todos(
        [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ],
        allow_parallel=True,
    )
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration via create_harness
# ---------------------------------------------------------------------------


async def _create_session(ctx: Context) -> Any:
    session = ctx.sessions.create("sess-todo")
    return session


@pytest.mark.asyncio
async def test_todo_write_replaces_whole_list(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = await _create_session(ctx)

    result = await ctx.tools.execute(
        ToolCallData(
            call_id="t1",
            name="todo_write",
            arguments={"todos": [{"content": "first", "status": "in_progress"}]},
        ),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["todos"] == [{"content": "first", "status": "in_progress"}]
    assert result.content["counts"] == {"pending": 0, "inProgress": 1, "completed": 0}

    # Second call replaces entirely.
    result = await ctx.tools.execute(
        ToolCallData(
            call_id="t2",
            name="todo_write",
            arguments={"todos": [{"content": "second", "status": "completed"}]},
        ),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["todos"] == [{"content": "second", "status": "completed"}]
    assert result.content["counts"] == {"pending": 0, "inProgress": 0, "completed": 1}

    # The session logged two todo/write events.
    todo_events = [e for e in session.events if e.type == "todo/write"]
    assert len(todo_events) == 2
    assert todo_events[0].payload["todos"] == [
        {"content": "first", "status": "in_progress"}
    ]
    assert todo_events[1].payload["todos"] == [
        {"content": "second", "status": "completed"}
    ]


@pytest.mark.asyncio
async def test_validate_todos_non_list_rejected() -> None:
    with pytest.raises(TodoArgsError, match="todos must be an array"):
        _validate_todos("not a list", allow_parallel=False)


def test_validate_todos_non_dict_item_rejected() -> None:
    with pytest.raises(TodoArgsError, match="must be an object"):
        _validate_todos(["not a dict"], allow_parallel=False)


def test_validate_todo_state_valid_list() -> None:
    assert _validate_todo_state([{"content": "x", "status": "pending"}]) == [
        {"content": "x", "status": "pending"}
    ]


def test_validate_todo_state_invalid_type_rejected() -> None:
    with pytest.raises(TypeError, match="todos state must be an array or null"):
        _validate_todo_state(42)


def test_apply_todos_unknown_event_keeps_state() -> None:
    from magitech_harness.session.events import SessionEvent

    state = [{"content": "x", "status": "pending"}]
    event = SessionEvent(
        id="x",
        session_id="s",
        type="user/message",
        timestamp=0.0,
        payload={"role": "user", "content": [{"type": "text", "text": "hi"}]},
    )
    assert _apply_todos(state, event) is state


def test_apply_todos_invalid_todo_write_keeps_state() -> None:
    from magitech_harness.session.events import SessionEvent

    state = [{"content": "x", "status": "pending"}]
    event = SessionEvent(
        id="x",
        session_id="s",
        type="todo/write",
        timestamp=0.0,
        payload={"todos": "not a list"},
    )
    assert _apply_todos(state, event) is state


@pytest.mark.asyncio
async def test_todo_write_without_session_rejected(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    result = await ctx.tools.execute(
        ToolCallData(call_id="t1", name="todo_write", arguments={"todos": []}),
        session=None,
        signal=AbortSignal(),
    )
    assert result.is_error is True


async def test_todo_write_invalid_arguments_rejected(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = await _create_session(ctx)

    result = await ctx.tools.execute(
        ToolCallData(
            call_id="t1",
            name="todo_write",
            arguments={"todos": [{"content": "", "status": "pending"}]},
        ),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_todo_projection_cleared_on_turn_start(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = await _create_session(ctx)

    await ctx.tools.execute(
        ToolCallData(
            call_id="t1",
            name="todo_write",
            arguments={"todos": [{"content": "x", "status": "in_progress"}]},
        ),
        session=session,
        signal=AbortSignal(),
    )

    # Simulate turn/start clearing the projection.
    session.append(EventType.TURN_START, {"turn": 1})

    from magitech_harness.session.projection_registry import SessionProjectionRegistry

    reg: SessionProjectionRegistry = ctx.sessionProjections
    snapshot = reg.snapshot(session, keys=["todos"])
    state = snapshot["values"].get("todos")
    assert state is None


@pytest.mark.asyncio
async def test_todo_write_default_mount_from_base(tmp_path: Path) -> None:
    """The base bundle mounts the tool-todo row; the tool must be reachable."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    from magitech_harness.tools.service import ToolsService

    tools: ToolsService = ctx.tools
    assert tools.get("todo_write") is not None
    assert tools.get("todo_write")["name"] == "todo_write"
