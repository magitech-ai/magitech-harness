"""Tests for the job-control tools (job_list, job_output, job_kill) over ctx.jobs.

Covers the DSH @deepseek-ai/dsh-tool-jobs reference behaviour:
listing, output read, and cancellation of background jobs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.session.events import ToolCallData
from magitech_harness.tools.jobs import _status_line


class _ManualHooks:
    def __init__(self) -> None:
        self.done: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self.cancel_calls: list[str | None] = []

    def cancel(self, reason: str | None = None) -> None:
        self.cancel_calls.append(reason)
        if not self.done.done():
            self.done.set_result({"status": "killed"})


async def _mount_jobs_tools(ctx: Context) -> None:
    await ctx.flush()
    # Attach a job controller so jobs can be started.
    ctx.jobs.attach_controller("test-ctrl")


async def _start_job(ctx: Context, kind: str = "bash", label: str = "test") -> str:
    def _run(_h: Any) -> _ManualHooks:
        return _ManualHooks()

    return ctx.jobs.start({"kind": kind, "label": label, "run": _run})


@pytest.mark.asyncio
async def test_job_list_empty(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    result = await ctx.tools.execute(
        ToolCallData(call_id="j1", name="job_list", arguments={}),
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["jobs"] == []


@pytest.mark.asyncio
async def test_job_list_returns_jobs(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    job_id = await _start_job(ctx)

    result = await ctx.tools.execute(
        ToolCallData(call_id="j2", name="job_list", arguments={}),
        signal=AbortSignal(),
    )
    assert result.is_error is False
    jobs = result.content["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["id"] == job_id
    assert jobs[0]["kind"] == "bash"
    assert jobs[0]["label"] == "test"
    assert jobs[0]["status"] == "running"


@pytest.mark.asyncio
async def test_job_output_reads_text(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    job_id = await _start_job(ctx)

    result = await ctx.tools.execute(
        ToolCallData(call_id="j3", name="job_output", arguments={"job_id": job_id}),
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert "job" in result.content
    assert result.content["job"]["id"] == job_id


@pytest.mark.asyncio
async def test_job_kill(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    job_id = await _start_job(ctx)

    result = await ctx.tools.execute(
        ToolCallData(call_id="j4", name="job_kill", arguments={"job_id": job_id}),
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["outcome"] == "cancellation-requested"
    assert result.content["job"]["id"] == job_id


@pytest.mark.asyncio
async def test_job_kill_unknown_job(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    result = await ctx.tools.execute(
        ToolCallData(
            call_id="j5", name="job_kill", arguments={"job_id": "nonexistent"}
        ),
        signal=AbortSignal(),
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_job_output_unknown_job(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    result = await ctx.tools.execute(
        ToolCallData(
            call_id="j6", name="job_output", arguments={"job_id": "nonexistent"}
        ),
        signal=AbortSignal(),
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_job_output_invalid_args(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    result = await ctx.tools.execute(
        ToolCallData(call_id="j7", name="job_output", arguments={"job_id": ""}),
        signal=AbortSignal(),
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_job_output_invalid_timeout(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    result = await ctx.tools.execute(
        ToolCallData(
            call_id="j7",
            name="job_output",
            arguments={"job_id": "x", "timeout_ms": -5},
        ),
        signal=AbortSignal(),
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_job_output_wait_running_job(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    job_id = await _start_job(ctx)
    # Don't settle the job: it stays running, so wait returns without error.
    result = await ctx.tools.execute(
        ToolCallData(
            call_id="j8",
            name="job_output",
            arguments={"job_id": job_id, "wait": True, "timeout_ms": 100},
        ),
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert "job" in result.content


@pytest.mark.asyncio
async def test_job_kill_empty_job_id(tmp_path: Path) -> None:
    ctx = create_harness(storage_dir=tmp_path)
    await _mount_jobs_tools(ctx)

    result = await ctx.tools.execute(
        ToolCallData(call_id="j9", name="job_kill", arguments={"job_id": ""}),
        signal=AbortSignal(),
    )
    assert result.is_error is True


def test_status_line_with_finished_at() -> None:
    job = {
        "id": "j1",
        "kind": "bash",
        "label": "test",
        "status": "completed",
        "startedAt": 100,
        "finishedAt": 200,
    }
    line = _status_line(job)
    assert "finished at 200ms" in line


def test_status_line_without_finished_at() -> None:
    job = {
        "id": "j1",
        "kind": "bash",
        "label": "test",
        "status": "running",
        "startedAt": 100,
    }
    line = _status_line(job)
    assert "finished at" not in line


@pytest.mark.asyncio
async def test_tool_jobs_default_mount_from_base(tmp_path: Path) -> None:
    """The base bundle mounts tool-jobs; all three tools must be reachable."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    assert ctx.tools.get("job_list") is not None
    assert ctx.tools.get("job_output") is not None
    assert ctx.tools.get("job_kill") is not None
