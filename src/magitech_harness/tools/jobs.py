"""Background-job tools over ``ctx.jobs``: ``job_list``, ``job_output``, ``job_kill``.

A Tier 1 profile-bundle plugin. Registers the three model-facing job-control
tools over the :class:`magitech_harness.jobs.service.JobRegistry` seam. This
matches the DSH @deepseek-ai/dsh-tool-jobs reference at
packages/jobs/tool-jobs/src/index.ts.
"""

from __future__ import annotations

from typing import Any

from animas import Context

from magitech_harness.tools.schema import define_tool

DEFAULT_WAIT_TIMEOUT_MS = 30_000
MAX_WAIT_TIMEOUT_MS = 600_000


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Project a JobView into the model-facing public job snapshot."""
    return {
        "id": job["id"],
        "kind": job["kind"],
        "label": job["label"],
        "status": job["status"],
        "detail": job.get("detail"),
        "startedAt": job["startedAt"],
        "finishedAt": job.get("finishedAt"),
    }


def _status_line(job: dict[str, Any]) -> str:
    finished = ""
    if job.get("finishedAt") is not None:
        finished = f" (finished at {job['finishedAt']}ms)"
    return f"{job['id']} [{job['kind']}] {job['status']}{finished} — {job['label']}"


def apply(
    ctx: Context,
    *,
    wait_timeout_ms: int = DEFAULT_WAIT_TIMEOUT_MS,
    max_wait_timeout_ms: int = MAX_WAIT_TIMEOUT_MS,
    **kwargs: Any,
) -> None:
    """Mount job_list, job_output, job_kill on ctx.tools."""
    jobs = ctx.jobs

    async def _handle_job_list(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        caller = exec_view.get("agent")
        job_list = jobs.list(caller)
        return {"jobs": [_public_job(j) for j in job_list]}

    async def _handle_job_output(
        args: dict[str, Any], exec_view: Any
    ) -> dict[str, Any]:
        job_id = args.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")

        caller = exec_view.get("agent")
        wait = args.get("wait", False)
        timeout_ms = args.get("timeout_ms")
        if timeout_ms is not None:
            if not isinstance(timeout_ms, (int, float)) or timeout_ms <= 0:
                raise ValueError(
                    "invalid timeout_ms: expected a positive number of milliseconds"
                )
            timeout_ms = min(timeout_ms, max_wait_timeout_ms)
        else:
            timeout_ms = wait_timeout_ms

        if wait:
            signal = exec_view.get("signal")
            await jobs.wait(job_id, timeout_ms, caller, signal)

        read = jobs.read(job_id, caller)
        chunks = read.get("chunks", [])
        text = "".join(chunk.get("text", "") for chunk in chunks)
        job = jobs.get(job_id, caller)
        return {
            "text": text,
            "job": _public_job(job) if job else {"id": job_id, "status": "unknown"},
        }

    async def _handle_job_kill(args: dict[str, Any], exec_view: Any) -> dict[str, Any]:
        job_id = args.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")

        caller = exec_view.get("agent")
        reason = args.get("reason")
        result = jobs.kill(job_id, caller, reason)
        job = jobs.get(job_id, caller)
        return {
            "outcome": "cancellation-requested" if result == "requested" else result,
            "job": _public_job(job) if job else {"id": job_id, "status": "unknown"},
        }

    ctx.tools.register(
        define_tool(
            name="job_list",
            description="List your background jobs (running and finished) with their ids, kinds, and statuses.",
            parameters={},
            output={
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "required": True},
                            "kind": {"type": "string", "required": True},
                            "label": {"type": "string", "required": True},
                            "status": {
                                "type": "string",
                                "required": True,
                                "enum": [
                                    "running",
                                    "stopping",
                                    "completed",
                                    "killed",
                                    "failed",
                                ],
                            },
                            "detail": {"type": "string"},
                            "startedAt": {"type": "integer", "required": True},
                            "finishedAt": {"type": "integer"},
                        },
                    },
                },
                "render": (
                    lambda _args, value: [
                        {
                            "type": "text",
                            "text": (
                                "(no background jobs)"
                                if not value
                                else "\n".join(
                                    f"{j['id']} [{j['kind']}] {j['status']} — {j['label']}"
                                    for j in value
                                )
                            ),
                        }
                    ]
                ),
            },
            execute=_handle_job_list,
        )
    )

    ctx.tools.register(
        define_tool(
            name="job_output",
            description=(
                "Read a background job: output since the previous read for "
                "stream jobs, or the result of a finished final-output job."
            ),
            parameters={
                "job_id": {
                    "type": "string",
                    "required": True,
                    "description": "Job id returned by the tool that started the background work.",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Block until the job finishes or the timeout expires; a timed-out wait leaves the job running. Defaults to false.",
                },
                "timeout_ms": {
                    "type": "number",
                    "description": "Max wait in milliseconds with wait: true. Defaults to and is capped by configuration.",
                },
            },
            output={
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string", "required": True},
                        "job": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": True,
                            "properties": {
                                "id": {"type": "string", "required": True},
                                "kind": {"type": "string", "required": True},
                                "label": {"type": "string", "required": True},
                                "status": {
                                    "type": "string",
                                    "required": True,
                                    "enum": [
                                        "running",
                                        "stopping",
                                        "completed",
                                        "killed",
                                        "failed",
                                    ],
                                },
                                "detail": {"type": "string"},
                                "startedAt": {"type": "integer", "required": True},
                                "finishedAt": {"type": "integer"},
                            },
                        },
                    },
                },
                "render": (
                    lambda _args, value: [
                        {
                            "type": "text",
                            "text": (
                                f"{value.get('text', '') or '(no new output)'}\n"
                                f"{_status_line(value.get('job', {}))}"
                            ),
                        }
                    ]
                ),
            },
            execute=_handle_job_output,
        )
    )

    ctx.tools.register(
        define_tool(
            name="job_kill",
            description="Request cancellation of a running background job.",
            parameters={
                "job_id": {
                    "type": "string",
                    "required": True,
                    "description": "Job id returned by the tool that started the background work.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional short reason, recorded in the log and forwarded to the job.",
                },
            },
            output={
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "required": True,
                            "enum": ["cancellation-requested", "already-finished"],
                        },
                        "job": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": True,
                            "properties": {
                                "id": {"type": "string", "required": True},
                                "kind": {"type": "string", "required": True},
                                "label": {"type": "string", "required": True},
                                "status": {
                                    "type": "string",
                                    "required": True,
                                    "enum": [
                                        "running",
                                        "stopping",
                                        "completed",
                                        "killed",
                                        "failed",
                                    ],
                                },
                                "detail": {"type": "string"},
                                "startedAt": {"type": "integer", "required": True},
                                "finishedAt": {"type": "integer"},
                            },
                        },
                    },
                },
                "render": (
                    lambda _args, value: [
                        {
                            "type": "text",
                            "text": (
                                f"job {value['job']['id']} had already finished {_status_line(value['job'])}"
                                if value["outcome"] == "already-finished"
                                else f"requested cancellation of job {value['job']['id']}"
                            ),
                        }
                    ]
                ),
            },
            execute=_handle_job_kill,
        )
    )


apply.inject = ["tools", "jobs"]  # type: ignore[attr-defined]

__all__ = [
    "DEFAULT_WAIT_TIMEOUT_MS",
    "MAX_WAIT_TIMEOUT_MS",
    "apply",
]
