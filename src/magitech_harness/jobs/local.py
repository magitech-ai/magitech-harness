"""Process-local in-memory implementation of :class:`JobRegistry`."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from anima import Context
from anima.core import _access_context

from magitech_harness.agent.cancellation import AbortSignal, deadline, timeout_of
from magitech_harness.jobs.events import JobEventHub, JobLayer
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
    JobOutputRead,
    JobOutputSource,
    JobOutputSourceRead,
    JobOutputSummary,
    JobRead,
    JobRegisteredData,
    JobSettleCause,
    JobSettledData,
    JobSpec,
    JobStatus,
    JobToolCallRef,
    JobView,
    is_terminal_status,
)
from magitech_harness.scope import ScopedLayers, scope_of


def _validate_positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"invalid {key}: expected an integer >= 1, got {value!r}")
    return value


@dataclass(slots=True)
class _ProducerState:
    progress: str | None = None
    job: _TrackedJob | None = None


@dataclass(slots=True)
class _TrackedJob:
    id: str
    tool_call: JobToolCallRef | None
    kind: str
    label: str
    output_limit_bytes: int | None
    owner: Any | None
    cancel: Callable[[str | None], None]
    status: JobStatus
    ring: OutputRing
    model_cursor: int
    result_delivered: bool
    state: _ProducerState
    detail: str | None
    result: Any | None
    started_at: int
    finished_at: int | None
    kill_reason: str | None
    settle_cause: JobSettleCause | None
    settled: asyncio.Future[None]
    wait_resolvers: set[Callable[[], None]] = field(default_factory=set)
    pump: PumpHandle | None = None
    spill_paths: list[str | None] = field(default_factory=list)


class _ScopedJobEvents:
    """``ctx.jobs.events`` view pinned to the accessing registrar Context."""

    def __init__(self, hub: JobEventHub, registrar: Context) -> None:
        self._hub = hub
        self._registrar = registrar

    def subscribe(
        self,
        filter_spec: JobEventFilter | None,
        listener: Callable[[JobEvent], Any],
    ) -> Callable[[], Awaitable[None] | None]:
        return self._hub.subscribe(self._registrar, filter_spec, listener)


class _LocalJobHandle:
    """Producer-side handle passed synchronously into ``spec['run']``."""

    def __init__(
        self,
        job_id: str,
        on_append: Callable[[str, JobAppendOptions | None], None],
        on_progress: Callable[[str], None],
    ) -> None:
        self._id = job_id
        self._on_append = on_append
        self._on_progress = on_progress

    @property
    def id(self) -> str:
        return self._id

    def append(self, text: str, options: JobAppendOptions | None = None) -> None:
        self._on_append(text, options)

    def update_progress(self, line: str) -> None:
        self._on_progress(line)


class _GuardedSource:
    """Pull-source wrapper containing throws into stream termination."""

    def __init__(
        self,
        job_id: str,
        source: JobOutputSource,
        warn: Callable[[str], None],
    ) -> None:
        self._job_id = job_id
        self._source = source
        self._warn = warn
        self._failed = False

    @property
    def channel(self) -> str | None:
        return self._source.channel

    def read(self, from_byte: int) -> JobOutputSourceRead:
        if self._failed:
            return {"text": "", "nextOffset": from_byte, "lossy": False}
        try:
            return self._source.read(from_byte)
        except Exception as exc:  # noqa: BLE001 - contain misbehaving sources
            self._failed = True
            self._warn(
                f"jobs: output source for {self._job_id} failed; "
                f"its stream stops here: {exc}"
            )
            return {"text": "", "nextOffset": from_byte, "lossy": False}


class LocalJobRegistry(JobRegistry):
    """In-memory background job registry with bounded per-job output rings."""

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx)
        raw = config or {}
        self._max_concurrent_jobs_per_owner = _validate_positive_int(
            raw,
            "maxConcurrentJobsPerOwner",
            DEFAULT_MAX_CONCURRENT_JOBS_PER_OWNER,
        )
        self._retain_bytes = _validate_positive_int(
            raw, "retainBytes", DEFAULT_RETAIN_BYTES
        )
        self._settled_retain_bytes = _validate_positive_int(
            raw, "settledRetainBytes", DEFAULT_SETTLED_RETAIN_BYTES
        )
        self._pump_poll_ms = _validate_positive_int(
            raw, "pumpPollMs", DEFAULT_PUMP_POLL_MS
        )
        self._self_ctx = ctx
        self._store: dict[str, _TrackedJob] = {}
        self._layers: ScopedLayers[JobLayer] = ScopedLayers(
            lambda _scope: JobLayer(), lambda: None
        )
        self._hub = JobEventHub(self._layers, lambda msg: ctx.logger.warning(msg))
        self._owner_cleanups: dict[int, Callable[[], Awaitable[None] | None]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        ctx.effect(lambda: self._dispose_all, "jobs teardown")

    def _caller_ctx(self) -> Context:
        return _access_context.get() or self.ctx

    @property
    def events(self) -> JobEvents:
        return _ScopedJobEvents(self._hub, self._caller_ctx())

    def start(self, spec: JobSpec) -> str:
        owner = self._resolve_owner(spec.get("owner"))
        if not self._serves_owner(owner):
            raise RuntimeError(
                "background jobs unavailable: no job controller serves this "
                "agent (load job controls in its composition)"
            )
        kind = spec["kind"]
        label = spec["label"]
        if not isinstance(kind, str) or len(kind) == 0:
            raise ValueError("invalid job kind: expected a non-empty string")
        if not isinstance(label, str) or len(label) == 0:
            raise ValueError("invalid job label: expected a non-empty string")
        output_limit = spec.get("outputLimitBytes")
        if output_limit is not None and (
            isinstance(output_limit, bool)
            or not isinstance(output_limit, int)
            or output_limit <= 0
        ):
            raise ValueError(
                "invalid outputLimitBytes: expected a positive integer, "
                f"got {output_limit!r}"
            )
        if owner is not None:
            self._ensure_owner_cleanup(owner)

        if self._active_job_count(owner) >= self._max_concurrent_jobs_per_owner:
            raise RuntimeError(
                "background job limit reached for this owner "
                f"(limit: {self._max_concurrent_jobs_per_owner}); "
                "use job_kill to stop an unneeded job, wait for it to "
                "finish, then retry"
            )

        while True:
            job_id = f"{kind}-{uuid.uuid4().hex[:16]}"
            if job_id not in self._store:
                break

        raw_tool_call = spec.get("toolCall")
        tool_call: JobToolCallRef | None = (
            cast(JobToolCallRef, dict(raw_tool_call))
            if raw_tool_call is not None
            else None
        )
        ring = OutputRing()
        state = _ProducerState()

        def _append_output(text: str, opts: JobAppendOptions | None = None) -> None:
            self._append_ring(state, ring, text, opts, "producer")

        def _set_progress(line: str) -> None:
            self._update_progress(state, line)

        handle: JobHandle = _LocalJobHandle(job_id, _append_output, _set_progress)
        hooks = spec["run"](handle)

        loop = asyncio.get_running_loop()
        settled: asyncio.Future[None] = loop.create_future()
        outputs = list(spec.get("output") or ())
        job = _TrackedJob(
            id=job_id,
            tool_call=tool_call,
            kind=kind,
            label=label,
            output_limit_bytes=output_limit,
            owner=owner,
            cancel=hooks.cancel,
            status="running",
            ring=ring,
            model_cursor=0,
            result_delivered=False,
            state=state,
            detail=None,
            result=None,
            started_at=int(time.time() * 1000),
            finished_at=None,
            kill_reason=None,
            settle_cause=None,
            settled=settled,
            spill_paths=[None for _ in outputs],
        )
        state.job = job
        self._store[job_id] = job
        self._record_session_event(job)
        self._emit({"type": "registered", "job": self._view(job)}, owner)

        async def _await_producer() -> tuple[JobOutcome, bool]:
            try:
                outcome = await hooks.done
                return outcome, True
            except Exception as exc:  # noqa: BLE001 - contain producer rejections
                self._self_ctx.logger.warning(
                    f"jobs: job {job.id} producer done rejected "
                    f"(producer contract violation): {exc}"
                )
                return {"status": "failed", "detail": str(exc)}, False

        producer_task = loop.create_task(_await_producer())

        if outputs:
            stop_future: asyncio.Future[None] = loop.create_future()

            def _resolve_stop(_done: object) -> None:
                if not stop_future.done():
                    stop_future.set_result(None)

            producer_task.add_done_callback(_resolve_stop)
            settled.add_done_callback(_resolve_stop)

            def _record_spill(idx: int, path: str | None) -> None:
                job.spill_paths[idx] = path

            guarded = [
                _GuardedSource(job.id, src, self._self_ctx.logger.warning)
                for src in outputs
            ]
            job.pump = start_pump(
                guarded,
                PumpSink(
                    append=lambda text, opts: self._append_ring(
                        state, ring, text, opts, "pump"
                    ),
                    spill=_record_spill,
                ),
                self._pump_poll_ms,
                stop_future,
            )

        async def _finalize() -> None:
            outcome, producer_reported = await producer_task
            if job.pump is not None:
                await job.pump.done
            self._settle(
                job,
                outcome,
                job.settle_cause or "producer",
                producer_reported,
            )

        finalize_task = loop.create_task(_finalize())
        self._tasks.add(finalize_task)
        finalize_task.add_done_callback(self._tasks.discard)
        return job_id

    def list(self, caller: str | None = None) -> list[JobView]:
        return [
            self._view(job)
            for job in self._store.values()
            if job.owner is None or job.owner.id == caller
        ]

    def get(self, job_id: str, caller: str | None = None) -> JobView:
        return self._view(self._expect(job_id, caller))

    def read(self, job_id: str, caller: str | None = None) -> JobRead:
        return self._read_job(self._expect(job_id, caller))

    def read_at(
        self, job_id: str, from_offset: int, caller: str | None = None
    ) -> JobOutputRead:
        job = self._expect(job_id, caller)
        if (
            isinstance(from_offset, bool)
            or not isinstance(from_offset, int)
            or from_offset < 0
        ):
            raise ValueError(
                "invalid output read offset: expected a non-negative integer, "
                f"got {from_offset!r}"
            )
        return job.ring.read_from(from_offset)

    def kill(
        self,
        job_id: str,
        caller: str | None = None,
        reason: str | None = None,
    ) -> Literal["requested", "already-finished"]:
        return self._kill_job(self._expect(job_id, caller), reason)

    async def wait(
        self,
        job_id: str,
        timeout_ms: float,
        caller: str | None = None,
        signal: AbortSignal | None = None,
    ) -> JobView:
        return await self._wait_job(self._expect(job_id, caller), timeout_ms, signal)

    def remove(self, job_id: str, caller: str | None = None) -> None:
        job = self._expect(job_id, caller)
        if not is_terminal_status(job.status):
            raise RuntimeError(
                f"job {job_id} is still {job.status}; "
                "kill it and wait for settlement before removing it"
            )
        self._drop([job])

    def attach_controller(self, name: str) -> Callable[[], Awaitable[None] | None]:
        _ = name
        token = object()
        return cast(
            Callable[[], Awaitable[None] | None],
            self._layers.effect(
                self._caller_ctx(),
                lambda layer: layer.controllers.append(token),
                {"label": "jobs.attachController()", "notify": False},
            ),
        )

    def _resolve_owner(self, session_id: str | None) -> Any | None:
        if session_id is None:
            return None
        agents = self._self_ctx.get("agents", None, strict=False)
        if agents is None:
            raise RuntimeError("background job ownership requires the agent registry")
        owner = agents.get(session_id)
        if owner is None:
            raise RuntimeError(
                f'session "{session_id}" has no live agent '
                "(background job owner must be live)"
            )
        return owner

    def _serves_owner(self, owner: Any | None) -> bool:
        if not self._layers.global_layer.controllers.is_empty():
            return True
        scope = None if owner is None else scope_of(owner.ctx)
        return any(
            not layer.controllers.is_empty()
            for layer in self._layers.chain_layers(scope)
        )

    def _active_job_count(self, owner: Any | None) -> int:
        return sum(
            1
            for job in self._store.values()
            if job.owner is owner and job.status in ("running", "stopping")
        )

    def _expect(self, job_id: str, caller: str | None) -> _TrackedJob:
        job = self._store.get(job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id}")
        if job.owner is not None and job.owner.id != caller:
            raise PermissionError(f"job {job_id} belongs to another session")
        return job

    def _view(self, job: _TrackedJob) -> JobView:
        spill_paths = list(dict.fromkeys(p for p in job.spill_paths if p is not None))
        output: JobOutputSummary = {
            "total": job.ring.total,
            "earliest": job.ring.earliest,
        }
        if spill_paths:
            output["spillPaths"] = spill_paths

        view: JobView = {
            "id": job.id,
            "kind": job.kind,
            "label": job.label,
            "status": job.status,
            "startedAt": job.started_at,
            "output": output,
        }
        if job.owner is not None:
            view["owner"] = job.owner.id
        if job.output_limit_bytes is not None:
            view["outputLimitBytes"] = job.output_limit_bytes
        if job.state.progress is not None:
            view["progress"] = job.state.progress
        if job.detail is not None:
            view["detail"] = job.detail
        if job.finished_at is not None:
            view["finishedAt"] = job.finished_at
        return view

    def _emit(self, event: JobEvent, owner: Any | None) -> None:
        self._hub.emit(event, owner)

    def _record_session_event(
        self,
        job: _TrackedJob,
        *,
        finished_at: int | None = None,
        status: JobStatus | None = None,
        cause: JobSettleCause | None = None,
        producer_reported: bool | None = None,
        detail: str | None = None,
    ) -> None:
        owner = job.owner
        if owner is None:
            return
        reg_data: JobRegisteredData = {
            "jobId": job.id,
            "owner": owner.id,
            "kind": job.kind,
            "label": job.label,
            "startedAt": job.started_at,
        }
        if job.tool_call is not None:
            reg_data["toolCall"] = job.tool_call
        try:
            if status is None:
                owner.session.append("job/registered", dict(reg_data))
            else:
                assert finished_at is not None
                assert cause is not None
                assert producer_reported is not None
                settled_data: JobSettledData = {
                    **reg_data,
                    "finishedAt": finished_at,
                    "status": status,
                    "cause": cause,
                    "producerReported": producer_reported,
                }
                if detail is not None:
                    settled_data["detail"] = detail
                owner.session.append("job/settled", dict(settled_data))
        except Exception as exc:  # noqa: BLE001 - recording must not break lifecycle
            self._self_ctx.logger.warning(
                f"jobs: failed to record lifecycle for {job.id}: {exc}"
            )

    def _read_job(self, job: _TrackedJob) -> JobRead:
        read = job.ring.read_from(job.model_cursor)
        job.model_cursor = job.ring.total
        result: Any | None = (
            job.result
            if is_terminal_status(job.status) and not job.result_delivered
            else None
        )
        if result is not None:
            job.result_delivered = True
        if is_terminal_status(job.status):
            job.ring.trim(self._settled_retain_bytes)
        payload: JobRead = {
            "chunks": read["chunks"],
            "lossy": read["lossy"],
            "job": self._view(job),
        }
        if result is not None:
            payload["result"] = result
        return payload

    def _kill_job(
        self, job: _TrackedJob, reason: str | None
    ) -> Literal["requested", "already-finished"]:
        if is_terminal_status(job.status):
            return "already-finished"
        job.cancel(reason)
        job.status = "stopping"
        if reason is not None:
            job.kill_reason = reason
        job.settle_cause = "kill"
        self._emit({"type": "stopping", "job": self._view(job)}, job.owner)
        return "requested"

    async def _wait_job(
        self,
        job: _TrackedJob,
        timeout_ms: float,
        signal: AbortSignal | None,
    ) -> JobView:
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, (int, float))
            or not math.isfinite(timeout_ms)
            or timeout_ms <= 0
        ):
            raise ValueError(
                "invalid wait timeout: expected a positive number of "
                f"milliseconds, got {timeout_ms!r}"
            )
        if not is_terminal_status(job.status):
            if signal is not None and signal.aborted:
                raise RuntimeError("wait aborted")
            loop = asyncio.get_running_loop()
            waiter_done: asyncio.Future[None] = loop.create_future()
            dl = deadline(signal, timeout_ms, TASK_WAIT_TIMEOUT)

            unsub: Callable[[], None] = lambda: None

            def _on_settled() -> None:
                job.wait_resolvers.discard(_on_settled)
                unsub()
                waiter_done.set_result(None)

            def _on_abort() -> None:
                job.wait_resolvers.discard(_on_settled)
                if timeout_of(dl.signal, TASK_WAIT_TIMEOUT) is not None:
                    waiter_done.set_result(None)
                else:
                    waiter_done.set_exception(RuntimeError("wait aborted"))

            job.wait_resolvers.add(_on_settled)
            unsub = dl.signal.on_abort(_on_abort)
            try:
                await waiter_done
            finally:
                job.wait_resolvers.discard(_on_settled)
                unsub()
                dl.dispose()
        return self._view(job)

    def _append_ring(
        self,
        state: _ProducerState,
        ring: OutputRing,
        text: str,
        options: JobAppendOptions | None,
        writer: Literal["producer", "pump"],
    ) -> None:
        job = state.job
        if job is not None and is_terminal_status(job.status):
            if writer == "producer":
                self._self_ctx.logger.warning(
                    f"jobs: append to settled job {job.id} dropped"
                )
            return
        if not ring.append(text, options, self._retain_bytes):
            return
        if job is not None:
            self._emit_output(job)

    def _emit_output(self, job: _TrackedJob) -> None:
        event: JobEvent = {
            "type": "output",
            "id": job.id,
            "total": job.ring.total,
        }
        if job.owner is not None:
            event["owner"] = job.owner.id
        self._emit(event, job.owner)

    def _update_progress(self, state: _ProducerState, line: str) -> None:
        job = state.job
        if job is not None and is_terminal_status(job.status):
            self._self_ctx.logger.warning(
                f"jobs: progress update on settled job {job.id} dropped"
            )
            return
        state.progress = line
        if job is not None:
            self._emit({"type": "progress", "job": self._view(job)}, job.owner)

    def _settle(
        self,
        job: _TrackedJob,
        outcome: JobOutcome,
        cause: JobSettleCause,
        producer_reported: bool,
    ) -> None:
        if is_terminal_status(job.status):
            return
        job.status = outcome["status"]
        outcome_detail = outcome.get("detail")
        if outcome["status"] == "killed" and job.kill_reason is not None:
            job.detail = (
                f"{outcome_detail}; {job.kill_reason}"
                if outcome_detail is not None
                else job.kill_reason
            )
        elif outcome_detail is not None:
            job.detail = outcome_detail
        job.state.progress = None
        job.result = outcome.get("result")
        job.finished_at = int(time.time() * 1000)
        self._record_session_event(
            job,
            finished_at=job.finished_at,
            status=outcome["status"],
            cause=cause,
            producer_reported=producer_reported,
            detail=job.detail,
        )
        job.ring.trim(
            max(self._settled_retain_bytes, job.ring.total - job.model_cursor)
        )
        resolvers = list(job.wait_resolvers)
        job.wait_resolvers.clear()
        for resolve_wait in resolvers:
            resolve_wait()
        job.settled.set_result(None)
        self._emit(
            {
                "type": "settled",
                "job": self._view(job),
                "cause": cause,
                "awaited": len(resolvers) > 0,
            },
            job.owner,
        )
        self._emit_output(job)

    def _ensure_owner_cleanup(self, owner: Any) -> None:
        key = id(owner)
        if key in self._owner_cleanups:
            return

        def _register() -> Callable[[], Awaitable[None]]:
            async def _cleanup() -> None:
                self._owner_cleanups.pop(key, None)
                await self._dispose_owned(owner)

            return _cleanup

        detach = owner.ctx.effect(_register, "jobs.ownerCleanup()")
        self._owner_cleanups[key] = detach

    async def _dispose_owned(self, owner: Any) -> None:
        owned = [job for job in self._store.values() if job.owner is owner]
        self._cancel_for_teardown(owned, "owner disposed")
        await asyncio.gather(*(job.settled for job in owned))
        self._drop(owned)

    def _drop(self, jobs: Sequence[_TrackedJob]) -> None:
        for job in jobs:
            self._store.pop(job.id, None)
            self._emit({"type": "removed", "job": self._view(job)}, job.owner)

    async def _dispose_all(self) -> None:
        all_jobs = list(self._store.values())
        self._cancel_for_teardown(all_jobs, "jobs service disposed")
        await asyncio.gather(*(job.settled for job in all_jobs))
        self._drop(all_jobs)
        cleanups = list(self._owner_cleanups.values())
        self._owner_cleanups.clear()
        results = [cleanup() for cleanup in cleanups]
        awaitables = [r for r in results if isinstance(r, Awaitable)]
        if awaitables:
            await asyncio.gather(*awaitables)

    def _cancel_for_teardown(self, jobs: Sequence[_TrackedJob], reason: str) -> None:
        for job in jobs:
            if is_terminal_status(job.status):
                continue
            job.settle_cause = "teardown"
            try:
                job.cancel(reason)
                job.status = "stopping"
                self._emit({"type": "stopping", "job": self._view(job)}, job.owner)
            except Exception as exc:  # noqa: BLE001 - force-fail on throwing teardown cancel
                detail = f"cancel threw during teardown; work may be orphaned: {exc}"
                self._self_ctx.logger.warning(
                    f"jobs: cancel of {job.id} threw during teardown; "
                    f"job record forced failed and work may be orphaned: {exc}"
                )
                self._settle(
                    job,
                    {"status": "failed", "detail": detail},
                    "teardown",
                    False,
                )


__all__ = ["LocalJobRegistry"]
