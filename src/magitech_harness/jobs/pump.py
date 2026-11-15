"""Pull-source polling pump pushing incremental bytes into a job's OutputRing."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from magitech_harness.jobs.types import JobAppendOptions, JobOutputSource


@dataclass(slots=True)
class PumpSink:
    """Callbacks invoked by :func:`start_pump` for each drained source chunk."""

    append: Callable[[str, JobAppendOptions | None], None]
    spill: Callable[[int, str | None], None]


@dataclass(slots=True)
class PumpHandle:
    """Lifecycle handle for an active output pump."""

    done: asyncio.Future[None]
    task: asyncio.Task[None] | None = None


def start_pump(
    sources: Sequence[JobOutputSource],
    sink: PumpSink,
    poll_ms: float,
    stop_signal: Awaitable[object],
) -> PumpHandle:
    """Synchronously drain *sources* once, then poll every *poll_ms* until *stop_signal*."""
    if not math.isfinite(poll_ms) or poll_ms <= 0:
        raise ValueError(
            f"invalid pump pollMs: expected a positive finite number of milliseconds, got {poll_ms!r}"
        )
    cursors = [0 for _ in sources]
    loop = asyncio.get_running_loop()
    done: asyncio.Future[None] = loop.create_future()

    def _drain_once() -> None:
        for idx, source in enumerate(sources):
            chunk = source.read(cursors[idx])
            cursors[idx] = chunk["nextOffset"]
            sink.spill(idx, chunk.get("spillPath"))
            if not chunk["text"]:
                continue
            lossy = bool(chunk.get("lossy"))
            if source.channel is None and not lossy:
                sink.append(chunk["text"], None)
            else:
                opts: JobAppendOptions = {}
                if source.channel is not None:
                    opts["channel"] = source.channel
                if lossy:
                    opts["gapBefore"] = True
                sink.append(chunk["text"], opts)

    # Initial synchronous drain ensures output already present right after spawn lands first.
    _drain_once()

    async def _loop() -> None:
        stop_task = asyncio.ensure_future(stop_signal)
        try:
            while not stop_task.done():
                sleep_task = asyncio.create_task(asyncio.sleep(poll_ms / 1000.0))
                await asyncio.wait(
                    (stop_task, sleep_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not sleep_task.done():
                    sleep_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sleep_task
                _drain_once()
        finally:
            done.set_result(None)

    task = asyncio.create_task(_loop())
    return PumpHandle(done=done, task=task)


__all__ = ["PumpHandle", "PumpSink", "start_pump"]
