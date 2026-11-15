"""Unit tests for magitech_harness.jobs.pump."""

from __future__ import annotations

import asyncio
import math

import pytest

from magitech_harness.jobs.pump import PumpSink, start_pump
from magitech_harness.jobs.types import JobAppendOptions, JobOutputSourceRead


class _FakeSource:
    def __init__(
        self,
        channel: str | None,
        steps: list[JobOutputSourceRead],
    ) -> None:
        self._channel = channel
        self._steps = steps
        self._calls = 0

    @property
    def channel(self) -> str | None:
        return self._channel

    def read(self, from_byte: int) -> JobOutputSourceRead:
        idx = min(self._calls, len(self._steps) - 1)
        self._calls += 1
        return self._steps[idx]


async def test_pump_rejects_invalid_poll_ms() -> None:
    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    stop.set_result(None)
    sink = PumpSink(append=lambda _t, _o: None, spill=lambda _i, _p: None)
    for bad in (0, -5, math.nan, math.inf):
        with pytest.raises(ValueError, match="invalid pump pollMs"):
            start_pump([], sink, bad, stop)


async def test_pump_drains_synchronously_polls_and_flushes_on_stop() -> None:
    loop = asyncio.get_running_loop()
    stop: asyncio.Future[None] = loop.create_future()

    src_out = _FakeSource(
        "stdout",
        [
            {"text": "first", "nextOffset": 5, "lossy": False, "spillPath": "/tmp/s1"},
            {"text": "", "nextOffset": 5, "lossy": False},
            {"text": "tail", "nextOffset": 9, "lossy": True, "spillPath": "/tmp/s2"},
        ],
    )
    src_plain = _FakeSource(
        None,
        [
            {"text": "plain", "nextOffset": 5, "lossy": False},
            {"text": "gap", "nextOffset": 8, "lossy": True},
        ],
    )

    appends: list[tuple[str, JobAppendOptions | None]] = []
    spills: list[tuple[int, str | None]] = []

    handle = start_pump(
        [src_out, src_plain],
        PumpSink(
            append=lambda text, opts: appends.append((text, opts)),
            spill=lambda idx, path: spills.append((idx, path)),
        ),
        poll_ms=5,
        stop_signal=stop,
    )

    # Synchronous initial drain already landed before awaiting!
    assert appends == [
        ("first", {"channel": "stdout"}),
        ("plain", None),
    ]
    assert spills == [(0, "/tmp/s1"), (1, None)]

    # Wait long enough for at least one periodic sleep tick to fire naturally
    await asyncio.sleep(0.02)
    stop.set_result(None)
    await handle.done

    assert ("tail", {"channel": "stdout", "gapBefore": True}) in appends
    assert ("gap", {"gapBefore": True}) in appends
    # Final stop wakes pump mid-sleep cleanly and withdraws/updates spill path
    assert (0, None) in spills
    assert (0, "/tmp/s2") in spills
