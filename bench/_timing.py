"""Timing primitives: monotonic measurement with warmup and repeat aggregation."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    """Aggregate of repeated wall-clock samples for one workload."""

    name: str
    unit: str
    iterations: int  # workload operations performed per timed sample
    samples_ms: tuple[float, ...]
    median_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    @property
    def unit_per_second(self) -> float:
        """Throughput in the workload's unit (ops/sec, turns/sec, etc.)."""
        return self.iterations / (self.median_ms / 1000.0)


def from_samples(
    name: str, unit: str, iterations: int, samples_ms: tuple[float, ...]
) -> Timing:
    """Build a Timing from externally gathered samples (e.g. async workloads)."""
    if not samples_ms:
        raise ValueError("samples_ms must not be empty")
    return Timing(
        name=name,
        unit=unit,
        iterations=iterations,
        samples_ms=samples_ms,
        median_ms=statistics.median(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        stdev_ms=statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    )


def measure(
    name: str,
    fn: Callable[[], object],
    *,
    iterations: int,
    unit: str,
    warmup: int = 2,
    repeat: int = 7,
    setup: Callable[[], None] | None = None,
) -> Timing:
    """Time ``fn`` (which performs ``iterations`` operations) ``repeat`` times.

    ``warmup`` untimed passes prime imports and caches; ``setup`` resets state
    between timed samples when the workload accumulates (e.g. an in-memory
    store). Each sample is wall-clock milliseconds for all ``iterations``.
    """
    if iterations < 1 or repeat < 1 or warmup < 0:
        raise ValueError("iterations and repeat must be >= 1; warmup >= 0")
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        if setup is not None:
            setup()
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples.append(elapsed_ms)
    return Timing(
        name=name,
        unit=unit,
        iterations=iterations,
        samples_ms=tuple(samples),
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
    )
