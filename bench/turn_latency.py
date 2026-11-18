"""Workload: end-to-end latency of one offline agent turn on the unified driver."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from bench._timing import Timing, from_samples
from magitech_harness import check_bundle_entries, create_harness
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.service import MockLlmAdapter, MockLLMProvider


class _MockProviderComponent:
    """Register the offline mock provider and adapter for the unified driver."""

    inject: ClassVar[list[str]] = ["llm"]

    def __init__(self, ctx: Any) -> None:
        ctx.llm.register_provider("mock", MockLLMProvider())
        ctx.llm.register_adapter(["mock"], MockLlmAdapter())


async def _one_turn(ctx: Any) -> None:
    handle = await ctx.agents.create(
        {"agentOptions": {"provider": "mock", "model": "mock"}}
    )
    agent = handle["agent"]
    agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Give a short greeting."}],
            source={"kind": "user"},
        )
    )
    await agent.when_idle()
    dispose = handle.get("dispose")
    if dispose is not None:
        result = dispose()
        if hasattr(result, "__await__"):
            await result


@dataclass(frozen=True)
class _Latency:
    samples_ms: tuple[float, ...]


async def _run(warmup: int, repeat: int) -> _Latency:
    # Boot once, outside all samples: turn latency excludes harness startup.
    ctx = create_harness(storage_dir=None)  # memory-only sessions
    ctx.use(_MockProviderComponent)
    await ctx.flush()
    check_bundle_entries(ctx)
    try:
        for _ in range(warmup):
            await _one_turn(ctx)
        samples: list[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            await _one_turn(ctx)
            samples.append((time.perf_counter() - start) * 1000.0)
        return _Latency(tuple(samples))
    finally:
        await ctx.flush()


def turn_latency(*, warmup: int, repeat: int) -> Timing:
    """One full turn per sample on a booted harness; boot cost is excluded."""
    if repeat < 1 or warmup < 0:
        raise ValueError("repeat must be >= 1; warmup >= 0")
    samples = asyncio.run(_run(warmup, repeat)).samples_ms
    return from_samples("turn-latency/one-turn", "turns", 1, samples)
