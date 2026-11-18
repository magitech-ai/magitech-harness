"""Workload: token-estimation throughput over synthetic messages (pure, no I/O)."""

from __future__ import annotations

from bench._timing import Timing, measure
from magitech_harness.llm.token_meter import estimate_content, estimate_message

_ITERATIONS = 20_000
_CONTENT = [
    {
        "type": "text",
        "text": "A deterministic sample sentence for token estimation. " * 12,
    }
]
_MESSAGE = {"role": "user", "content": _CONTENT, "id": "m1", "source": {"kind": "user"}}


def token_throughput(*, warmup: int, repeat: int) -> Timing:
    """Time ``_ITERATIONS`` message+content estimates per sample."""

    def run() -> None:
        for _ in range(_ITERATIONS):
            estimate_message(_MESSAGE)
            estimate_content(_CONTENT)

    return measure(
        "token-throughput/estimate-message",
        run,
        iterations=_ITERATIONS,
        unit="estimates",
        warmup=warmup,
        repeat=repeat,
    )
