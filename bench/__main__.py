"""Benchmark runner: ``uv run -m bench``.

Options:
  --repeat N     timed samples per workload (default 7)
  --warmup N     untimed priming passes (default 2)
  --only NAME    run one workload by id (repeatable)
  --list         list workload ids without running
  --json         emit machine-readable results to stdout
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable

from bench._timing import Timing
from bench.projection_fold import projection_fold
from bench.session_replay import session_replay
from bench.token_throughput import token_throughput
from bench.turn_latency import turn_latency

WORKLOADS: dict[str, Callable[..., Timing]] = {
    "turn-latency": turn_latency,
    "token-throughput": token_throughput,
    "session-replay": session_replay,
    "projection-fold": projection_fold,
}


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _render(timing: Timing) -> str:
    rate = timing.unit_per_second
    return (
        f"{timing.name:<40} "
        f"median={timing.median_ms:10.3f} ms  "
        f"min={timing.min_ms:10.3f}  max={timing.max_ms:10.3f}  "
        f"stdev={timing.stdev_ms:8.3f}  "
        f"~{rate:,.0f} {timing.unit}/s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench", description="Harness benchmark suite"
    )
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for name in WORKLOADS:
            print(name)
        return 0

    if args.repeat < 1 or args.warmup < 0:
        print("--repeat must be >= 1 and --warmup >= 0", file=sys.stderr)
        return 2

    selected = args.only or list(WORKLOADS)
    unknown = [n for n in selected if n not in WORKLOADS]
    if unknown:
        print(f"unknown workload(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    timings: list[Timing] = []
    for name in selected:
        timing = WORKLOADS[name](warmup=args.warmup, repeat=args.repeat)
        timings.append(timing)

    if args.json:
        payload = {
            "environment": _environment(),
            "results": [
                {
                    "name": t.name,
                    "unit": t.unit,
                    "iterations_per_sample": t.iterations,
                    "samples_ms": list(t.samples_ms),
                    "median_ms": t.median_ms,
                    "min_ms": t.min_ms,
                    "max_ms": t.max_ms,
                    "stdev_ms": t.stdev_ms,
                    "unit_per_second": t.unit_per_second,
                }
                for t in timings
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        env = _environment()
        print(f"Python {env['python']} | {env['platform']} | {env['machine']}")
        print(f"repeat={args.repeat} warmup={args.warmup}")
        print("-" * 100)
        for t in timings:
            print(_render(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
