# Benchmark suite

Python-native benchmarks for the harness. Four workloads cover the paths the
1.0 review called out: turn latency, token throughput, session load/replay, and
projection fold. The suite uses only the standard library and the installed
harness — no benchmark dependency, no shipped entry point, no network.

## Running

```sh
uv run -m bench                       # all workloads, repeat=7 warmup=2
uv run -m bench --only turn-latency   # one workload
uv run -m bench --list                # workload ids
uv run -m bench --json                # machine-readable samples + environment
uv run -m bench --repeat 15 --warmup 3
```

The package lives outside `src/`, so it is not in the wheel and is exempt from
the `--cov=magitech_harness --cov-fail-under=100` gate. It never asserts timing
thresholds: performance is not a CI gate here. Numbers are a local reference
point and a regression-detection aid, not a pass/fail contract.

## Workloads

| id | measures | entry point | workload |
|---|---|---|---|
| `turn-latency` | one offline agent turn on the unified driver | `agents.create` → `followup` → `when_idle` | 1 turn/sample, harness booted once outside timing, 1 warmup turn, fresh agent each sample |
| `token-throughput` | token estimation over synthetic messages | `estimate_message` + `estimate_content` | 20,000 estimates/sample, pure functions |
| `session-replay` | cold load + projection of a v4 log | `Session.load` → `derive_messages` | one 120-turn (~245 event) log written once, loaded fresh each sample |
| `projection-fold` | surface fold over a fixed event list | `fold_surface` | same ~245-event list, one full fold/sample, pure |

The session and fold workloads share a deterministic synthetic v4 log built by
`bench/_synthetic.py` and validated through the real format layer
(`restore_released_v4_artifact` + `assert_released_v4_relationships`). Values are
neutral so they remain stable across parity pins.

## Methodology

- Clock: `time.perf_counter` (monotonic, ns resolution); all figures are wall
  milliseconds. Each sample times the full workload; we report median, min,
  max, and sample stdev across `repeat` samples after `warmup` untimed passes.
- Isolation: turn latency boots once and excludes startup; session replay
  writes the log once and times only the cold load; token and fold workloads
  are pure with no I/O. Temporary roots use `tempfile.TemporaryDirectory`.
- No environment variable overrides performance budgets (matching the
  reference bench governance); sample count and workload size are CLI flags,
  not thresholds.
- Single-process, single-machine. No scaling model or machine-profile
  normalization is applied — compare numbers only within the same machine and
  environment.

## Recorded baseline

Reference medians from the baseline run below. These are this-machine reference
values (not CI budgets); see "Environment". Regressions are judged by comparing
a new `--json` run against these within the same environment.

| workload | median | unit | ~rate |
|---|---|---|---|
| `turn-latency/one-turn` | 3.145 | ms/turn | ~318 turns/s |
| `token-throughput/estimate-message` | 33.234 | ms / 20k estimates | ~602k estimates/s |
| `session-replay/cold-load` | 10.838 | ms/session (~245 events) | ~92 sessions/s |
| `projection-fold/surface` | 0.442 | ms/fold (~245 events) | ~2,261 folds/s |

### Environment

- Command: `uv run -m bench` (repeat=7, warmup=2)
- Python 3.14.9, Linux (x86_64, 8 logical CPUs), uv-managed virtualenv
- Commit: the C5 benchmark-suite merge on `main`

The baseline was captured on the machine above. Because the suite applies no
machine-profile scaling, record a fresh baseline when moving to different
hardware or a different Python/runtime before using these numbers as a
regression reference.

Unresolved questions: none.
