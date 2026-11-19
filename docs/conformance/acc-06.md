# ACC-06 — cross-tooling round-trip record

ACC-06 is the acceptance criterion that a session log written by one tooling
opens in the other, and vice versa. This page records the live evidence for the
1.0 gate. No reference bytes are committed to this tree; fixtures are read from
the pinned reference checkout at validation time and only counts are recorded.

## What changed historically

The original Phase 2 row was a `not-done` **manual** gate: "a session written
here opens in the other tooling and vice versa", requiring a built install of
the other tooling. That gate was retired when an in-process bidirectional
wire-format test replaced it and marked the row `done`.
The literal built-install run was never automated in CI.

Per the Phase 8 plan, C4 re-runs the gate against recorded fixtures as oracles
and records the result rather than leaving it implicit.

## Live evidence (run 2026-11-18)

Environment: harness at `main` after C0–C3; reference pinned at
`fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159`; Python 3.12 / uv; pnpm 12.21.0 and
node v26.13.1 available (a `packageManager` pin of 11.28.5 is not installed and
`corepack` is absent, so a from-source build of the other tooling is not a
reproducible gate here).

| Check | Command / method | Result |
|---|---|---|
| Harness v4 wire round-trip | `pytest tests/integration/test_cross_tooling_roundtrip.py` | 2 passed — harness-written v4 logs reload identically; external v4 catalog/team logs and projection checkpoints load clean |
| Recorded fixtures replay through the harness | `pytest tests/integration/test_c4_fixture_replay.py` | 4 passed — basic turn, title, compaction checkpoint, approval audit all cold-load and derive the oracle messages |
| Reference fixtures accepted by the harness format layer | Hydrate all `snapshots/*/*/session.v4.jsonl` from the pinned reference and run them through `restore_released_v4_artifact` + `assert_released_v4_relationships` | **252 of 252 accepted, 0 refused** |

The third check is the cross-tooling direction that needs no build: every
recorded reference v4 session, after the same token hydration the reference
replay harness performs, passes the harness's v4 admission and relationship
validation. Together with the first check (harness writes v4 that the format
layer treats as a released-generation log), this covers both directions at the
wire-format level.

## Not covered

A live `harness → reference runtime → harness` run against a built install of
a built-install run is not reproducible in this environment (no matching
pnpm, no corepack, no built wheel). It was already retired in-repo. This record
does not claim it; it records the wire-format and recorded-fixture evidence that
is runnable and green. The fresh parity-map re-audit (C6) treats the reference
as a moving target and is the right place to revisit a built-install run if one
becomes reproducible.

Unresolved questions: none.
