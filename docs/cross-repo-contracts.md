# Cross-repository contracts

The three repositories form one runtime stack. This page owns the boundaries;
each repository keeps its own source of truth for detail.

## Ownership table

| Repo | Owns | Depends on | Contract it provides | Contract it consumes |
|---|---|---|---|---|
| animas | `Context`, `Service`, `Fiber`, event modes, effects/disposers, YAML loader | — | mount / emit / waterfall / effect; profile loading | — |
| magitech-harness | runtime composition, turn loop, sessions, config-layer resolution, native guards, **all dsh base-bundle tools and services** | animas | `ctx.sessions`, `ctx.tools`, `ctx.llm`, `ctx.agents`, `ctx.agentLoop`, `ctx.goals`, `ctx.layers`; base-bundle tool rows; native guard seams | animas kernel primitives |
| mth-components | optional, on-demand, and third-party tools, providers, guards, bridges | animas + harness | `apply(ctx, config)` components via `magitech.components` / `index.json` | harness service seams (`ctx.tools`, `ctx.llm`, `ctx.compaction`) |

Ownership is mechanical, not taste: anything mounted by dsh's
`packages/bundle/base/cordis.patch.yml` belongs to `magitech-harness`; anything
shipped as an optional, on-demand, or third-party bundle belongs to
`mth-components`. [ADR 0022](adr/0022-supersede-zero-builtin-domain-tools-base-bundle-in-harness.md)
records the decision; [PARITY.md](../PARITY.md) carries the per-row evidence.

## What is mounted by default

The shipped row list is [`profiles/base.yaml`](../src/magitech_harness/profiles/base.yaml).
The canonical guard owner is the
[Guard ownership table](architecture.md#guard-ownership). Marketplace components
are opt-in: discover them with `mth component discover` and add the printed row
to a profile.

## Where a change belongs

One concern per repo; source outranks docs; PR order is harness first, plugins
second when both are touched.

## Versioning and install

`animas` is at `2.0.0`; the harness and plugins are at `0.3.0` and declare
`animas>=2,<3`. Editable path sources are for local development; published
wheels replace them at install time.

## Repository references

- [animas guidance](https://github.com/magitech-ai/animas) and its `AGENTS.md`
- [harness guidance](AGENTS.md) and [architecture](architecture.md)
- [mth-components guidance](https://github.com/magitech-ai/mth-components) and its `AGENTS.md`

Unresolved questions: none.
