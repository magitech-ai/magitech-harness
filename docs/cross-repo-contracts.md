# Cross-repository contracts

The three repositories form one runtime stack. This page owns the boundaries;
each repository keeps its own source of truth for detail.

## Ownership table

| Repo | Owns | Depends on | Contract it provides | Contract it consumes |
|---|---|---|---|---|
| anima | `Context`, `Service`, `Fiber`, event modes, effects/disposers, YAML loader | — | mount / emit / waterfall / effect; profile loading | — |
| magitech-harness | runtime composition, turn loop, sessions, config-layer resolution, native guards | anima | `ctx.sessions`, `ctx.tools`, `ctx.llm`, `ctx.agents`, `ctx.agentLoop`, `ctx.layers`; native guard seams | anima kernel primitives |
| mth-plugins | domain tools, providers, marketplace guards, bridges | anima + harness | `apply(ctx, config)` components via `magitech.components` / `index.json` | harness service seams (`ctx.tools`, `ctx.llm`, `ctx.compaction`) |

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

All three repositories are at `0.3.0`; the harness and plugins declare
`anima>=0.3,<0.4`. Editable path sources are for local development; published
wheels replace them at install time.

## Repository references

- [anima guidance](https://github.com/magitech-ai/anima) and its `AGENTS.md`
- [harness guidance](AGENTS.md) and [architecture](architecture.md)
- [mth-plugins guidance](https://github.com/magitech-ai/mth-plugins) and its `AGENTS.md`

Unresolved questions: none.
