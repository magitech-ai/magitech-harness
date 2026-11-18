# Magitech Harness architecture

This document is an ordered map of the runtime. It names the composition
boundary, the kernel, the turn loop, the capability seams, and the extension
points. It does not define types or per-service detail: see
[subsystems/](subsystems/README.md) for a subsystem's surface, semantics, and
examples, and the [catalogs](tool-catalog.md) for the shipped surface.

Source of truth: the mounted rows in
[`profiles/base.yaml`](../src/magitech_harness/profiles/base.yaml), the
composition in [`__init__.py`](../src/magitech_harness/__init__.py), and the
decision records in [adr/](adr/).

## Anima and the Context

The harness is built on the `anima` micro-kernel. A `Context` is the root
object; services are mounted on it as rows and addressed as attributes such as
`ctx.sessions`, `ctx.tools`, `ctx.llm`, and `ctx.agent_loop`. Anima supplies the
event bus (observers, waterfalls, scoped emit), the effect/disposer registry,
and fiber-local context. Sessions are the durable substrate; the kernel is not a
persistence layer.

**[Python]** The reference runtime is built on Cordis; the Python port builds on
`anima`. The seam vocabulary (mount, emit, waterfall, dispose) is equivalent;
the package name is not. See [adr/0001](adr/0001-event-sourced-sessions-with-compaction-seam.md).

## Profiles and bundles

A profile is a declarative bundle of component rows. `create_harness()`
(`__init__.py`) composes a run in this order:

1. Create a bare `Context`.
2. Compute the interpolation environment, setting or clearing
   `MAGITECH_SESSIONS_DIR` from the caller's `storage_dir`.
3. Install the loader with the shipped base bundle
   [`profiles/base.yaml`](../src/magitech_harness/profiles/base.yaml).
4. Run an optional `prepare(ctx)` hook.
5. Load the base rows; if a `profile` argument was supplied, merge its rows over
   the base rows with `profiles/merge.py`, then merge each `patches` entry in
   order.
6. `flush()` the context so mounted services and components initialize.

Merge semantics: a known row id overlays the base row's keys in place; an
unknown id appends a new row. Duplicate ids, non-string ids, and rows without a
resolvable `name` fail at boot — there is no silent fallback. See
[adr/0005](adr/0005-profile-bootstrapping-and-decoupled-cli.md) and the
[config catalog](config-catalog.md).

**[Python]** The reference uses a CLI `--patch` chain. This runtime resolves
layers as shipped base bundle → `--profile` → each `--patch` in argv order, all
through the same id-keyed row merge.

## Composition of the default bundle

The default bundle mounts, in dependency order:

| Concern | Seam | Subsystem |
|---|---|---|
| Prompt scaffolding | `ctx.system_prompt` | [system-prompt](subsystems/system-prompt.md) |
| Tool catalogue and pipeline | `ctx.tools` | [tools](subsystems/tools.md) |
| Native tool executor | mounted plugin | [tools](subsystems/tools.md) |
| Model streaming and routing | `ctx.llm` | [llm](subsystems/llm.md) |
| Filesystem and observation | `ctx.fs` | [filesystem](subsystems/filesystem.md) |
| Subprocess and shell | mounted executors | [shell-and-subprocess](subsystems/shell-and-subprocess.md) |
| Context service | `ctx.context` | [context](subsystems/context.md) |
| Compaction and pruning | `ctx.compaction` | [compaction](subsystems/compaction.md) |
| Background jobs | `ctx.jobs` | [jobs](subsystems/jobs.md) |
| Working directory | `ctx.workingDirectory` | [sessions](subsystems/sessions.md) |
| Approval policy | `ctx.approval` | [approval](subsystems/approval.md) |
| Layer and component registry | `ctx.layers`, `ctx.components` | [configuration](subsystems/configuration.md) |
| Session log, persistence, projections | `ctx.sessions` | [sessions](subsystems/sessions.md) |
| Retry and token metering | mounted plugins | [llm](subsystems/llm.md) |
| Agent registry and loop | `ctx.agent_loop` | [agent-loop](subsystems/agent-loop.md) |
| Repeat/timeout guards | mounted plugins | [tools](subsystems/tools.md) |

The harness owns no built-in domain tools or providers; those live in
`mth-plugins`. See [adr/0002](adr/0002-zero-builtin-domain-tools-and-providers.md)
and [adr/0012](adr/0012-tier-1-provider-substrate-in-harness.md).

## The turn loop

A turn begins with user input and resolves through reasoning, tool calls, and a
model response. The loop (`agent/react_loop.py`) drives these steps:

1. `agent/pre-step` — per-step waterfall; checkpoint and compaction listeners
   observe it. User-sourced inbox entries become `user/message` events.
2. `agent/request` — waterfall around the model request. Switching provider or
   reasoning effort here fails loudly.
3. `llm/stream` — waterfall around the provider stream; middleware may replace
   the stream. Checkpoint flush happens before the stream body.
4. Tool batches — `ctx.tools.execute` runs the scoped `tools/pre-execute`,
   `tools/execute`, and `tools/post-execute` waterfalls and the observe-only
   `tools/result` emit. Batch `additionalContexts` are spliced into the next
   step's inbox (advisory; tool/result content is never rewritten).
5. `agent/assistant-stream`, `agent/turn-stopping`, `agent/request-error` —
   the remaining agent events.

See [adr/0003](adr/0003-cooperative-cancellation-via-abort-signal.md),
[adr/0006](adr/0006-asynchronous-generator-model-streaming-protocol.md), and
[adr/0007](adr/0007-two-layer-invariant-prompt-scaffolding.md).

## Session log and projection

The session log is an append-only sequence of typed events. A small set of
surface types (`system/message`, `developer/message`, `user/message`,
`assistant/message`, `tool/result`) carry `append` or `replace` operations that
fold into the model-visible message list (`derive_messages`). Everything else
is log-only audit detail. Compaction writes a `user/message` surface `replace`
so the derived transcript actually changes. The stored format is a frozen
adjacent chain v0→v1→v2→v3→v4; corrupt or foreign logs are refused, never
silently repaired. See [sessions](subsystems/sessions.md) and
[compaction](subsystems/compaction.md).

## Capability seams and extension points

Behavior is added at a seam, not by editing the kernel:

- **Mount a component** via a profile row (a service class or an `apply` plugin).
  Registrations are effects: return a disposer and clean up fully.
- **Observe an event** with `ctx.on(event, listener)`; observers cannot veto.
- **Wrap a waterfall** with `ctx.on(event, listener, mode=WATERFALL)`; a listener
  calls `next(...)` to continue and may replace the result or decision.
- **Register a tool** on `ctx.tools`; domain tools live in `mth-plugins`.
- **Register a projection** on the session projection registry.
- **Register a compaction engine or pruner** on `ctx.compaction`.

Fail loudly on misconfiguration: unknown config keys, duplicate registrations,
and unresolvable dependencies raise at boot.

## Where new behavior goes

| Want to… | Add it in… |
|---|---|
| Change what the model sees each step | system-prompt section or projection |
| Gate or annotate a tool call | a `tools/*` waterfall listener (guard) |
| Add a durable fact about a turn | a session event type with an owner |
| Change derived history | a surface op on a surface event type |
| Add a domain capability | an `mth-plugins` tool component |
| Introduce a config knob | the owning service, with a shipped default in the config catalog |

Unresolved questions: none.
