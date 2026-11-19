# Subsystems

One reference page per owned subsystem. Each page defines its surface, current
semantics, and a worked example; cross-cutting behavior lives in
[architecture.md](../architecture.md), and exhaustive shipped surfaces live in
the [catalogs](../tool-catalog.md).

These pages are maintained from source. A change that reshapes a documented
service, event, or contract updates its owning page in the same change. The
authoritative declarations are in `src/magitech_harness/` and the `mth-plugins`
package; see each page's **Source** line.

## Kernel and composition

- [configuration](configuration.md) — config-layer resolution, composition overlays, config keys.
- [system-prompt](system-prompt.md) — invariant scaffolding, section registration.
- [agent-loop](agent-loop.md) — turn/step state machine, request and stream waterfalls.

## Session and history

- [sessions](sessions.md) — append-only log, surface types, projections, format chain.
- [compaction](compaction.md) — cut points, checkpoint replace, tool-result pruning.
- [jobs](jobs.md) — background jobs, lifecycle events.
- [context](context.md) — context service, cancellation and signals.

## Execution

- [tools](tools.md) — tool definitions, scoped pipeline, decisions, additionalContexts.
- [approval](approval.md) — policy, ask/decide audit, fail-closed behavior.
- [shell-and-subprocess](shell-and-subprocess.md) — bounded output, spill, process trees.
- [filesystem](filesystem.md) — path confinement, read/write/edit.
- [llm](llm.md) — provider registry, streaming, retry, listing cache, token metering.
- [subagent](subagent.md) — child harnesses, delegation scope, fail-closed policy.

Unresolved questions: none.
