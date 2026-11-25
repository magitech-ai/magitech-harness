# Subagent

**Source:** `src/magitech_harness/subagent/`, mounted by the base bundle rows
`subagent`, `subagent-spawn-in-process`, `subagent-fork-in-process`,
`tool-subagent`, `tool-subagent-fork` and `tool-subagent-control`.

The subagent domain is Tier 1 substrate under ADR 0022, not a marketplace
component. `ctx.subagents` owns provider registration, the live child tree,
lineage, capacity, name resolution, messaging and interruption.

## Tools

| Tool | Purpose |
|---|---|
| `subagent` | delegate through the `spawn` provider; returns the child name |
| `subagent_fork` | delegate through the `fork` provider, seeding the child from the parent's completed turns |
| `send_message` | deliver a follow-up to an agent in the caller's tree |
| `list_agents` | list the root and reachable descendants with status and capacity |
| `interrupt_agent` | request cooperative cancellation of a descendant |
| `wait_agent` | wait for pending next-step input, with a bounded timeout |
| `list_subagent_models` | model discovery |

## Delegation scope

A child harness gets a fresh context with the full base bundle. The parent's
providers are copied through the public seams; the child is pinned to a
fail-closed approval scope, so it never inherits the parent's interactive
approver or one-shot grants. A branch-worktree child runs under its own
workspace. Abort signals propagate from parent to child. Child descriptors and
the catalog are logged as `subagent/descriptor` and `subagent/catalog` events.

**[Python]** The dsh reference composes child scopes from parent presets and
pins a delegation triad into the child's session log. This port pins the
approval policy and logs lineage; full preset/sandbox delegation events are not
yet emitted.

## Configuration

Rows: `subagent` (`maxActiveSubagents`, `maxDepth`) and the three tool rows.
The manager still accepts code-only `subsessions_dir` and `workspace_root`.

Unresolved questions: none.
