# Subagent

**Source:** `mth-components/src/mth_components/tools/subagent/`.

The subagent tool spawns child harnesses to handle delegated work. It is a
domain tool from `mth-components`, not part of the harness core.

## Tools

| Tool | Purpose |
|---|---|
| `invoke_subagent` | spawn a child with a role, prompt, optional model, workspace mode (`inherit`/`branch`), tool allow-list, and wait flag |
| `send_message` | run another turn on an existing subagent |
| `manage_subagents` | `list`, `kill`, or `kill_all`; aborts tasks, removes worktrees, closes sessions |

## Delegation scope

A child harness gets a fresh context with the default bundle. The parent's
providers and tool objects are copied into it, but the child is pinned to a
fail-closed approval scope: its approval policy is set to `never` at creation,
so it never inherits the parent's interactive approver or one-shot grants. A
branch-worktree child runs under its own workspace. Abort signals propagate from
parent to child.

**[Python]** The reference composes child scopes from parent presets and pins a
delegation triad (permission preset, sandbox mode, approval `never`) into the
child's session log. This port pins the approval policy; full preset/sandbox
delegation events are not yet emitted.

## Configuration

Component config: `subsessions_dir` and `workspace_root` (code-derived). There
are no base-bundle YAML keys for subagents.

Unresolved questions: none.
