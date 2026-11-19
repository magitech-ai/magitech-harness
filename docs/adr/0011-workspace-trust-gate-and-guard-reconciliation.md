# ADR 0011: Workspace Trust Gate and Guard Reconciliation

## Status
Accepted

## Date
2026-11-17

## Context
Phase 4 (P4-7) reconciles the approval, compaction, and truncation guards
and formalizes the workspace trust gate. Agentic tool execution touches the
local filesystem, shell, and network; a workspace must be explicitly trusted
before mutating operations are permitted. Three guards already exist in the
marketplace (`approval-guard`, `compaction-guard`, `truncation-guard`) plus
`workspace-trust-gate`. They must compose without conflicting verdicts.

## Decision
1. **Trust Gate First**: `workspace-trust-gate` is evaluated before any
   mutating tool executes. An untrusted workspace denies all mutating tools
   fail-closed, independent of approval policy.
2. **Layered Guard Order**: `tools/pre-execute` waterfall runs guards in the
   order: trust-gate → approval-guard → (tool execution) → truncation-guard
   (post-execute) → compaction-guard (post-turn). Each guard may deny; denial
   short-circuits subsequent guards.
3. **Audit Events**: Each guard emits its decision via `ctx.emit` on the
   corresponding event (`approval/asked`, `approval/decided`, `repeat-tool-reminder`,
   `compaction/start`, `compaction/end`). Denials are always audited.
4. **Fail-Closed Default**: Absence of a guard or service (e.g. no
   `ctx.approvals`) defaults to deny for mutating tools; safe read tools
   (read_file, grep, etc.) bypass approval but still respect the trust gate.

**Current guard ownership:** approval, repeat-reminder, and timeout guards are now native base-bundle rows; see the [guard-ownership table](../architecture.md#guard-ownership). `truncation-guard`, `compaction-guard`, and `workspace-trust-gate` remain marketplace components.

## Consequences
- **Positive**: A single, well-ordered guard chain prevents policy conflicts
  and ensures untrusted workspaces cannot mutate state.
- **Positive**: Audit events give a complete trail of authorization and
  compaction decisions.
- **Negative**: Adds latency to every tool call (serial guard evaluation).
- **Negative**: Guard ordering must be preserved by all marketplace components;
  a mis-ordered guard could bypass the trust gate.
EOF
echo done