# 0023. Subagent tool names and compatibility aliases

Date: 2026-11-25
Status: **superseded** by [ADR 0024](0024-remove-compatibility-aliases.md)
(2026-11-25). This body is preserved as history; the alias decision no longer applies.
Related: [ADR 0022](0022-supersede-zero-builtin-domain-tools-base-bundle-in-harness.md)

## Context

ADR 0022 moved the dsh base-bundle subagent domain from `mth-components` into
`magitech-harness`. The pinned reference (`4b399c39`) registers these
model-facing names: `subagent` at
`packages/subagent/tool-subagent/src/index.ts:209`, and `send_message`,
`interrupt_agent`, `list_agents`, `wait_agent` in
`packages/subagent/tool-subagent-control/src/`.

The pre-parity marketplace exposed `invoke_subagent`, `send_message`, and
`manage_subagents`. Removing the first and third names outright would break
callers that use them — a breaking public API change requiring a stop-and-ask
under the session rules.

## Decision

- The dsh names are canonical: `subagent`, `subagent_fork`, `send_message`,
  `list_agents`, `interrupt_agent`, `wait_agent`, plus `list_subagent_models`
  for model discovery.
- `invoke_subagent` and `manage_subagents` stay registered as **compatibility
  aliases** in the harness. They are additive, documented as aliases in the
  tool catalog, and carry no new capability. Callers migrating to dsh names do
  not need a coordinated cutover.
- The control tools mount once, in their own `tool-subagent-control` row, so the
  two delegation instances cannot double-register them.
- Lineage is logged as `subagent/descriptor` and `subagent/catalog` session
  events on the **caller's** session, preserving Model-visible ⟺ Logged.

## Consequences

- No public tool name is removed; no stop-and-ask decision was needed.
- The harness carries two extra alias definitions that the reference does not
  have. They are labelled Magitech compatibility, not parity.
- A client that checks for exact tool sets sees extra names; one that checks
  for the dsh names sees all of them.
