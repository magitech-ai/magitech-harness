# 0024. Remove the subagent compatibility aliases

Date: 2026-11-25
Status: accepted (user decision, evidence-backed)
Supersedes: [ADR 0023](0023-subagent-tool-names-and-compatibility-aliases.md)

## Context

ADR 0023 kept `invoke_subagent` and `manage_subagents` as aliases when the
subagent domain moved into the harness. The user subsequently stated that this
session's goal is parity, not preserving old code, architecture or backward
compatibility. The aliases are not registered by dsh at pin `4b399c39`: its
delegation tool is named `subagent`
(`packages/subagent/tool-subagent/src/index.ts:209`), with the control tools in
`packages/subagent/tool-subagent-control/src/`. Keeping the aliases adds two
model-facing names the reference does not expose and makes the tool catalog
claim parity with extra surface.

## Decision

- Remove the `invoke_subagent` and `manage_subagents` tool registrations and
  their tests. The dsh-named surface is the only surface.
- Keep `list_subagent_models`, which the reference registers when model
  discovery is enabled.
- PARITY tool rows use the dsh names only; no alias row is counted as parity.
- The manager's copy filter targets the dsh tool names, not the retired names.

## Consequences

- Callers using the retired names must migrate to `subagent` and
  `list_agents`/`interrupt_agent`. This is an intentional API break authorised
  by the user.
- ADR 0023 remains as history; its alias decision no longer applies.
