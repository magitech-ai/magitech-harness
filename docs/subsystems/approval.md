# Approval

**Source:** `approval/service.py`, `approval/guard.py`.

The approval subsystem decides whether a tool call may proceed and records an
audit pair for every decision. The default bundled policy is `ask`; the guard
is mounted as `approval-guard` in the base bundle.

## Policy

`ctx.approval.policy` is one of `ask` or `never`. Setting the policy appends an
`approval/policy` event. A child harness created by a subagent is pinned to
`never` at creation (delegation scope) so it never inherits the parent's
interactive approver or one-shot grants.

## The ask/decide flow

For a tool call whose `requiresApproval` is true, the pre-execute guard asks the
approval service. The service emits the interactive `approval/request` waterfall
for a host presenter to resolve. On resolution it appends two turn-enclosed
events: `approval/asked` (the question) and `approval/decided` (the outcome).
The decision is one-shot.

Fail-closed behavior: if no presenter or handler is wired, a request is recorded
as `unavailable` and the call is denied. A lost connection or invalid response
also denies. The guard never computes approval from a tool's name alone; the
`read_only_tools` allow-list is a secondary convenience, and path confinement is
enforced in the filesystem tools themselves. See
[filesystem](filesystem.md).

**[Python]** The reference models approval with preset policies and an
escalation ladder. This runtime exposes the `ask`/`never` policy plus the
workspace trust gate (adr/0011); there is no Electron desktop approval UI.

## Trust and workspace gates

Workspace trust is handled by a separate gate in `mth-plugins` (`guards/trust`),
backed by a JSON trust store. Trust and approval are distinct: trust decides
whether a workspace is accepted; approval decides whether a specific call
proceeds.

Unresolved questions: none.
