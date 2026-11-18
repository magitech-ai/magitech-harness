# Approve a tool call

## Prerequisites

- A tool that declares `requiresApproval` (a canonically defined tool).
- The default bundled approval policy is `ask`; see [approval](../subsystems/approval.md).

## Steps

1. Wire a presenter on the approval service. The interactive `approval/request`
   waterfall is where a host renders the question and returns a decision:

   ```python
   from magitech_harness import create_harness

   ctx = create_harness(storage_dir=".sessions")
   await ctx.flush()

   async def handler(request):
       # Render the request in your UI; return an ApprovalOutcome string.
       return "allowed-once"  # or "rejected" / "cancelled" / "unavailable"

   ctx.approval.set_handler(handler)
   ```

2. Run the turn. When a tool requires approval, the pre-execute guard asks the
   service, which calls the presenter and resolves the decision.

3. Inspect the audit pair appended to the session:

   ```python
   asked = [e for e in agent.session.events if e.type == "approval/asked"]
   decided = [e for e in agent.session.events if e.type == "approval/decided"]
   ```

4. To disable prompts for a fully trusted run, overlay the profile with
   `policy: never` (see [write profile](write-profile.md)).

## Verify

Every `approval/asked` has a matching `approval/decided` within the same turn,
and a denied call produces an error result without executing the tool body.

**Warning:** decisions are one-shot and fail closed. With no presenter or
handler wired, a request is recorded as `unavailable` and the call is denied.
A child harness is pinned to `never` and never inherits the parent approver.
