# ADR 0009: Centralized Approval Service and Serialization Mutex

## Status
Accepted

## Date
2026-11-08

## Context
In agentic architectures with concurrent subagents, background tasks, or multi-step tool calls, authorization checks introduce concurrency hazards:
1. **Terminal / Stdin Scrambling**: Multiple concurrent tasks requesting user confirmation simultaneously produce interleaved terminal output and race conditions over `stdin`.
2. **Payload Discrepancies**: If approval decisions are bound only to tool names rather than exact payload contents, an approval granted for a benign command could be exploited to run a destructive command under the same tool name.
3. **Decoupling Mechanics from Policies**: The core engine must serialize I/O and verify payload identity without embedding opinionated approval policies (which belong in marketplace guard plugins).

## Decision
1. **Approval Broker Substrate (`ctx.approvals`)**:
   Implement `ApprovalService` mounted on `ctx.approvals` as a foundational engine service.
2. **Serialization Mutex**:
   Enforce an internal `asyncio.Lock` inside `ApprovalService.request_approval` ensuring all approval requests are processed strictly serially across all concurrent coroutines and subagents.
3. **Canonical SHA-256 Argument Digestion**:
   Argument payloads are deterministically hashed:
   - Formatted as canonical JSON with sorted keys (`sort_keys=True`) and stripped whitespace (`separators=(",", ":")`).
   - Hashed via `sha256` to produce a `sha256:<hex>` digest.
   - Verdicts cached or remembered are strictly bound to identical argument digests.
4. **Waterfall Decoupling**:
   `request_approval` routes through `ctx.waterfall("approval/request", req, default_handler=...)`, allowing policy guard plugins (e.g. `approval-guard`) to audit, auto-approve, or prompt the user.

## Consequences
- **Positive**: Prevents terminal I/O collisions across concurrent subagents and background tasks.
- **Positive**: Strict payload hashing prevents verdict confusion attacks across disparate tool arguments.
- **Positive**: The engine remains completely unopinionated; approval policies are isolated to marketplace plugins.
