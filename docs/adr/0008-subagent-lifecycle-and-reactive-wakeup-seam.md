# 0008. Sub-Agent Lifecycle, Persisted Sub-Sessions, and Reactive Wakeup Seam

Date: 2026-11-07
Status: accepted

## Context

Multi-agent delegation systems frequently exhibit three significant architectural flaws:
1. **Ephemeral Transcripts**: Running sub-agents in-memory discards execution traces upon exit, preventing debugging, trajectory scoring, and session replay.
2. **Polling Waste**: Parent agents polling child processes or waiting in busy loops consume unnecessary CPU cycles and context window token budgets.
3. **Resource Pollution & Collision**: Unconstrained child agents can mutate parent configurations, close shared provider connections, or interleave terminal approval dialogs.

## Decision

We formalize the sub-agent architecture within the harness:
1. **Always-Persisted Sub-Sessions**: Every delegated sub-agent execution persists its entire event log to disk under `subsessions/<subsession_id>.jsonl` using the standard `SessionStore`. Sub-session storage keeps top-level session lists clean while providing an immutable record for trajectory inspection and auditability.
2. **Reactive Wakeup Seam**: When a sub-agent completes its assigned goal or yields for feedback, it transitions to an idle state (retaining its warm session context) and posts a completion event into the parent harness's turn queue (`ctx.agent_loop`). The parent wakes reactively with zero polling.
3. **Shared Substrate with Non-Closing Guard**: Sub-agents share the parent's model provider configurations and inherit cooperative cancellation (`AbortSignal.any([parent_signal, child_signal])`). Terminating or completing a child agent never closes parent connections or shared resources.
4. **Tool Scoping**: Callers specify `allowed_tools: frozenset[str] | None` on the child context to restrict tool access (e.g. read-only researcher sub-agents).
5. **Event Lineage**: All events generated during sub-agent execution carry explicit `subagent_id` and `parent_tool_call_id` attribution metadata.

## Consequences

- Full observability and deterministic replay for all sub-agent executions.
- Parent agents execute concurrently or sleep without polling loops.
- Clean isolation between parent and child lifecycles.
