# 0003. Cooperative Turn Cancellation via AbortSignal

Date: 2026-11-07
Status: accepted

## Context

Interactive REPL sessions and automated agent executions frequently need to cancel in-flight model streams or long-running tool tasks (e.g. user presses Ctrl+C / SIGINT, or a timeout occurs). Abruptly killing tasks or raising unhandled `asyncio.CancelledError` inside third-party SDKs can corrupt open file handles, leave orphan child processes, and drop partial results already delivered by the model.

## Decision

We introduce an explicit cooperative cancellation mechanism using `AbortSignal`:
1. `AbortSignal` holds an internal cancellation state with callbacks and an `abort(reason)` method.
2. The active turn loop and LLM streaming generator accept an optional `AbortSignal`.
3. The interactive REPL traps `SIGINT` (Ctrl+C), marks the active signal as aborted, and allows the in-flight HTTP stream or tool subprocess to terminate cooperatively.
4. Downstream providers and tool runners inspect `signal.aborted` or call `signal.check()` at iteration boundaries and cleanly close sockets and process pipes.

## Consequences

- The interactive host process and REPL never terminate on Ctrl+C during a turn; instead, control cleanly returns to the prompt.
- Model deltas streamed before interruption are safely recorded in the session log with an interruption marker.
- Providers and tool plugins must periodically check `signal.aborted` during long-running async iterations.
