# 0006. Asynchronous Generator Model Streaming Protocol

Date: 2026-11-07
Status: accepted

## Context

Streaming token deltas from language models can be implemented via observer callbacks, asynchronous queues, or asynchronous generators. Observers often invert control flow and complicate error propagation, while queue-based approaches introduce synchronization overhead and lifecycle management burdens.

## Decision

We define the model streaming protocol using native Python asynchronous generators:
1. In the `LLMProvider` protocol, streaming is declared as `def stream(self, request: ModelRequest, ...) -> AsyncIterator[ModelChunk]` (returning an asynchronous iterator rather than an awaitable coroutine returning an iterator).
2. The core `LLMService` wraps streaming through the Anima waterfall event `llm/stream`, allowing guardrails to inspect or modify chunk streams on the fly.
3. `AgentLoopService.stream_turn(...)` consumes the generator via `async for chunk in ...`, yielding incremental token deltas, reasoning chunks, and tool lifecycle events directly to the caller.

## Consequences

- Callers consume streams with standard, idiomatic Python `async for` loops.
- Static type checkers (`mypy`) cleanly recognize the protocol method as returning an `AsyncIterator` without ambiguous coroutine wrapping errors.
- Cancellation and backpressure are naturally handled when the consumer breaks or exits the async loop.
