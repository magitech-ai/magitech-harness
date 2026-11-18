# Agent loop

**Source:** `agent/react_loop.py`, `agent/agent_loop.py`, `agent/registry.py`,
`agent/contracts.py`.

The agent loop is the turn/step state machine. `AgentRegistry` owns agent
handles; `AgentLoop` registers them with the loop. An agent accepts user input
through `followup(message)` and becomes idle after `when_idle()`.

## A step

Each step runs a fixed sequence of waterfalls:

| Event | Purpose |
|---|---|
| `agent/pre-step` | claim inbox entries into the request; checkpoint and compaction listeners observe |
| `agent/request` | issue the model request; changing provider or effort here fails loudly |
| `llm/stream` | stream the model response; middleware may replace the stream |
| tool batch | run the `tools/*` pipeline; aggregate `additionalContexts` into the next inbox |
| `agent/assistant-stream`, `agent/turn-stopping` | stream and termination notifications |

A failed attempt is handled through `agent/request-error`; the retry policy lives
under the LLM subsystem. Parallel tool calls are bounded (default `10`, from
`maxParallelToolCalls`).

## Cancellation

Every dispatch carries a cooperative `AbortSignal`. Cancellation aborts a signal
rather than killing a task; tools observe the signal and settle. See
[adr/0003](../adr/0003-cooperative-cancellation-via-abort-signal.md) and
[context](context.md).

## Inbox and additionalContexts

User input, tool `additionalContexts`, and other advisory messages enter through
the inbox. The pre-step claims them as `messages` on the request; the step
appends non-runtime-context entries as `user/message` events, preserving their
`source`. This is how the repeat reminder becomes model-visible without
rewriting a tool result.

Unresolved questions: none.
