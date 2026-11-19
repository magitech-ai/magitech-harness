# Tools

**Source:** `tools/contracts.py`, `tools/schema.py`, `tools/pipeline.py`,
`tools/service.py`, `tools/definition.py`.

The tools subsystem registers model-facing tools and runs them through a scoped
three-phase waterfall. The harness owns no domain tools; all tools come from
`mth-plugins` components. The exhaustive list is in the
[tool catalog](../tool-catalog.md).

## Definition APIs

There are two registration shapes:

- The legacy `Tool` / `define_tool(name, description, parameters, execute)` form
  stores a handler verbatim. All tools currently shipped use this form.
- The canonical `define_tool(spec)` form validates a `ToolDefinition` with an
  output schema and render function, plus optional `timeoutMs`,
  `requiresApproval`, `deferLoading`, presentation hooks, and concurrency flags.

`ToolsService.register()` rejects the reserved name `run_code` and emits
`tools/change`. Because shipped tools use the legacy form, none currently
declares `requiresApproval` or `timeoutMs`; those fields are consumed only when
a tool is defined canonically.

## The scoped pipeline

For each tool call, `ctx.tools.execute` runs:

1. **`tools/pre-execute`** on an `exec_view` carrying the call id, name, frozen
   arguments, signal, `requiresApproval`, and optional agent/tool-set/parent.
   Decisions: `allow` (default), `deny` (requires a non-empty reason → error
   result), `cancel`, or `ask` (resolved by the approval service).
2. **`tools/execute`** around the body. The default handler dispatches the
   tool; a declared `timeoutMs` is enforced by a cooperative scoped-deadline
   guard that produces a structured `TOOL_TIMEOUT` error. A sync body with a
   declared timeout runs in a worker thread so the deadline can fire. The
   result is normalized to `{isError, value, content, meta, additionalContexts,
   concludesTurn}`.
3. **`tools/post-execute`** on `(exec_view, result)`. A decision may replace
   `content` or `value` (not both), supply advisory `additionalContexts`, or
   `block` with a `feedback` list. `additionalContexts` are merged; empty lists
   are dropped.
4. **`tools/result`** is an observe-only scoped emit; observer errors are logged
   and never escape. A durable `tool/result` event is appended with
   `surface_op="append"` and `source_event_seqs=[call_seq]`.

Waterfall listeners are scoped to the agent, so parent listeners are visible to
descendant scopes. Faults in a listener are fail-closed where appropriate.

## additionalContexts

`additionalContexts` is the advisory channel from a guard back to the model:
a list of message mappings. Batch results aggregate them and the agent loop
splices them into the next step's inbox as user-sourced messages. They never
rewrite `tool/result` content. The repeat-tool reminder uses this channel to
deliver a sourced `repeat-tool-reminder` notice at consecutive-call thresholds
`{3, 5, 8}`; it resets on a different tracked call or a new user prompt.

## Mounted guards

The default bundle mounts the approval, timeout, repeat-reminder, and
checkpoint guards as native rows. Ownership of every guard, including the
opt-in marketplace guards, is in [Guard ownership](../architecture.md#guard-ownership).
Behavior: timeout is a scoped-deadline `TOOL_TIMEOUT` enabled per tool from its
`timeoutMs`; the repeat reminder is an advisory nudge out of identical call
loops (`tools/repeat_guard.py`); checkpoint flushes the session before a
top-level body.

## Example: registering a tool

```python
from magitech_harness.tools.definition import define_tool

echo = define_tool(
    name="echo",
    description="Echo the given text",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    execute=lambda args: {"text": args["text"]},
)
ctx.tools.register(echo)
```

Unresolved questions: none.
