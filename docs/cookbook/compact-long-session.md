# Compact a long session

## Prerequisites

- A session with enough history that early events can be folded without
  splitting a tool call/result pair.

## Steps

1. Trigger compaction through the session waterfall:

   ```python
   result = await ctx.waterfall(
       "session/compact",
       {"session": agent.session, "status": "requested"},
       default_handler=lambda p: p,
   )
   assert result["compacted"] is True
   ```

2. In the REPL, the equivalent is the `/compact` slash command.

3. Inspect the derived transcript. The early span collapses into one checkpoint
   message framed by `<compacted-summary>` markers; the tail survives:

   ```python
   messages = agent.session.derive_messages()
   ```

4. Continue the turn normally; compaction sends no new user input and the next
   step uses the retained context.

## Verify

`agent.session.events` contains `compaction/start`, `compaction/summary`, a
`user/message` surface `replace`, and `compaction/end`. The derived message list
is shorter and contains the checkpoint framing. Oversized tool results can be
collapsed separately by the registered `tool-result` pruner, which emits the
byte-exact middle-pruned marker. See [compaction](../subsystems/compaction.md).

**Warning:** the `compaction/summary` event alone is log-only. The derived
transcript changes only because of the `user/message` surface `replace` written
alongside it; the system-head message is never shadowed.
