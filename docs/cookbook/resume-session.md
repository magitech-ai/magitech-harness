# Resume a session

## Prerequisites

- A session already created with a known id (see [first turn](first-turn.md)).
- The same `storage_dir` that held the original session.

## Steps

1. List sessions to find the id:

   ```sh
   uv run mth --storage .sessions session list
   ```

2. Boot a harness against the same store and resume the session:

   ```python
   from magitech_harness import create_harness
   from magitech_harness.llm.message import create_user_message

   ctx = create_harness(storage_dir=".sessions")
   await ctx.flush()
   handle = await ctx.agents.resume({"resumeSessionId": "session-id-from-list"})
   agent = handle["agent"]
   ```

3. Send the follow-up and wait for the turn:

   ```python
   agent.followup(
       create_user_message(
           content=[{"type": "text", "text": "And what is its population?"}],
           source={"kind": "user"},
       )
   )
   await agent.when_idle()
   ```

4. Or resume from the CLI:

   ```sh
   uv run mth --storage .sessions headless "And what is its population?" \
     --session-id session-id-from-list
   ```

## Verify

`agent.session.derive_messages()` includes the original turns before the new
user input, and a new `turn/start` / `turn/end` pair is appended.

**Warning:** resume restores persisted history only. It does not restore shell
processes, browser sessions, or other external state. A missing, unreadable, or
occupied session fails rather than creating a new one.
