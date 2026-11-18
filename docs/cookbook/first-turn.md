# Run your first turn

## Prerequisites

- Python 3.12+ and `uv`, with the project environment synced (`uv sync`).
- A model provider configured, or the built-in `mock` provider for an offline run.

## Steps

1. Boot a harness with an isolated storage directory so the run does not touch
   another session store:

   ```python
   from magitech_harness import create_harness
   from magitech_harness.llm.message import create_user_message

   ctx = create_harness(storage_dir=".sessions")
   await ctx.flush()
   ```

2. Create an agent through the registry and give it the first user input:

   ```python
   handle = await ctx.agents.create({"agentOptions": {}})
   agent = handle["agent"]
   agent.followup(
       create_user_message(
           content=[{"type": "text", "text": "What is the capital of France?"}],
           source={"kind": "user"},
       )
   )
   ```

3. Wait for the turn to settle and read the derived history:

   ```python
   await agent.when_idle()
   messages = agent.session.derive_messages()
   print(messages[-1]["content"])
   ```

4. Alternatively, use the CLI for an offline smoke test with the mock provider:

   ```sh
   uv run mth --storage .sessions run "What is the capital of France?"
   ```

## Verify

`agent.session.events` contains a `turn/start`, a `user/message`, an
`assistant/message`, and a `turn/end`, and `mth session list` shows the session
id.

**Warning:** always pass an explicit `storage_dir` (or `--storage`) for a new
run. Without it the harness uses a memory-only session store and nothing is
persisted for `mth session` to inspect.
