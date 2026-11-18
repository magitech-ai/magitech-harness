# Run headless with JSON output

## Prerequisites

- A configured provider or the built-in `mock` provider.
- A task prompt that does not require interactive approval.

## Steps

1. Run the task through the parity agent loop with `--json`:

   ```sh
   uv run mth --storage .sessions headless "Summarize the repository" --json
   ```

2. Capture the output as newline-delimited JSON if a downstream tool consumes
   it:

   ```sh
   uv run mth headless "Summarize the repository" --json > result.jsonl
   ```

3. To continue a stored session, pass `--session-id`:

   ```sh
   uv run mth headless "Now list the tests" --json --session-id <id>
   ```

## Verify

stdout is a single JSON object describing the final outcome. Human-readable
diagnostics go to stderr, so redirecting stdout does not mix logs with the
result. `mth session list` shows the new session.

**Warning:** `--json` emits the final outcome, not a stream of intermediate
events. An unknown `--session-id` is an error rather than a fresh session.
