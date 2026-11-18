# Sessions

A session is the durable, append-only record of a run. Pass `--storage DIR` (or
`storage_dir=` in Python) to persist sessions; without it, sessions live only in
memory for that process.

## Inspecting

```sh
uv run mth --storage .sessions session list
uv run mth --storage .sessions session show <id>
```

The log is the source of truth. A small set of surface events folds into the
model-visible message list; the rest are audit detail. See
[sessions](../subsystems/sessions.md).

## Resuming

Record the session id from `session list`, then resume with
`headless --session-id <id>` or `ctx.agents.resume(id)` in Python. Resume restores
persisted history only; it does not restore external processes. See
[resume a session](../cookbook/resume-session.md).

## Maintenance

Compaction folds old history into a checkpoint without losing tool call/result
pairing; invoke it with the `/compact` slash command or the `session/compact`
waterfall. See [compact a long session](../cookbook/compact-long-session.md).

Unresolved questions: none.
