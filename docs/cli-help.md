# CLI help

Source of truth: `build_parser()` in `src/magitech_harness/cli.py`. The launcher
is invoked as `mth` (also available as `mth` via the package entry point). This
page is maintained by hand from the parser; the docs review checks it against
`build_parser()`.

## Global options

| Option | Meaning |
|---|---|
| `--storage DIR` | Directory for JSONL session logs (default `.sessions`) |
| `--profile PATH` | Optional YAML profile overlay for loaded components |
| `--model SLUG` | Model slug, e.g. `openrouter/anthropic/claude-3.5-sonnet` |

When no subcommand is given, the launcher starts the interactive REPL. With a
subcommand, it runs that command and exits.

## `mth repl`

Starts the interactive REPL shell. Each input line becomes a user turn; the loop
streams assistant output and tool cards to the terminal. `SIGINT` cancels the
current turn.

## `mth run PROMPT`

Runs a single prompt through the agent loop and prints the result.

## `mth headless TASK`

Runs one task through the parity agent loop for non-interactive use.

| Option | Meaning |
|---|---|
| `--session-id ID` | Adopt or resume a stored session; an unknown id is an error |
| `--json` | Emit the final outcome as JSON instead of the human-readable message |

## `mth session`

Inspects and manages stored sessions.

- `mth session list` — list all sessions.
- `mth session show SESSION_ID` — show a session's events.

## `mth component`

Inspects components.

- `mth component list` — list installed/loaded components.

**[Python]** The reference CLI has `dsh plugin`, `dsh acp`, `dsh sdk`,
`dsh sdk-minimal`, and `dsh web` commands. This runtime exposes `repl`, `run`,
`headless`, `session`, and `component`; there is no desktop/web client or ACP
command.

Unresolved questions: none.
