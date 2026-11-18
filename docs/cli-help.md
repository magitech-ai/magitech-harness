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
| `--patch PATH` | Extra patch-list overlay applied after the profile (repeatable) |
| `--model SLUG` | Model slug, e.g. `openrouter/anthropic/claude-3.5-sonnet` |

Component rows resolve in layer order: the shipped bundle, then `--profile`,
then each `--patch` in argv order. A later layer overrides an earlier row with
the same id; a missing or malformed overlay file fails at boot. The overlay
flags precede the subcommand (for example `mth --patch p.yaml headless "task"`).

When no subcommand is given, the launcher starts the interactive REPL. With a
subcommand, it runs that command and exits.

## `mth repl`

Starts the interactive REPL shell. Each input line becomes a user turn; the loop
streams assistant output and tool cards to the terminal. `SIGINT` cancels the
current turn.

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
`dsh sdk-minimal`, and `dsh web` commands. This runtime exposes `repl`,
`headless`, `session`, and `component`; there is no desktop/web client or ACP
command. The verb-first `run` form is not present — `headless` is the single
one-shot prompt command.

Unresolved questions: none.
