# CLI help

Source of truth: `build_parser()` in `src/magitech_harness/cli.py`. The launcher
is invoked as `mth` (also available as `magitech` and `magitech-harness` via the package entry points). This
page is maintained by hand from the parser; the docs review checks it against
`build_parser()`.

## Global options

| Option | Meaning |
|---|---|
| `--version` | Print `mth <version>` and exit |
| `--storage DIR` | Directory for JSONL session logs (default `.sessions`) |
| `--profile NAME\|PATH` | Named profile under `$MTH_HOME/profiles` or a YAML overlay path |
| `--patch PATH` | Extra patch-list overlay applied after the profile (repeatable) |
| `--model SLUG` | Model slug, e.g. `openrouter/anthropic/claude-3.5-sonnet` |
| `--from-default-profile NAME` | Reset the named profile from a shipped template, then boot it |
| `--dump-config` | Print the composed profile tree as JSON and exit without booting |
| `--dump-default-config` | Print only the shipped bundle layers; takes no profile overlay or `--patch` |
| `--dump-config-schema` | Print JSON Schema for profile rows and exit without booting |

Component rows resolve in overlay order: the shipped bundle, then `--profile`,
then each `--patch` in argv order. A later overlay overrides an earlier row with
the same id; a missing or malformed overlay file fails at boot. A bare profile
name resolves to `$MTH_HOME/profiles/<name>/profile.yaml`, initialized from the
shipped `base` or `headless` template on first use. The overlay flags precede
the subcommand (for example `mth --patch p.yaml headless "task"`). See
[configuration](subsystems/configuration.md#bundle-and-profile-composition-overlay-chain)
for the owned description; this is not the four-tier `ctx.layers` resolution.

The three dump flags are mutually exclusive and take no subcommand. They print
a `mth-config-dump/v1` JSON document with one layer per source and the composed
row list; no components mount and no row orchestration runs.

When no subcommand is given, the launcher starts the interactive REPL. With a
subcommand, it runs that command and exits.

## `mth repl`

Starts the interactive REPL shell. Each input line becomes a user turn; the loop
streams assistant output and tool cards to the terminal. `SIGINT` cancels the
current turn.

## `mth headless [TASK]`

Runs one task through the parity agent loop for non-interactive use. An omitted
task or a lone `-` reads the prompt from stdin; a blank prompt fails before a
turn starts.

| Option | Meaning |
|---|---|
| `--session-id ID` | Adopt or resume a stored session; an unknown id is an error |
| `--json` | Project the run as newline-delimited events on stdout instead of the final text |

`--json` writes one `session` event before the task, `status`, `thinking`,
`text`, `tool_call`, and `tool_result` events as the turn commits, and a final
`final` event carrying the lossless answer and durable turn reason. Strings and
keys are byte-bounded; the final answer is not truncated. Provider reasoning
also goes to stderr under an `mth: reasoning:` heading in both output modes.

**[Python]** The brand prefix in diagnostics is `mth:` rather than `dsh:`, and
the working directory in the opening `session` event comes from the Python
process cwd. The event vocabulary, field names, and bounds follow the pinned
reference; see [PARITY.md](../../PARITY.md) row 103.

## `mth session`

Inspects and manages stored sessions.

- `mth session list` — list all sessions.
- `mth session show SESSION_ID` — show a session's events.

## `mth plugin --profile NAME ACTION [COMPONENT]`

Edits the named profile's marketplace component list at
`$MTH_HOME/profiles/<name>/components.yaml`. Actions are `list`, `add`, `remove`,
and `show`.

**[Python]** dsh forwards `plugin` to pnpm inside the profile directory;
mth edits an explicit component-name list for the official `mth-components`
marketplace. This is a stack substitution: there is no pnpm profile dependency
tree in a Python wheel. See [PARITY.md](../../PARITY.md) row 102.

## `mth component`

Inspects components.

- `mth component list` — list installed/loaded components.
- `mth component discover` — list components from installed packages.
- `mth component install NAME` — print the profile row for a component.

**[Python]** `dsh acp`, `dsh sdk`, `dsh sdk-minimal`, and `dsh web` are named
profiles in the reference, not launcher subcommands. Their Python surfaces are
tracked separately in [PARITY.md](../../PARITY.md) Phase 1; Web and Desktop are
owned by the sibling `magitech-ui` package. The verb-first `run` form is not
present — `headless` is the single one-shot prompt command.

Unresolved questions: none.
