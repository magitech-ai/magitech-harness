# 0005. Profile Bootstrapping and Decoupled CLI Architecture

Date: 2026-11-07
Status: accepted

## Context

Many agent CLI tools tightly couple command-line option parsing, environment configuration, and interactive terminal loops into a monolithic main function. This structure prevents reusing the runtime for alternative interfaces (headless automated pipelines, web gateways, testing runners) and forces all profiles to inherit a single rigid argument set.

## Decision

We decouple application launching into a lightweight profile bootstrapper:
1. The `mth` CLI binary parses only launcher-level configuration (`--profile <name>`, `--patch <file>`).
2. It captures all remaining unparsed arguments into an immutable dictionary on the root Anima context (`ctx.cmdline_args`).
3. It wires fail-loud process shutdown hooks into `ctx.app_exit` and resolves the target profile.
4. The interactive multi-turn REPL is itself implemented as an Anima component, loaded by the profile, which reads `ctx.cmdline_args` to configure model selection, session resumption, and prompt options.

## Consequences

- The launcher binary remains thin and decoupled from specific UI or terminal dependencies.
- Headless, automated, or embedded applications use the exact same launcher substrate without starting a terminal REPL.
- Command-line argument semantics for inner features are owned by the profile and components that consume them.

## Amendment (2026-11-14) — boot shape kept as `main_async`

The Decision above was not implemented as written: no profile
bootstrapper, `ctx.cmdline_args`, `ctx.app_exit`, or REPL-as-component
exists, and the CLI never parsed only launcher-level configuration.
What ships instead is the `main_async` sequence in `cli.py`, kept
deliberately during the all-plugin kernel rework:

1. `create_harness` loads the default bundle rows and settles them.
2. The boot audit (`check_bundle_entries`) fails loud on any row that
   did not activate.
3. A user `--profile` loads after the bundle rows.
4. When no provider is registered, the offline mock provider registers
   as the fallback.
5. Dispatch runs (`run`, `session`, `component`, or the REPL).

The bootstrapper shape in the Decision is set aside, not silently
replaced: this amendment records that the `main_async` sequence is the
accepted boot shape, and the earlier design stays unimplemented until
a later decision revives it. The original Decision text above is kept
unchanged as the record of what was first decided.
