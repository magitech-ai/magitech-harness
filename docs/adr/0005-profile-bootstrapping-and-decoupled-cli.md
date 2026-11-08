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
4. The interactive multi-turn REPL is itself implemented as an Anima plugin, loaded by the profile, which reads `ctx.cmdline_args` to configure model selection, session resumption, and prompt options.

## Consequences

- The launcher binary remains thin and decoupled from specific UI or terminal dependencies.
- Headless, automated, or embedded applications use the exact same launcher substrate without starting a terminal REPL.
- Command-line argument semantics for inner features are owned by the profile and plugins that consume them.
