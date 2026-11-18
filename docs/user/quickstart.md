# Quickstart

## Install

The harness requires Python 3.12+ and `uv`. From a checkout:

```sh
uv sync
```

## Configure a provider

Set the credential env var named by your provider configuration, then pass a
model slug with `--model`:

```sh
export OPENROUTER_API_KEY=...
uv run mth --storage .sessions --model "openrouter/anthropic/claude-3.5-sonnet" repl
```

Without a configured provider the runtime has no model to call; the CLI wires a
`mock` provider only for offline smoke tests.

## Run and inspect

```sh
uv run mth --storage .sessions headless "Hello"
uv run mth --storage .sessions session list
uv run mth --storage .sessions session show <id>
```

## Next

See [providers](providers.md) for provider and model routing,
[sessions](sessions.md) for persistence, and the
[cookbook](../cookbook/README.md) for task-focused walkthroughs.

Unresolved questions: none.
