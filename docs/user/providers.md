# Providers

`ctx.llm` routes model requests by slug, for example
`openrouter/anthropic/claude-3.5-sonnet`. A provider adapter is registered under
a slug and streams async-generator chunks through the `llm/stream` waterfall.
See [llm](../subsystems/llm.md) for retry and metering.

## Credentials

Each provider names an environment variable through `apiKeyEnv`. The value is
read at call time; a missing or empty value fails with `MISSING_CREDENTIAL`. Never
commit credentials to a profile or repository.

## Model listing

`list_models` uses a timed cache (`modelsTimeoutMs` `5000`, `modelsMaxAgeMs`
`600000`). A stale or failed listing does not block a direct model slug.

## Retry

Retry settings belong under each provider's configuration, not on the
`llm-retry` mount row. The defaults are `maxRetries` `5`, initial delay `500 ms`,
max delay `10000 ms`, and jitter ratio `0.1`.

Unresolved questions: none.
