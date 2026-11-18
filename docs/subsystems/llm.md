# LLM

**Source:** `llm/service.py`, `llm/retry.py`, `llm/retry_policy.py`,
`llm/listing_config.py`, `llm/pi_ai/config.py`, `llm/token_meter.py`.

`ctx.llm` owns provider registration, model routing, streaming, retry, and token
metering. The harness ships the Tier 1 provider substrate; concrete providers
and credentials come from host configuration. See
[adr/0012](../adr/0012-tier-1-provider-substrate-in-harness.md).

## Registry and streaming

Providers register adapters keyed by slug. `list_models` consults a timed cache
(`modelsTimeoutMs` `5000`, `modelsMaxAgeMs` `600000`); provider credentials are
read from the env var named by `apiKeyEnv`, and a missing value is a
`MISSING_CREDENTIAL` error. A stream is an async generator of chunks; `llm/stream`
is a waterfall so middleware can replace it, and `llm/stream-start` /
`llm/stream-end` bracket it. See
[adr/0006](../adr/0006-asynchronous-generator-model-streaming-protocol.md).

## Retry

Retry settings belong under each provider's configuration, not under the
`llm-retry` mount row — supplying `retryPolicy` there is an error. A retry policy
has `mode` (`normal`), `maxRetries` (`5`), and `backoff` with
`initialDelayMs` (`500`), `maxDelayMs` (`10000`), and `jitterRatio` (`0.1`).
Retryable codes cover empty responses, rate limits, server errors, timeouts, and
transport errors. The provider substrate also has `timeoutMs` and
`streamIdleTimeoutMs` (both `300000`), and image budget settings with
`extra="forbid"` validation.

## Token metering

The `token-meter` mount accepts no settings; any key is an error. It records
usage from model responses.

Unresolved questions: none.
