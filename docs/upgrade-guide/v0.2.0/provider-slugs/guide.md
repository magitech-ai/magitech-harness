---
kind: upgrade-guide
description: Multi-provider registry and slug routing replace ad-hoc model selection.
---

# Provider slugs

## Change

Model selection now goes through a multi-provider registry with slug routing, for
example `openrouter/anthropic/claude-3.5-sonnet`. Provider configuration moved
under each provider: retry policy, timeouts, image budgets, and the credential
env var (`apiKeyEnv`) are provider-owned. The `llm-retry` mount no longer accepts
a `retryPolicy` key. See [llm](../../../subsystems/llm.md).

The runtime also adopted strict identity naming: code and documentation use
Magitech names only; outside project and brand names do not appear in the
shipped surface.

## Migration

1. Replace any ad-hoc model argument with a provider-qualified slug via `--model`
   or the agent options.
2. Move retry and timeout settings from the `llm-retry` mount into the owning
   provider's configuration. Supplying `retryPolicy` to the mount is now an
   error.
3. Ensure each provider's `apiKeyEnv` names a set environment variable; a
   missing value fails at call time with `MISSING_CREDENTIAL`.
4. Rename any internal identifiers or docs strings that carried an upstream
   project name. Verify with a repository-wide search and a green test suite.
