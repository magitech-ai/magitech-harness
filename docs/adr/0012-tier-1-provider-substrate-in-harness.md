# 0012. Tier 1 Provider Substrate in Harness

Date: 2026-11-17
Status: accepted; Tier 1 precedent **broadened** by
[ADR 0022](0022-supersede-zero-builtin-domain-tools-base-bundle-in-harness.md)
(2026-11-25). This body stays as history: its `llm-pi-ai` placement is unchanged,
but the "narrow exception" framing no longer bounds ownership — every dsh
base-bundle tool row ships here.

## Context

ADR 0002 keeps the harness free of built-in model providers: all providers reside in the `mth-components` marketplace so the engine stays decoupled and every provider is independently versioned. The parity reference, however, ships its multi-provider adapter (`llm-pi-ai`: catalog integration, wire protocols, profile resolution, credentials per request) inside the main repository, because it is the substrate every composition routes through rather than an optional marketplace extra. The parity port initially followed ADR 0002 (P3-8b landed the adapter in `mth-components` as `pi-ai-provider`); the placement mismatch became concrete once the harness absorbed the adapter's own dependencies (`LLMService`, retry policy, adapter failures, listing surfaces).

## Decision

1. The multi-provider adapter substrate — `magitech_harness.llm.pi_ai` (catalog, protocols, profile config, provider data, component entry `apply`) — ships **in the harness repository**. This is the "Tier 1" provider placement: reference-backed adapter substrate lives in-tree.
2. **Tier 1 is not a default-mounted provider.** No `base.yaml` row mounts it; it stays dormant until a composition mounts it via `ctx.use(magitech_harness.llm.pi_ai.apply, ...)`, a profile row (`name: magitech_harness.llm.pi_ai:apply`), or `create_harness(profile=...)`. ADR 0002's zero-default-providers posture is unchanged in behavior.
3. Marketplace provider plugins (`openrouter-provider`, `pollinations-provider`) remain exclusively in `mth-components`; the `pi-ai-provider` marketplace registration is removed in favor of the in-tree substrate.
4. The harness defines the service protocols and the adapter substrate; everything else (domain tools, guardrails, single-vendor provider plugins) remains marketplace-owned.

## Consequences

- The adapter, its catalog data, and its tests are versioned with the harness, eliminating cross-repo drift between the adapter and the `LLMService` seams it consumes.
- `mth-components` no longer vendors the adapter; compositions referencing the old `pi-ai-provider` marketplace entry must mount `magitech_harness.llm.pi_ai:apply` instead.
- The harness gains an `httpx` dependency (the adapter's HTTP transport).
