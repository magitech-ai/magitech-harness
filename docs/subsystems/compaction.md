# Compaction

**Source:** `compaction/service.py`, `mth-components/src/mth_components/guards/compaction/`
(`component.py`, `compaction.py`, `pruner.py`).

Compaction folds historical session events before a balanced cut point into a
checkpoint summary and exposes a separate tool-result pruner. The guard and
pruner components are marketplace (`mth-components`) modules; the `ctx.compaction`
service they use is native and mounted in the base bundle. See
[Guard ownership](../architecture.md#guard-ownership).

## Cut point selection

`find_safe_cut_index` picks the largest index near the target fraction whose
boundary is a [`Tool Pairing`](../../CONTEXT.md) — it never splits a `tool/call`
from its `tool/result`. The system message at node 0 is never shadowed. See
[adr/0001](../adr/0001-event-sourced-sessions-with-compaction-seam.md).

## The checkpoint replace

Compaction writes two things:

1. A log-only `compaction/summary` event with the summary blocks, shadowed
   range, shadowed sequence numbers, and token estimate.
2. The operative `user/message` surface `replace` over the shadowed surface
   nodes. Its content frames the summary between `<compacted-summary>` and
   `</compacted-summary>` markers and carries a `compaction` source. This is
   what actually changes `derive_messages()`: the shadowed messages collapse
   into one checkpoint message while the priced tail survives.

Lifecycle events `compaction/start` and `compaction/end` bracket the operation.
The trigger is the `session/compact` waterfall; the REPL exposes it as `/compact`.

## Tool-result pruner

The pruner collapses the middle of an oversized tool result into a fixed marker,
keeping the head and tail code points. It is registered on
`ctx.compaction` as `tool-result` and emits a `compaction/prune` event.

The marker is byte-exact and reference-parity:

```text

[... tool result middle pruned ...]

```

Defaults are `thresholdChars=8192`, `headChars=4096`, `tailChars=1024`, measured
in Unicode code points. Text is pruned across text blocks; non-text blocks are
preserved. A result at or below the threshold is left untouched.

## Configuration

The compaction guard resolves token budgets from the model's context window
(`resolveCompactSpec`): `thresholdRatio` defaults to `0.8`, `retainRatio` to
`0.16`, and `headroomTokens` to `65_536`. Exact `(provider, model)` overrides
are supported via `modelPolicies`. The context window comes from the adapter's
resolved model info, then the durable `request/context` event, then a configured
or fallback constant. `threshold_tokens = min(contextWindow × thresholdRatio,
messageBudget − headroom)` and `retain_tokens = floor(messageBudget ×
retainRatio)`. The cut keeps a verbatim tail of roughly `retain_tokens` before
falling back to a pairing-safe boundary. The pruner budgets above are code
defaults.

Unresolved questions: none.
