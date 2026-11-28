# 0026. Model-authored reason on every tool call

Date: 2026-11-28
Status: accepted (user decision); effective only after the parity gate in [ADR 0025](0025-full-surface-parity-and-the-divergence-gate.md) passes
Supersedes: none

## Context

The user identified a stand-out difference in the intended Magitech agent: for
every tool call, the model provides a one-sentence reason or phrase explaining
why it wants to call that tool. dsh's tool schemas do not require such a field,
so adding it during the parity phase would make the model-facing surface
diverge from the pinned reference before parity is proven.

The reason is a concise justification for the call, not a chain-of-thought
trace. It belongs with the tool call itself, so it must be durable and visible
to every surface that renders tool activity.

## Decision

1. Every tool definition — base bundle and marketplace — carries a required
   `reason` string: one sentence or phrase authored by the model at call time.
2. The reason is committed with the `tool/call` session event. The
   Model-visible ⟺ Logged invariant applies to it exactly as it does to tool
   arguments.
3. The tool pipeline admits a call only when `reason` is present and non-blank;
   a missing or blank reason fails with a typed error before the tool executes.
4. System-prompt scaffolding instructs the model to supply the reason, and the
   REPL, headless NDJSON projection, and Web/Desktop tool cards render it.
5. No tool schema changes with this decision while the ADR 0025 parity gate is
   open. The field ships in the divergence phase after every in-scope row is
   DONE.

## Consequences

- Reference-shaped golden fixtures captured during the parity phase describe
   the pre-divergence contract; post-divergence fixtures add the `reason` field
   and the rejection path for a missing one.
- Tool schemas, the pipeline guard, the session event payload, and all UI/CLI
   projections need coordinated changes and executable probes.
- Marketplace components in `mth-components` adopt the same required field;
   components that predate it must be updated before mounting.
- The tool-call reason is an intentional divergence from dsh and is recorded
   here so it is never mistaken for a parity regression.
