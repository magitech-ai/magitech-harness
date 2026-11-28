# 0025. Full-surface parity scope and the divergence gate

Date: 2026-11-28
Status: accepted (user decision, evidence-backed)
Supersedes: the scope carve-outs in [ADR 0022](0022-supersede-zero-builtin-domain-tools-base-bundle-in-harness.md) (Decision 4) and the `OUT OF SCOPE` list in [PARITY.md](../../PARITY.md). The ownership rule and base-bundle placement in ADR 0022 remain in force.

## Context

The port target is the dsh pin recorded in [PARITY.md](../../PARITY.md), with `animas` in place of Cordis. The user's stated goal is a close-to-1:1 Python implementation of that pin across the **whole harness surface** — CLI launcher, headless, SDK, ACP, Web, Desktop, plugin manager, and the model-facing PTC/workflow tools — with Web and Desktop both built on NiceGUI and `nicegui-ai`. Divergence based on this workspace's own testing, prototypes, and experiments starts only after parity is proven complete.

The current parity map was written under a core-first scope. It maps 101 base-bundle rows plus the tool catalog, records 108 DONE, 2 PARTIAL, and 18 OUT OF SCOPE rows, and defines DONE as "reference shape implemented and proven by an executable probe". The OUT OF SCOPE list names exactly the app surfaces now in scope: GUI/web/desktop clients, the headless runner app, the plugin manager remainder, DSH-specific service packages, and the PTC/workflow runtime. Those entries were a sequencing decision, not a product boundary.

Two consequences of the old standard are already visible in the ledger. Rows 52 (`tool-fs`) and 26 (`credentials`) are DONE while their probe notes list unported behavior: output schema and escalation fields, cancellation and diff metadata, a cross-process writer lock, a hot-reload watcher, and comment-preserving writes. The section [Missing capabilities that must be added](../../PARITY.md#missing-capabilities-that-must-be-added-to-magitech-harness-in-scope) still lists four items that are now DONE, contradicting the acceptance summary. Shape-level DONE therefore cannot serve as proof of a 1:1 port.

Repository roles are unchanged: `magitech-harness` owns the base bundle and the host-facing contract; `mth-components` is the official marketplace for approved components outside the default bundle, and every optional integration belongs there.

## Decision

1. **Scope.** Parity covers every dsh surface named in the Context. `OUT OF SCOPE` is reserved for explicit user exclusions: the five non-Notion connectors (Linear, Feishu, DingTalk, WeCom, Tencent Meeting), the Windows ACL backend, Electron-specific packaging, and experimental bundles not named here. Deferred items are recorded as PARTIAL with a named acceptor, never silently dropped.

2. **Ownership.**
   - `magitech-harness`: the base bundle, the host-facing contract, the shared event projection, and provider-neutral seams. The shipped base bundle still matches dsh's.
   - `mth-components`: the official marketplace. Notion is the only ported connector; DeepSeek account sign-in, liveshare, an A2UI renderer, and later optional integrations live here.
   - A new sibling package, `magitech-ui`, owns the Web and Desktop hosts. It exposes `mth-web` and `mth-desktop`, shares one NiceGUI component layer from `nicegui-ai`, depends on the harness, and is never imported by it. Desktop host packaging belongs to this package.

3. **Event contract.** `mth` session events — the durable registry in `magitech_harness/session/events.py` — are the authoritative log; `Model-visible ⟺ Logged` applies to them. AG-UI is the normalized event contract between the harness and both UIs: a derived projection with lossless raw payload for dsh-specific presentation data. Both UIs consume the same projection. A2UI is a post-parity marketplace component, not part of the gate.

4. **Host API and peers.** Dsh's generated TypeScript descriptors, Zod factories, and gateway stream machinery have no equivalent artifact under animas and NiceGUI. The parity target is observable controller behavior (session, workspace, settings, job, terminal, workspace-files), exposed through one explicit, versioned Python API shared by both UIs. The gate is single-owner: every host-facing call carries an explicit principal, there is no implicit fallback, and access is fail-closed. The liveshare role model and its peer-sharing server are post-parity marketplace work.

5. **PTC and workflow.** The model-facing surface — `workflow` and `run_code` tool names, schemas, output envelopes, hook vocabulary, and call limits — is in scope. For the gate, scripts keep JavaScript/TypeScript authorship and run in a sandboxed JS runtime driven from Python, with `agent()` routed to the subagent manager, cancellation to `AbortSignal`, output limits to `spill`, and confinement to `ctx.sandbox`. Python-authored scripts are a post-parity experiment.

6. **Connectors and DeepSeek.** Notion is the only connector ported; it surfaces through the marketplace-backed Plugins page. DeepSeek as a model provider is in the base through the provider-neutral `ctx.llm` seam, and DeepSeek Search/Fetch are opt-in web providers. DeepSeek account sign-in is deferred to the marketplace phase: rows 24 and 25 are PARTIAL with the reason accepted by Malcom. The UI account surface is built against the abstract credential seam in `magitech_harness/credentials/engine.py`, and offline contract fixtures for the pinned account flow are captured during the parity phase.

## Gate

Parity is complete when both conditions hold for every in-scope row:

1. **Coverage.** The row has a named owning repository, a Python implementation location, and a recorded verdict. Deferred rows name an acceptor and a phase.
2. **Behavioral equivalence.** An executable probe compares observable behavior against a fixture captured from the pinned reference: wire frames, event payloads, CLI help and exit codes, tool output shapes, or UI state transitions. A shape-only probe does not qualify for rows added under this ADR.

Divergence opens only when both conditions pass. Rows admitted under the old shape-level standard are re-probed where their notes list unported behavior; `tool-fs` and `credentials` are the first two.

## Consequences

- [PARITY.md](../../PARITY.md) expands from a base-bundle map to a full-package map. Its `OUT OF SCOPE` list becomes a deferred-with-acceptor list, the stale missing-capabilities section is reconciled with the acceptance summary, and verdict counts are recomputed.
- [ADR 0022](0022-supersede-zero-builtin-domain-tools-base-bundle-in-harness.md) keeps its ownership rule; only its core-first scope is superseded. Its body is preserved as history.
- `docs/cross-repo-contracts.md` gains `magitech-ui` and the host-facing contract. The CLI, profile composition, SDK, and ACP surfaces become the critical path ahead of UI work.
- No runtime code changes with this decision. It sets the scope and the proof standard for the work that follows.
