# AGENTS.md — Documentation standard

This file defines document structure, writing rules, and the quality checklist for
this repository's human-facing docs. It is the Python/uv adaptation of the parity
reference's `docs/AGENTS.md`; adjustments are marked **[Python]** with the reason.

## Document structure

Classify every in-scope document as a tutorial or a reference. Tutorials follow an
ordered path to an outcome and introduce only what each step needs. References
define a lookup scope and current behavior without a teaching sequence. Separate
substantial tutorial and reference content; label a section when either is small.

Author in this order: locate the document in the tree; set its permitted detail;
choose tutorial or reference; for a tutorial, order concepts by prerequisite and
difficulty; relocate descendant-owned detail; replace lower-level explanations
with links to their owner.

### Tier taxonomy: one home per fact

| Tier | Job | Does NOT belong there |
|---|---|---|
| Root `AGENTS.md` | Standing orders for an agent in every session, one to three lines each, linking its home | Stories, worked examples, situational procedures |
| `docs/AGENTS.md` (this file) | Documentation rules for this subtree | Runtime contracts, decision rationale |
| `CONTEXT.md` | Project glossary: core entities and the vocabulary docs must use | Architecture maps, per-service detail |
| `docs/adr/` | Architecture decision records: the why and what was given up | Current-state reference, acceptance checklists |
| `docs/architecture.md` | Ordered map: composition, kernel, loop, seams, extension points | Type definitions, per-service detail, decision rationale |
| `docs/subsystems/` | One reference page per owned subsystem: surface, semantics, examples | Behavior narration (→ architecture), package internals |
| `docs/cookbook/` | Step-by-step how-tos with numbered verify steps | Design rationale (→ ADR) |
| `docs/user/` | Product-facing guides for end users | Generated tables, contributor procedures |
| `docs/tool-catalog.md`, `docs/config-catalog.md`, `docs/cli-help.md` | Exhaustive generated-style references for the shipped surface | Hand-edited drift from source; rationale |
| `docs/upgrade-guide/<version>/` | Per-release externally perceptible breaking changes | Unreleased plans |

**[Python]** The reference generates its catalogs from TypeScript source with
`cordis-surface` regions and freshness gates. We do not port that generator; our
catalogs are maintained by hand from source and are checked against the code in
the docs review. Each catalog page states its source of truth and the check used.

## Writing rules

- **Document current state.** History lives in commits, PRs, and ADRs. Use present
  tense for shipped behavior; do not write "implemented!" or "future: …".
- **Use the glossary vocabulary.** When naming a domain concept, use the term in
  `CONTEXT.md`. Do not drift to synonyms the glossary avoids.
- **Paragraphs render correctly soft-wrapped.** The reference keeps one physical
  line per paragraph for diff stability under its doc generator; we do not port
  that generator, so wrapped source paragraphs are acceptable. Code blocks,
  tables, and list structure keep their formatting.
- **Mark Python adjustments explicitly.** Where our behavior genuinely differs
  from the reference (e.g. `read_file` vs `read`, no Electron desktop, wheels not
  pnpm workspaces), say so with the reason in one line and link the owning ADR.
- **Verify code examples.** A fenced `python` example with a public API call must
  match the current signature. Prefer examples lifted from tests.
- **Links are relative Markdown.** Do not hard-code machine paths or commit hashes
  with hex letters; use PR numbers and relative links.
- **Comments state complete contracts, not reasoning transcripts.** Preserve
  behavior, failure, timing, ownership, and exceptions; delete narration and
  test walkthroughs.

## Wordcount and scope budgets

**[Python]** The reference enforces doc budgets via `pnpm run verify-doc-budgets`
with a manifest of per-file ceilings. We do not port that gate. Review governs
scope: a document that exceeds its purpose is a review finding. Keep a subsystem
page to its own subsystem; put cross-cutting behavior in `architecture.md`.

## The slop checklist

Hunt these in any doc; remove them before merge:

- Duplicated rules: search a distinctive phrase; keep one home and link the rest.
- History outside its permitted tier: state current facts and link the owner.
- Implementation-status annotations ("implemented!", "future: …") — status rots.
- Hand-restated catalogs when source is authoritative; cite the source instead.
- Reasoning transcripts: step-by-step implementation narration, proof of obvious
  branches, test walkthroughs, rejected local alternatives.
- Rationale repeated beside sibling methods instead of once at the owning place.
- Paragraph walls carrying several rules and parenthetical asides. Split them.
- Emphasis inflation: bold/CAPS/"critically" everywhere means nothing stands out.
- Spec-speak in ADRs for shipped decisions: "should", acceptance checklists.

## Repository references

- Glossary: [`CONTEXT.md`](../CONTEXT.md)
- Decisions: [`adr/`](adr/)
- Architecture: [`architecture.md`](architecture.md)
- Subsystems: [`subsystems/`](subsystems/README.md)
- Cookbook: [`cookbook/`](cookbook/README.md)
- Catalogs: [`tool-catalog.md`](tool-catalog.md), [`config-catalog.md`](config-catalog.md), [`cli-help.md`](cli-help.md)
- Conformance: [`conformance/acc-06.md`](conformance/acc-06.md)
- Benchmarks: [`../bench/README.md`](../bench/README.md) (outside `docs/`)

Unresolved questions: none.
