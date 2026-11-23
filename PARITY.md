# Parity map — 1.0 gate (Phase 8 re-audit)

This is the fresh parity map required by the ratified Phase 8 handoff (§4.2
piece 4, §4.4). It replaces the stale Phase 0a pin note. It is the document that
proves acceptance criterion 1: **every in-scope item is DONE or PARTIAL with an
accepted gap, and nothing in-scope is MISSING.** "In-scope" is the 1.0 milestone
(Phases 0–6); the Phase 7 tail and all §7 exclusions are out of scope.

## Verdict vocabulary

- **DONE** — reference shape implemented and proven by a live test.
- **PARTIAL** — implemented with a gap; the gap reason is recorded and was
  accepted by Malcom in the ratified §8 decisions or a pinned conformance row.
- **DIVERGED** — a ratified decision is not met. DIVERGED in-scope fails the gate
  (same effect as MISSING) until fixed or explicitly accepted by Malcom.
- **MISSING** — in-scope capability absent. Fails the gate.

A reason accepted by an agent is not accepted by Malcom. Every non-DONE row
names who accepted its reason.

## Audit target and pins

- **Reference:** frozen at `fcc0ad46…` (0.2.2-rc.1), the pin the implementation
  was built and conformance-recorded against. Malcom decided (2026-11-18) to
  freeze the 1.0 audit here; rc.3 (`39163dd3…`) drift is listed in the
  post-1.0 appendix, not counted against 1.0.
- **Reference identity:** DeepSeek Harness, built on Cordis and its preprint
  "A Programming Paradigm for Spatiotemporal Composability" (arXiv:2608.25512).
  This Python stack implements that kernel paradigm; the lineage is published
  in `anima/README.md`, attributed in `anima/NOTICE`, and recorded in
  agents-home decision D-025.
- **Python repos:** anima `0.3.0` @ release bump, magitech-harness `0.3.0` @ C6-fix
  merge, mth-plugins `0.3.0` @ release bump merge. All clean on `main`.
- **Suites (100% statement + branch coverage):** anima 1228, harness 2285,
  plugins 402. CI green on all 6 legs per repo at every Phase 8 merge.

## Phase 0 — kernel and configuration

| Item | Verdict | Evidence / accepted reason |
|---|---|---|
| Anima event waterfalls, scoped emit/filter, effect/disposer | DONE | `anima/events.py`, `anima/core.py`; conformance `test_kernel_k6` |
| Parallel dispatch aggregates failures (one → original error, ≥2 → `ExceptionGroup`) | DONE | PIN-01 re-ratified 2026-11-19 at Malcom's decision: anima now uses all-settled semantics with aggregation; conformance probe added in `anima/tests/integration/test_phase1_acceptance.py` |
| `Context.provide` notifies at registration | PARTIAL | PIN-02; accepted by Malcom via ratified conformance map |
| Four-tier config layers + two homes | DONE | `components/layers.py`, `$MAGITECH_HOME` + `$MAGITECH_AGENTS_HOME` (`shell_env`); C6 fix PR #58 |
| Bundle → profile → `--patch` layer order | DONE | `create_harness(patches=)`, `profiles/merge.py`, CLI `--patch`; PR #58 |
| Profile merge, fail-loud misconfiguration | DONE | `test_create_harness_profile.py` |

## Phase 1 — agent loop and LLM

| Item | Verdict | Evidence / accepted reason |
|---|---|---|
| One unified agent driver (legacy retired) | DONE | `agent/react_loop.py`; conformance FIND-01 |
| Driver phases, Inbox, parallel batches, pre/post decisions | DONE | `test_phase3_acceptance.py`, `test_kernel_k6.py` |
| Turn-end reasons, cancellation, AgentRegistry/Handle/Factory | DONE | conformance FIND-02/03/04 |
| `max-steps` exhaustion reports `completed` (no new wire value) | PARTIAL | Phase 3 spec decision; accepted by Malcom (no new frozen wire value) |
| LlmAdapter, retry, BlockAssembler, token meter, listing cache | DONE | `test_phase3_acceptance.py`, `llm/` unit suites |
| Reference adapters + pi-ai substrate; full JSON-Schema compile | DONE | `llm/pi_ai/`, `tools/json_schema.py` |
| DeepSeek/pi-ai provider/model identifiers in catalog | PARTIAL | Accepted by Malcom's 2026-11-17 ruling (user-facing identifiers allowed) |

## Phase 2 — session log

| Item | Verdict | Evidence / accepted reason |
|---|---|---|
| Format v4 + adjacent migrations v0→v4 | DONE | `session/format.py`; conformance FMT-01..06 |
| 63-type event registry, surface fold, projections, persistence | DONE | conformance EVT/STO/PER/PRJ rows |
| Checkpoint policy (`llm/stream`, `tools/execute`, `agent/pre-step`; not `turn/end`) | DONE | conformance CHK rows, `checkpoint_policy.py` |
| Titles (5/40/80 fallback + provider path) | DONE | conformance TTL rows, `test_title_roundtrip.py` |
| Non-streaming checkpointing | PARTIAL | DEF-03; accepted by Malcom via ratified conformance map |
| Compression via plain JSONL (compressed gen is a hard error) | PARTIAL | DEF-01; accepted deferral |
| Repair command refused by default (no silent repair) | PARTIAL | DEF-02; accepted decision |
| Crash-recovery end-to-end | PARTIAL | DEF-04; accepted deferral (needs built install) |
| CLI title drain on exit | PARTIAL | K6-CLI-TITLE; accepted/pinned |
| `message-quotes` (Client UI) | — | Out of scope (§7 / OOS-03); not counted |

## Phase 3 — tools and shell services

| Item | Verdict | Evidence / accepted reason |
|---|---|---|
| Tool pipeline, JSON-Schema compile, parallel batches | DONE | conformance FIND-07/10, `test_pipeline.py` |
| `bash`, `read_file`, `write_file`, `edit_file`, `execute_python` | DONE | tool catalog; Python `_file` suffixes = documented adjustment |
| `glob`, `grep` (fs-search-tool, Tier 1) | DONE | C6 fix PR #45; `test_fs_search.py` |
| Approval wiring in pipeline, one-shot grants, audit pair | DONE | conformance FIND-09/10/11/14 |
| `fs`, `shell`, `subprocess`, `jobs` services | DONE | conformance `test_phase3_acceptance` infra row |
| `run_code` reserved name | DONE | `tools/service.py` |

## Phase 4 — guards and safety

| Item | Verdict | Evidence / accepted reason |
|---|---|---|
| Approval service, policy/turn/audit, fail-closed default guard | DONE | conformance FIND-09/12/14 |
| Repeat reminder {3,5,8} consecutive | DONE | conformance FIND-18 |
| Timeout policy (cooperative TOOL_TIMEOUT, sync + async) | DONE | conformance FIND-19 |
| Compaction: surface replace + 0.8/0.16/65536 token budgets + wired pruner | DONE | C6 fix PR #45, conformance FIND-15 |
| CodeAct escape battery, fail-closed sandbox | DONE | conformance FIND-08/13 |
| Windows ACL sandbox backend | PARTIAL | Finding #13; accepted by Malcom in ratified plan C1 (documented gap) |
| fs `sandbox_policy` enforcement on all paths | PARTIAL | Finding #13; accepted by Malcom (documented gap) |
| Sandbox escalation (`sandbox_permissions`, justification, wider modes) | PARTIAL | Finding #13; accepted by Malcom (documented gap) |
| Workspace-trust gate, truncation guard, Pollinations, git-shadow checkpoint | DONE | Four kept inventions (ratified §8), labelled magitech-only extensions |
| SAFE_READ path confinement | DONE | conformance FIND-20 |

## Phase 5 — boot, profiles, CLI, Tier 1

| Item | Verdict | Evidence / accepted reason |
|---|---|---|
| Profile boot, CLI, single one-shot command | DONE | PR #58; `test_cli.py`; conformance K6-CLI |
| Tier 1 12-plugin list | DONE | All 12 present (see tool/provider catalogs and `index.json`); C6 fix closed fs-search |
| Four Tier 2 plugins (skills/session-search/subagent/steering) | — | Out of 1.0 scope (ratified §8); present as drift, not counted |
| Everything-is-a-Plugin Branch A (plugins in harness as bundle rows) | DONE | Ratified §8; native guards in harness base bundle |

## Phase 6 — conformance and secondary systems landed before 1.0

| Item | Verdict | Evidence / accepted reason |
|---|---|---|
| Recorded v4 fixtures replay (replay/title/compaction/approval) | DONE | `test_c4_fixture_replay.py`, conformance FIX-01..04 |
| 20 findings as live conformance cases | DONE | `test_c4_conformance.py` (FIND-01..20) |
| ACC-06 bidirectional wire round-trip | DONE | `test_cross_tooling_roundtrip.py`, conformance ACC-06; see `docs/conformance/acc-06.md` |
| Spill, session-query, telemetry, credentials, schedule (P7 landed early) | DONE | Conformance-covered; Phase 7 tail but landed and proven |
| Lineage disclosed openly (former §12 scrub requirement) | DONE | Ratified 2026-11-22 in agents-home D-025 and anima ADR 0012: the Cordis / DeepSeek Harness lineage is stated in `anima/README.md`, attributed in `anima/NOTICE`, and pinned above. The scrub-check gate was retired; no history rewrite is needed or authorised. The two 2026-11-07 commit messages remain in harness history by decision. |

## Acceptance summary

- In-scope MISSING: **none**.
- In-scope DIVERGED: **none**.
- PARTIAL rows: all carry reasons accepted by Malcom (ratified §8, §8 rulings,
  pinned conformance rows, or the 2026-11-18 scrub decision, now superseded) —
  none accepted by an agent on his behalf.
- Criterion 1 (the parity map) therefore holds at the rc.1 audit target.
- Re-ratification 2026-11-19: PIN-01 moved from PARTIAL to DONE at Malcom's
  decision after anima adopted failure aggregation (all-settled semantics,
  `ExceptionGroup` for ≥2 failures); the acceptance test and both parity maps
  were updated in the same refinement program.
- §12's "clean in tree AND history" line was **superseded on 2026-11-22** by
  ratified lineage disclosure (D-025 / ADR 0012); no history rewrite is needed
  or authorised, and the two residual commit messages remain by decision.

## Post-1.0 appendix — rc.3 drift (NOT counted against 1.0)

Per Malcom's 2026-11-18 freeze decision, these rc.1→rc.3 in-scope behavioral
changes are recorded for the post-1.0 tail, not fixed for 1.0:

- Runtime-context `<system-reminder>` framing and sections-based dedup (model-visible prompt text).
- `Agent.cancel` try/finally, lifetime-signal wake guard, teardown drain on cancel failure.
- Structured tool-denial identity (`ApprovalError`/`APPROVAL_*`, guard `info`, `AbortError` reclassification).
- pi-ai prefix/thinking binding, leading-HTTP-status classification, Gemini discovery.
- Working-directory typed errors + `working-directory/access` event.
- Compaction `answered-route` projection and last-assistant-route target selection.
- LLM error-code registry metadata; DeepSeek Files-API 20s timeout/backoff.

These become the first items of the Phase 7 rolling tail / re-audit cadence after 1.0.
