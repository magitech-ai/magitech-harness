# 0022. Supersede ADR 0002: base-bundle domain tools belong in magitech-harness

Date: 2026-11-25
Status: accepted (user decision, evidence-backed)
Supersedes: ADR 0002 (zero built-in domain tools), ADR 0012 (Tier 1 provider substrate placement)

## Context

The user's port target is DeepSeek Harness (dsh) commit `4b399c39a6d1479cd84ca591b4d7c88e24d38a96`
(v0.2.3-alpha.2), cloned read-only at
`/mnt/storage/projects/magitech/.dsh-scratch/dsh`. The port must reach close-to-1:1 behavioral and API
parity, mapped feature-by-feature in `PARITY.md` with exact upstream file:line citations and executable
probes.

Architecture rule, applied mechanically (user instruction, restated here as decision):

> Any tool in DSH's `packages/bundle/base/cordis.patch.yml` belongs in `magitech-harness`. Optional,
> on-demand and third-party tools belong in `mth-components`. Kernel semantics (events, effects, scopes,
> platform entry conditions) belong in `animas`.

DSH's base bundle at the pin has 101 rows (parses as plain YAML after the `!!js` expressions are
translated; see the translator ADR). Of those, the rows that register model-facing tools or core
services — `llm`, `session`, `session-title`, `session-title-llm`, `agent`, `jobs`, `llm-retry`,
`llm-pi-ai`, `session-persistence-jsonl`, `session-query-sqlite`, `working-directory`,
`session-projection`, `session-projection-cache`, `subprocess`, `sandbox`, `sandbox-policy`, `approval`,
`permission`, `shell-env`, `tool-bash`, `tool-pwsh`, `tool-jobs`, `fs-observation-policy`, `tool-fs`,
`tool-fs-search`, `agent-instructions`, `skill`, `skill-filesystem`, `tool-skill`, `upstream-skills`,
`commands`, `command-feedback`, `goal`, `goal-round-driver`, `command-goal`, `plan-mode`,
`token-meter`, `compaction-basic`, `command-compact`, `subagent`, `agent-team`, `tool-agent-team`,
`subagent-spawn-in-process`, `subagent-fork-in-process`, `tool-subagent-control`, `tool-subagent`,
`tool-subagent-fork`, `tool-workflow`, `timeout-policy`, `spill-local`, `spill-policy`,
`session-checkpoint-policy`, `tool-result-pruner`, `image-offload`, `tool-todo`, `tool-goal`,
`repeat-tool-reminder`, `web`, `web-search-deepseek`, `web-deepseek-api`, `web-fetch-http`, `tool-web`,
`mcp-resources`, `tools`, `tool-executor-native`, `system-prompt`, `agent-loop`, `fs-sandbox`,
`llm-deepseek`, `llm-deepseek-account` — are *base-bundle* rows and therefore belong in
`magitech-harness`, not `mth-components`.

ADR 0002's blanket "zero built-in domain tools" is contradicted by this rule: the dsh base bundle IS
the harness's Tier 1 substrate, and its tools must ship in the harness repo to be mounted by
`profiles/base.yaml`. ADR 0012 made exactly this exception for `llm-pi-ai` (the multi-provider adapter
substrate); this ADR broadens the same Tier 1 precedent to every base-bundle tool row.

Evidence:
- DSH base patch, 101 rows: `packages/bundle/base/cordis.patch.yml` (pin), model-facing tool names at
  `packages/shell/tool-bash/src/index.ts:415` (`bash`), `packages/fs/tool-fs/src/read.ts:79` (`read`),
  `packages/fs/tool-fs/src/write.ts:75` (`write`), `packages/fs/tool-fs/src/edit.ts:87` (`edit`),
  `packages/fs/tool-fs-search/src/glob.ts:323` (`glob`), `packages/fs/tool-fs-search/src/grep.ts:288`
  (`grep`), `packages/jobs/tool-jobs/src/index.ts` (`job_output` 550, `job_list` 572, `job_kill` 435),
  `packages/todo/tool-todo/src/index.ts:137` (`todo_write`), `packages/goal/tool-goal/src/index.ts`
  (`get_goal` 197, `create_goal` 209, `update_goal` 236), `packages/subagent/tool-subagent-control/src/index.ts:45`
  (`send_message`), `:126` (`interrupt_agent`), `packages/subagent/tool-subagent-control/src/list-agents.ts:46`
  (`list_agents`), `:91` (`wait_agent`), `packages/team/tool-agent-team/src/index.ts:119` (`team_task_create`),
  `:145` (`team_task_list`), `:186` (`team_task_get`), `:202` (`team_task_update`),
  `packages/plan/plan-mode/src/index.ts:252` (`plan`), `packages/workflow/tool-workflow/src/index.ts`
  (`workflow`), `packages/web/tool-web/src/search.ts:367` (`web_search`), `packages/web/tool-web/src/fetch.ts:460`
  (`web_fetch`), `packages/interaction/tool-ask-user/src/index.ts:40` (`ask_user_question`).
- `packages/AGENTS.md` package rule: service packages default-export their service class; function
  plugins named-export `name`/`inject`/`Config`/`apply`. The Python equivalent is `apply(ctx, **kwargs)`
  components mounted as profile rows.

## Decision

1. **Supersede ADR 0002.** `magitech-harness` no longer ships "zero built-in domain tools." Instead it
   ships every dsh base-bundle tool row as a Python component mounted by `profiles/base.yaml`. The
   "everything is a component" principle is unchanged: each tool is an `apply(ctx, **kwargs)` component
   registering onto `ctx.tools`, `ctx.llm`, `ctx.sessions`, `ctx.compaction`, etc., and nothing is
   default-mounted beyond what `base.yaml` declares.
2. **Broaden ADR 0012's Tier 1 precedent.** Tier 1 substrate (in-tree, default-mounted by `base.yaml`,
   opt-in behavior) now covers all base-bundle tool/service rows, not only `llm-pi-ai`. `mth-components`
   keeps only optional, on-demand, and third-party tools/providers/guards/bridges.
3. `profiles/base.yaml` is the single source of truth for what is default-mounted. Each row's Python
   implementation location is cited in `PARITY.md` (Phase 0 table).
4. Marketplace-owned rows (plugin manager, hmr, config-editor, settings, authorization, credentials,
   storage, otel telemetry, attachment, upstream-skills deepseek account/auth, etc.) remain in
   `mth-components` or are out of scope (GUI, desktop, Windows ACL, external connectors).

## Consequences

- `docs/architecture.md`, `CONTEXT.md`, `AGENTS.md`, and `docs/cross-repo-contracts.md` must be
  rewritten to reflect base-bundle tools in the harness; the guard-ownership table must list each
  base guard as native.
- ADR 0002's body is preserved as history; it is not edited. This ADR points at it as superseded.
- ADR 0012's body is preserved as history; its README/placement statements are superseded here.
- The parity map `PARITY.md` (superseding the fcc0ad46 record) is the enforcement artifact: every
  base-bundle row carries Owner = magitech-harness or mth-components and a Python file:line or MISSING.
- No silent violations: any row moved between repos requires a new ADR superseding this one.

## Acceptance

Every base-bundle tool row in `PARITY.md` Phase 0 must be DONE (implemented + executable probe) or
PARTIAL (named acceptor reason) before the port is considered complete.
