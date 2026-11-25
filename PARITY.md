# Parity map — DeepSeek Harness v0.2.3-alpha.2 (pin 4b399c39a6d1479cd84ca591b4d7c88e24d38a96)

This is the authoritative, evidence-backed 1:1 parity map between DeepSeek Harness (dsh) commit
`4b399c39a6d1479cd84ca591b4d7c88e24d38a96` (master head, 2026-11-22) and the Python stack
`animas` (kernel) / `magitech-harness` (host runtime, turn engine, Tier 1 substrate) /
`mth-components` (marketplace for optional, on-demand, third-party tools).

It replaces `fcc0ad46` (0.2.2-rc.1), which is now historical provenance only. Lineage is preserved
and attributed: see `animas/docs/adr/0012-lineage-disclosure.md`, `animas/NOTICE`, and
`agents-home` decision D-025. Nothing is scrubbed.

## Verdict vocabulary

- **DONE** — reference shape implemented and proven by an executable probe (test or CLI probe).
- **PARTIAL** — implemented with a recorded gap; the gap reason names who accepted it (Malcom by name,
  or a pinned row), never an agent speaking for Malcom.
- **DIVERGED** — a ratified decision is not met; DIVERGED in-scope fails the gate until fixed or accepted.
- **MISSING** — in-scope capability absent. Fails the gate.
- **OUT OF SCOPE** — deliberately not part of this parity (GUI, desktop, web client, external connectors,
  Windows ACL sandbox). Recorded here so nothing is silently dropped.

## Repository ownership rule (mechanical)

> Any tool in DSH's `packages/bundle/base/cordis.patch.yml` belongs in `magitech-harness`. Optional,
> on-demand and third-party tools belong in `mth-components`. Kernel semantics (events, effects,
> scopes, platform entry conditions) belong in `animas`.

The DSH base bundle at the pin has **101 rows**. Each row below maps to exactly one owning repo and one
Python implementation location (or MISSING). DSH file:line citations point at the pinned clone under
`/mnt/storage/projects/magitech/.dsh-scratch/dsh` (read-only). Python file:line citations point at this
checkout.

## Closed set of `!!js` config expressions (ADR 0022 boundary)

DSH's base patch uses exactly 16 `!!js` expressions, all translated at port time into declarative
Animas-native profiles by the checked-in translator `scripts/profile_translator.py` with golden tests
in `tests/test_profile_translator.py`. Never add a runtime expression evaluator to animas.

| # | DSH expression (base patch row) | Declarative equivalent | Python location |
|---|---|---|---|
| 1 | `!!js "!ctx.get('profileContext')"` | `enabled_on: [ profile ]` / `disabled_on: []` | base.yaml `plugin-manager`, `config-editor`, `settings` |
| 2 | `!!js "ctx.get('profileContext')?.name === 'desktop' && ['darwin','win32'].includes(process.platform) ? process.platform : null"` | `desktopPlatform: null` (no-op value) | base.yaml `deepseek-account` — translatable, no Python behavior |
| 3 | `!!js dshHomePath('sessions')` | `root: ${MAGITECH_SESSIONS_DIR}` | base.yaml `session-persistence-jsonl` |
| 4 | `!!js dshHomePath('storages')` | `root: ${MAGITECH_SESSIONS_DIR}/storages` | base.yaml `storage-json` |
| 5 | `!!js process.env.DSH_TELEMETRY_MODE || 'FEEDBACK_ONLY'` | `mode: ${DSH_TELEMETRY_MODE:FEEDBACK_ONLY}` | base.yaml `session-telemetry-otel` |
| 6 | `!!js process.env.DSH_TELEMETRY_OTLP_URL ?? 'https://...'` | `exporter.url: ${DSH_TELEMETRY_OTLP_URL:...}` | base.yaml `session-telemetry-otel` |
| 7 | `!!js process.env.DSH_PERMISSION_MODE ?? 'workspace-write'` | `mode: ${DSH_PERMISSION_MODE:workspace-write}` | base.yaml `sandbox-policy` |
| 8 | `!!js process.cwd()` | `workspaceRoot: ${PWD}` | base.yaml `sandbox-policy` |
| 9 | `!!js process.platform === 'win32'` | `disabled_on: [ win32 ]` | base.yaml `tool-bash`, `bash-sandbox` |
| 10| `!!js process.platform !== 'win32'` | `enabled_on: [ win32 ]` | base.yaml `tool-pwsh`, `pwsh-sandbox` |
| 11| `!!js "(process.env.DSH_PERMISSION_MODE ?? 'workspace-write') === 'danger-full-access' ? 'never' : 'ask'"` | `policy: ${DSH_PERMISSION_MODE:ask:danger=never}` | base.yaml `approval` |
| 12| `!!js process.env.DSH_TELEMETRY_DISABLED` (outer config, not base) | `disabled: ${DSH_TELEMETRY_DISABLED:true}` | base.yaml `session-telemetry-otel` |

The translator golden tests pin each expression → declarative row. See `scripts/profile_translator.py`.

## Phase 0 — base-bundle tool rows (101 rows)

Each DSH row: package, DSH source, owning repo, Python implementation, verdict, probe.

| # | DSH id | DSH package | DSH file:line | Owner | Python impl (file:line) | Verdict | Probe |
|---|---|---|---|---|---|---|---|
| 1 | tool-plugin-manager | `@deepseek-ai/dsh-plugin-manager/tools` | packages/boot/plugin-manager/src/tools.ts:92 | mth-components (plugin manager is opt-in infra) | NONE | OUT OF SCOPE | plugin manager is optional, not base Tier 1 |
| 2 | plugin-manager | `@deepseek-ai/dsh-plugin-manager` | packages/boot/plugin-manager/src/index.ts | mth-components | NONE | OUT OF SCOPE | optional bundle |
| 3 | timer | `@deepseek-ai/cordis-plugin-timer` | vendor/ | animas | NONE | OUT OF SCOPE | kernel timer, not harness |
| 4 | hmr | `@deepseek-ai/dsh-hmr` | packages/boot/hmr/src/index.ts | mth-components | NONE | OUT OF SCOPE | optional |
| 5 | llm | `@deepseek-ai/dsh-llm` | packages/llm/llm/src/index.ts | magitech-harness | `src/magitech_harness/llm/service.py` | DONE | `test_llm_service.py` |
| 6 | deepseek-llm-api-extensions | `@deepseek-ai/dsh-deepseek-llm-api-extensions` | packages/llm/deepseek-llm-api-extensions/src/ | DSH-specific | NONE | OUT OF SCOPE | DeepSeek-specific extension |
| 7 | session | `@deepseek-ai/dsh-session` | packages/core/session/src/index.ts | magitech-harness | `src/magitech_harness/session/service.py` | DONE | `test_session_service.py` |
| 8 | session-log-deepseek | `@deepseek-ai/dsh-session-log-deepseek` | packages/session/session-log-deepseek/src/ | DSH-specific | NONE | OUT OF SCOPE | DSH-specific log contributor |
| 9 | typert | `@deepseek-ai/dsh-typert-registry` | packages/typert/registry/src/ | DSH-specific | NONE | OUT OF SCOPE | type graph runtime |
| 10 | typert-loader | `@deepseek-ai/dsh-typert-loader` | packages/typert/loader/src/ | DSH-specific | NONE | OUT OF SCOPE | |
| 11 | typert-gateway | `@deepseek-ai/dsh-api-gateway` | packages/api/gateway/src/ | DSH-specific | NONE | OUT OF SCOPE | API gateway |
| 12 | remote-access | `@deepseek-ai/dsh-api-remote-access` | packages/api/remote-access/src/ | DSH-specific | NONE | OUT OF SCOPE | remote access |
| 13 | session-title | `@deepseek-ai/dsh-session-title` | packages/session/session-title/src/index.ts | magitech-harness | `src/magitech_harness/session/title.py` | DONE | `test_title.py` |
| 14 | session-title-llm | `@deepseek-ai/dsh-session-title-first-prompt-llm` | packages/session/session-title-first-prompt-llm/src/index.ts | magitech-harness | `src/magitech_harness/session/title_first_prompt.py` | DONE | `test_title_first_prompt.py` |
| 15 | user-questions | `@deepseek-ai/dsh-user-questions` | packages/interaction/user-questions/src/index.ts | magitech-harness | `src/magitech_harness/context/user_questions.py` | DONE | `test_pruner_and_default_model.py` |
| 16 | agent | `@deepseek-ai/dsh-agent` | packages/core/agent/src/index.ts | magitech-harness | `src/magitech_harness/agent/registry.py` | DONE | `test_agent_registry.py` |
| 17 | plugin-package-inventory-deepseek | `@deepseek-ai/dsh-plugin-package-inventory-deepseek` | packages/llm/plugin-package-inventory-deepseek/src/ | DSH-specific | NONE | OUT OF SCOPE | |
| 18 | agent-default-model | `@deepseek-ai/dsh-agent-default-model` | packages/core/agent-default-model/src/index.ts | magitech-harness | `src/magitech_harness/agent/default_model.py` | DONE | `test_pruner_and_default_model.py`; base row defaults to `kilo`/`kilo-auto/free`, verified live with `cost: 0` and no API key |
| 19 | jobs | `@deepseek-ai/dsh-jobs-local` | packages/jobs/jobs-local/src/index.ts | magitech-harness | `src/magitech_harness/jobs/local.py` | DONE | `test_jobs_local.py` |
| 20 | llm-retry | `@deepseek-ai/dsh-llm-retry` | packages/llm/llm-retry/src/index.ts | magitech-harness | `src/magitech_harness/llm/retry.py` | DONE | `test_llm_retry.py` |
| 21 | config-editor | `@deepseek-ai/dsh-config-editor` | packages/boot/config-editor/src/index.ts | mth-components | NONE | OUT OF SCOPE | optional |
| 22 | settings | `@deepseek-ai/dsh-settings` | packages/settings/settings/src/index.ts | mth-components | NONE | OUT OF SCOPE | optional |
| 23 | authorization | `@deepseek-ai/dsh-authorization` | packages/credentials/authorization/src/index.ts | mth-components | NONE | OUT OF SCOPE | credentials marketplace |
| 24 | deepseek-account | `@deepseek-ai/dsh-deepseek-account` | packages/account/deepseek-account/src/index.ts | DSH-specific | NONE | OUT OF SCOPE | DeepSeek account |
| 25 | deepseek-auth | `@deepseek-ai/dsh-deepseek-auth` | packages/account/deepseek-auth/src/index.ts | DSH-specific | NONE | OUT OF SCOPE | |
| 26 | credentials | `@deepseek-ai/dsh-credentials-local` | packages/credentials/credentials-local/src/index.ts | mth-components | NONE | OUT OF SCOPE | credentials marketplace |
| 27 | llm-pi-ai | `@deepseek-ai/dsh-llm-pi-ai` | packages/llm/llm-pi-ai/src/index.ts | magitech-harness (ADR 0012 Tier 1) | `src/magitech_harness/llm/pi_ai/component.py` | DONE | `test_pi_ai_component.py` |
| 28 | session-persistence-jsonl | `@deepseek-ai/dsh-session-persistence-jsonl` | packages/session/session-persistence-jsonl/src/index.ts | magitech-harness | `src/magitech_harness/session/persistence_jsonl.py` | DONE | `test_persistence_jsonl.py` |
| 29 | attachment-local | `@deepseek-ai/dsh-attachment-local` | packages/attachment/attachment-local/src/index.ts | mth-components | NONE | OUT OF SCOPE | optional attachment backend |
| 30 | session-query-sqlite | `@deepseek-ai/dsh-session-query-sqlite` | packages/session-query/session-query-sqlite/src/index.ts | mth-components | `src/mth_components/sessions/search/component.py` | DONE | `test_session_search.py` |
| 31 | working-directory | `@deepseek-ai/dsh-working-directory` | packages/session/working-directory/src/index.ts | magitech-harness | `src/magitech_harness/session/working_directory.py` | DONE | `test_working_directory.py` |
| 32 | session-projection | `@deepseek-ai/dsh-session-projection` | packages/session/session-projection/src/index.ts | magitech-harness | `src/magitech_harness/session/projection_registry.py` | DONE | `test_projection_registry.py` |
| 33 | storage | `@deepseek-ai/dsh-storage` | packages/storage/storage/src/index.ts | mth-components | NONE | OUT OF SCOPE | optional storage |
| 34 | storage-json | `@deepseek-ai/dsh-storage-json` | packages/storage/storage-json/src/index.ts | mth-components | NONE | OUT OF SCOPE | |
| 35 | storage-domain | `@deepseek-ai/dsh-storage-domain` | packages/storage/storage-domain/src/index.ts | mth-components | NONE | OUT OF SCOPE | |
| 36 | session-projection-cache | `@deepseek-ai/dsh-session-projection-cache` | packages/session/session-projection-cache/src/index.ts | magitech-harness | `src/magitech_harness/session/projection_cache.py` | DONE | `test_projection_cache.py` |
| 37 | otel | `@deepseek-ai/dsh-otel` | packages/telemetry/otel/src/index.ts | mth-components | NONE | OUT OF SCOPE | optional telemetry |
| 38 | session-telemetry-otel | `@deepseek-ai/dsh-session-telemetry-otel` | packages/session/session-telemetry-otel/src/index.ts | mth-components | NONE | OUT OF SCOPE | optional telemetry |
| 39 | subprocess | `@deepseek-ai/dsh-subprocess-local` | packages/subprocess/subprocess-local/src/index.ts | magitech-harness | `src/magitech_harness/subprocess/service.py` | DONE | `test_subprocess_service.py` |
| 40 | cli-account | `@deepseek-ai/dsh-cli-account` | packages/connector/cli-account/src/index.ts | DSH-specific | NONE | OUT OF SCOPE | |
| 41 | sandbox | `@deepseek-ai/dsh-sandbox-local` | packages/sandbox/sandbox-local/src/index.ts | mth-components | NONE | PARTIAL | sandbox service not default-mounted |
| 42 | sandbox-policy | `@deepseek-ai/dsh-sandbox-policy` | packages/sandbox/sandbox-policy/src/index.ts | mth-components | `src/mth_components/guards/...` | PARTIAL | policy not mounted |
| 43 | bash-sandbox | `@deepseek-ai/dsh-bash-sandbox` | packages/shell/bash-sandbox/src/index.ts | magitech-harness | `src/magitech_harness/shell/bash_local.py` | PARTIAL | `test_shell_service.py`; sandbox enforcement is the documented finding-#13 gap |
| 44 | pwsh-sandbox | `@deepseek-ai/dsh-pwsh-sandbox` | packages/shell/pwsh-sandbox/src/index.ts | mth-components | NONE | PARTIAL | pwsh not ported (Linux CI) |
| 45 | approval | `@deepseek-ai/dsh-user-approval` | packages/interaction/user-approval/src/index.ts | magitech-harness | `src/magitech_harness/approval/service.py` | DONE | `test_approval_service.py` |
| 46 | permission | `@deepseek-ai/dsh-permission-presets` | packages/interaction/permission-presets/src/index.ts | magitech-harness | `src/magitech_harness/approval/service.py` (policy) | DONE | `test_approval_presets.py` |
| 47 | shell-env | `@deepseek-ai/dsh-shell-env` | packages/shell/shell-env/src/index.ts | magitech-harness | `src/magitech_harness/shell_env/service.py` | DONE | `test_shell_env.py` |
| 48 | tool-bash | `@deepseek-ai/dsh-tool-bash` | packages/shell/tool-bash/src/index.ts:415 | magitech-harness (base tool) | `src/magitech_harness/tools/bash_tool.py` | DONE | `tests/unit/magitech_harness/tools/test_bash_tool.py` |
| 49 | tool-pwsh | `@deepseek-ai/dsh-tool-pwsh` | packages/shell/tool-pwsh/src/index.ts | mth-components | NONE | PARTIAL | pwsh not ported |
| 50 | tool-jobs | `@deepseek-ai/dsh-tool-jobs` | packages/jobs/tool-jobs/src/index.ts:435 | magitech-harness (base tool) | `src/magitech_harness/tools/jobs.py` | DONE | `test_jobs.py` |
| 51 | fs-observation-policy | `@deepseek-ai/dsh-fs-observation-policy` | packages/fs/fs-observation-policy/src/index.ts | magitech-harness | `src/magitech_harness/fs/observation_policy.py` | DONE | `test_observation_policy.py` |
| 52 | tool-fs | `@deepseek-ai/dsh-tool-fs` | packages/fs/tool-fs/src/index.ts | magitech-harness (base tool) | `src/magitech_harness/tools/fs.py` | DONE | `test_fs_{read,write,edit}.py`; `apply_patch` stays unported and refuses to mount |
| 53 | tool-fs-search | `@deepseek-ai/dsh-tool-fs-search` | packages/fs/tool-fs-search/src/index.ts | magitech-harness (base tool) | `src/magitech_harness/tools/fs_search.py` | DONE | `test_fs_search.py` |
| 54 | agent-instructions | `@deepseek-ai/dsh-agent-instructions` | packages/context/agent-instructions/src/index.ts | magitech-harness | NONE | PARTIAL | agent instructions row not mounted |
| 55 | skill | `@deepseek-ai/dsh-skill` | packages/skill/skill/src/index.ts | mth-components | `src/mth_components/bridges/skills_registry/component.py` | DONE | `test_skills_registry.py` |
| 56 | skill-filesystem | `@deepseek-ai/dsh-skill-filesystem` | packages/skill/skill-filesystem/src/index.ts | mth-components | `src/mth_components/bridges/skills_filesystem/component.py` | DONE | `test_skills_filesystem.py` |
| 57 | tool-skill | `@deepseek-ai/dsh-tool-skill` | packages/skill/tool-skill/src/index.ts | mth-components | `src/mth_components/bridges/skills_loader/component.py` | DONE | `test_skills_loader.py` |
| 58 | upstream-skills | `@deepseek-ai/dsh-upstream-skills` | packages/connector/upstream-skills/src/index.ts | mth-components | NONE | PARTIAL | upstream skills not mounted |
| 59 | commands | `@deepseek-ai/dsh-commands` | packages/interaction/commands/src/index.ts | magitech-harness | `src/magitech_harness/commands/` | DONE | `test_commands.py` (durable registry, alias resolution, `command/run`/`command/done` lifecycle, REPL wired through the registry) |
| 60 | command-feedback | `@deepseek-ai/dsh-command-feedback` | packages/feedback/command-feedback/src/index.ts | magitech-harness | `src/magitech_harness/feedback/` | DONE | `test_commands.py` (`feedback/record` event, `sessionFeedback` service, `/feedback` command) |
| 61 | goal | `@deepseek-ai/dsh-goal` | packages/goal/goal/src/index.ts | magitech-harness (base tool) | `src/magitech_harness/goal/{types,projection,service}.py` | DONE | `test_goal.py` |
| 62 | goal-round-driver | `@deepseek-ai/dsh-goal-round-driver` | packages/goal/goal-round-driver/src/index.ts | magitech-harness | NONE | PARTIAL | automatic continuation rounds not wired into the loop; entry/complete/block authority is implemented and probe-tested |
| 63 | command-goal | `@deepseek-ai/dsh-command-goal` | packages/goal/command-goal/src/index.ts | magitech-harness | `src/magitech_harness/goal/command.py` | DONE | `test_goal.py`, `test_cli.py` |
| 64 | plan-mode | `@deepseek-ai/dsh-plan-mode` | packages/plan/plan-mode/src/index.ts | magitech-harness (base tool) | `src/magitech_harness/plan/` | DONE | `test_plan.py` (durable `plan` projection, step-boundary admission, conditional `exit_plan_mode` with user review) |
| 65 | token-meter | `@deepseek-ai/dsh-token-meter` | packages/llm/token-meter/src/index.ts | magitech-harness | `src/magitech_harness/llm/token_meter.py` | DONE | `test_token_meter.py` |
| 66 | compaction-basic | `@deepseek-ai/dsh-compaction-basic` | packages/compaction/compaction-basic/src/index.ts | magitech-harness | `src/magitech_harness/compaction/service.py` | DONE | `test_compaction_service.py` |
| 67 | command-compact | `@deepseek-ai/dsh-command-compact` | packages/compaction/command-compact/src/index.ts | magitech-harness | `src/magitech_harness/cli.py` (`/compact` → `session/compact` waterfall) | DONE | `test_cli.py` |
| 68 | subagent | `@deepseek-ai/dsh-subagent` | packages/core/subagent/src/index.ts | magitech-harness | `src/magitech_harness/subagent/service.py` | DONE | `test_manager.py`, `test_surface.py`; lineage logged as `subagent/descriptor` |
| 69 | agent-team | `@deepseek-ai/dsh-agent-team` | packages/team/agent-team/src/index.ts | magitech-harness | `src/magitech_harness/team/` | DONE | `test_team.py` (event-sourced board, CAS revisions, dependency graph, readiness/write-scope views) |
| 70 | tool-agent-team | `@deepseek-ai/dsh-tool-agent-team` | packages/team/tool-agent-team/src/index.ts:119 | magitech-harness (base tool) | `src/magitech_harness/team/tools.py` | DONE | `test_team.py` (4 tools, conditional admission, Team policy section) |
| 71 | subagent-spawn-in-process | `@deepseek-ai/dsh-subagent-spawn-in-process` | packages/subagent/subagent-spawn-in-process/src/index.ts | magitech-harness | `src/magitech_harness/subagent/providers.py` | DONE | `test_surface.py` |
| 72 | subagent-fork-in-process | `@deepseek-ai/dsh-subagent-fork-in-process` | packages/subagent/subagent-fork-in-process/src/index.ts | magitech-harness | `src/magitech_harness/subagent/providers.py` | DONE | `test_surface.py` (replays parent events with surface ops) |
| 73 | tool-subagent-control | `@deepseek-ai/dsh-tool-subagent-control` | packages/subagent/tool-subagent-control/src/index.ts:45 | magitech-harness (base tool) | `src/magitech_harness/subagent/tools.py` | DONE | `test_surface.py`, `test_manager.py` |
| 74 | tool-subagent | `@deepseek-ai/dsh-tool-subagent` | packages/subagent/tool-subagent/src/index.ts:103 | magitech-harness (base tool) | `src/magitech_harness/subagent/tools.py` | DONE | `test_surface.py`; dsh tool name `subagent` plus `list_subagent_models` |
| 75 | tool-subagent-fork | `@deepseek-ai/dsh-tool-subagent` (fork) | packages/subagent/tool-subagent/src/index.ts:103 | magitech-harness (base tool) | `src/magitech_harness/subagent/tools.py` | DONE | `test_manager.py`; tool name `subagent_fork` |
| 76 | ptc-runtime | `@deepseek-ai/dsh-ptc-runtime-node` | packages/ptc-runtime/ptc-runtime-node/src/ | DSH-specific | NONE | OUT OF SCOPE | PTC Node runtime |
| 77 | workflow-ptc | `@deepseek-ai/dsh-workflow-ptc` | packages/workflow/workflow-ptc/src/ | DSH-specific | NONE | OUT OF SCOPE | |
| 78 | tool-workflow | `@deepseek-ai/dsh-tool-workflow` | packages/workflow/tool-workflow/src/index.ts | DSH-specific | NONE | OUT OF SCOPE | The `workflow` tool executes JavaScript orchestration scripts through `@deepseek-ai/dsh-workflow-ptc` (row 77, already out of scope). There is no Python/JS workflow engine in this harness; the seam would have no backend. |
| 79 | timeout-policy | `@deepseek-ai/dsh-tool-call-timeout-policy` | packages/guard/timeout-policy/src/index.ts | magitech-harness | `src/magitech_harness/tools/timeout_guard.py` | DONE | `test_timeout_guard.py` |
| 80 | spill-local | `@deepseek-ai/dsh-spill-local` | packages/spill/spill-local/src/index.ts | magitech-harness | `src/magitech_harness/spill/local.py` | DONE | `test_spill_local.py` |
| 81 | spill-policy | `@deepseek-ai/dsh-spill-policy` | packages/spill/spill-policy/src/index.ts | magitech-harness | `src/magitech_harness/spill/policy.py` | DONE | `test_spill_policy.py` |
| 82 | session-checkpoint-policy | `@deepseek-ai/dsh-session-checkpoint-policy` | packages/session/session-checkpoint-policy/src/index.ts | magitech-harness | `src/magitech_harness/session/checkpoint_policy.py` | DONE | `test_checkpoint_policy.py` |
| 83 | tool-result-pruner | `@deepseek-ai/dsh-compaction-tool-result-pruner` | packages/compaction/compaction-tool-result-pruner/src/index.ts | magitech-harness | `src/magitech_harness/compaction/pruner.py` | DONE | `test_pruner_and_default_model.py` |
| 84 | image-offload | `@deepseek-ai/dsh-compaction-image-offload` | packages/compaction/compaction-image-offload/src/index.ts | magitech-harness | `src/magitech_harness/compaction/image_offload.py` | DONE | `test_image_offload.py` (durable projection, oldest-image selection, `agent/request-error` retry wiring) |
| 85 | tool-todo | `@deepseek-ai/dsh-tool-todo` | packages/todo/tool-todo/src/index.ts:137 | magitech-harness (base tool) | `src/magitech_harness/tools/todo.py` | DONE | `tests/unit/magitech_harness/tools/test_todo.py` |
| 86 | tool-goal | `@deepseek-ai/dsh-tool-goal` | packages/goal/tool-goal/src/index.ts:197 | magitech-harness (base tool) | `src/magitech_harness/goal/tools.py` | DONE | `test_goal.py` |
| 87 | repeat-tool-reminder | `@deepseek-ai/dsh-repeat-tool-reminder` | packages/guard/repeat-tool-reminder/src/index.ts | magitech-harness | `src/magitech_harness/tools/repeat_guard.py` | DONE | `test_repeat_guard.py` |
| 88 | web | `@deepseek-ai/dsh-web` | packages/web/web/src/index.ts | magitech-harness | `src/magitech_harness/web/service.py` | DONE | `test_web.py` (provider protocol, route selection, duplicate-id rejection, `list_providers`, `web/providers-changed`, `WEB_*` taxonomy, central result cap). Typert Remote RPC wrapper is out of scope (row 12). |
| 89 | web-search-deepseek | `@deepseek-ai/dsh-web-search-deepseek` | packages/web/web-search-deepseek/src/index.ts | DSH-specific | NONE | OUT OF SCOPE | DeepSeek account-sign-in provider: requires `dsh-deepseek-auth`, `dsh-credentials` and `dsh-launch-environment`, none of which exist in this harness (default route is Kilo). The provider-neutral `ctx.web` seam is ported (row 88) and accepts any registered search provider. |
| 90 | web-deepseek-api | `@deepseek-ai/dsh-web-deepseek-api` | packages/web/web-deepseek-api/src/index.ts | DSH-specific | NONE | OUT OF SCOPE | DeepSeek Search/Fetch API provider with the same `dsh-deepseek-auth` account-credential dependency as row 89; no equivalent account service in this harness. |
| 91 | web-fetch-http | `@deepseek-ai/dsh-web-fetch-http` | packages/web/web-fetch-http/src/index.ts | magitech-harness | `src/magitech_harness/web/service.py` | DONE | `test_web.py` (byte/char caps, content-type classification, charset, same-origin redirects, bounded timeout, User-Agent) |
| 92 | tool-web | `@deepseek-ai/dsh-tool-web` | packages/web/tool-web/src/search.ts:367 | magitech-harness (base tool) | `src/magitech_harness/web/tools.py` | DONE | `test_web.py` (`web_search`, `web_fetch`; route-aware `when` admission, untrusted-data guidance) |
| 93 | mcp-resources | `@deepseek-ai/dsh-mcp-resources` | packages/mcp/mcp-resources/src/index.ts | mth-components | `src/mth_components/bridges/mcp/component.py` | DONE | `test_mcp_bridge.py` |
| 94 | tools | `@deepseek-ai/dsh-tools` | packages/core/tools/src/index.ts | magitech-harness | `src/magitech_harness/tools/service.py` | DONE | `test_tools_service.py` |
| 95 | tool-executor-native | `@deepseek-ai/dsh-tool-executor-native` | packages/core/tool-executor-native/src/index.ts | magitech-harness | `src/magitech_harness/tools/native_executor.py` | DONE | `test_native_executor.py` |
| 96 | tool-executor-ptc | `@deepseek-ai/dsh-tool-executor-ptc` | packages/core/tool-executor-ptc/src/index.ts | DSH-specific | NONE | OUT OF SCOPE | PTC executor |
| 97 | system-prompt | `@deepseek-ai/dsh-system-prompt` | packages/core/system-prompt/src/index.ts | magitech-harness | `src/magitech_harness/system_prompt/service.py` | DONE | `test_system_prompt.py` |
| 98 | agent-loop | `@deepseek-ai/dsh-agent-loop` | packages/core/agent-loop/src/index.ts | magitech-harness | `src/magitech_harness/agent/react_loop.py` | DONE | `test_agent_loop.py` |
| 99 | fs-sandbox | `@deepseek-ai/dsh-fs-sandbox` | packages/fs/fs-sandbox/src/index.ts | magitech-harness | `src/magitech_harness/fs/local.py` (sandboxed) | DONE | `test_fs_local.py` |
| 100 | llm-deepseek | `@deepseek-ai/dsh-llm-deepseek-api-key` | packages/llm/llm-deepseek-api-key/src/index.ts | magitech-harness (Tier 1 pi-ai) | `src/magitech_harness/llm/pi_ai/` | DONE | `test_pi_ai_deepseek.py` |
| 101 | llm-deepseek-account | `@deepseek-ai/dsh-llm-deepseek-account` | packages/llm/llm-deepseek-account/src/index.ts | magitech-harness (Tier 1 pi-ai) | `src/magitech_harness/llm/pi_ai/` | DONE | `test_pi_ai_deepseek_account.py` |

## Tier 1 model-facing tool inventory (base tools, must be executable)

DSH base registers these model-facing tool names; each must exist in the Python stack with a probe.

| DSH tool name | DSH registration file:line | Python name | Python impl | Verdict | Probe |
|---|---|---|---|---|---|
| `bash` | packages/shell/tool-bash/src/index.ts:415 | `bash` | `src/magitech_harness/tools/bash_tool.py` | DONE | `test_bash_tool.py` |
| `read` | packages/fs/tool-fs/src/read.ts:79 | `read_file` | `src/magitech_harness/tools/fs.py` | DONE | `test_fs_read.py` |
| `write` | packages/fs/tool-fs/src/write.ts:75 | `write_file` | `src/magitech_harness/tools/fs.py` | DONE | `test_fs_write.py` |
| `edit` | packages/fs/tool-fs/src/edit.ts:87 | `edit_file` | `src/magitech_harness/tools/fs.py` | DONE | `test_fs_edit.py` |
| `glob` | packages/fs/tool-fs-search/src/glob.ts:323 | `glob` | `src/magitech_harness/tools/fs_search.py` | DONE | `test_fs_search.py` |
| `grep` | packages/fs/tool-fs-search/src/grep.ts:288 | `grep` | `src/magitech_harness/tools/fs_search.py` | DONE | `test_fs_search.py` |
| `job_output` | packages/jobs/tool-jobs/src/index.ts:550 | `job_output` | `src/magitech_harness/tools/jobs.py` | DONE | `test_jobs.py` |
| `job_list` | packages/jobs/tool-jobs/src/index.ts:572 | `job_list` | `src/magitech_harness/tools/jobs.py` | DONE | `test_jobs.py` |
| `job_kill` | packages/jobs/tool-jobs/src/index.ts:435 | `job_kill` | `src/magitech_harness/tools/jobs.py` | DONE | `test_jobs.py` |
| `todo_write` | packages/todo/tool-todo/src/index.ts:137 | `todo_write` | `src/magitech_harness/tools/todo.py` | DONE | `test_todo.py` |
| `get_goal` | packages/goal/tool-goal/src/index.ts:197 | `get_goal` | `src/magitech_harness/goal/tools.py` | DONE | `test_goal.py` |
| `create_goal` | packages/goal/tool-goal/src/index.ts:209 | `create_goal` | `src/magitech_harness/goal/tools.py` | DONE | `test_goal.py` |
| `update_goal` | packages/goal/tool-goal/src/index.ts:236 | `update_goal` | `src/magitech_harness/goal/tools.py` | DONE | `test_goal.py` |
| `send_message` | packages/subagent/tool-subagent-control/src/index.ts:45 | `send_message` | `src/magitech_harness/subagent/tools.py` | DONE | `test_surface.py`, `test_review_probes.py` |
| `interrupt_agent` | packages/subagent/tool-subagent-control/src/index.ts:126 | `interrupt_agent` | `src/magitech_harness/subagent/tools.py` | DONE | `test_surface.py`, `test_review_probes.py` |
| `list_agents` | packages/subagent/tool-subagent-control/src/list-agents.ts:46 | `list_agents` | `src/magitech_harness/subagent/tools.py` | DONE | `test_surface.py`, `test_review_probes.py` |
| `wait_agent` | packages/subagent/tool-subagent-control/src/wait-agent.ts:91 | `wait_agent` | `src/magitech_harness/subagent/tools.py` | DONE | `test_surface.py`, `test_review_probes.py` |
| `team_task_create` | packages/team/tool-agent-team/src/index.ts:119 | `team_task_create` | `src/magitech_harness/team/tools.py` | DONE | `test_team.py` |
| `team_task_list` | packages/team/tool-agent-team/src/index.ts:145 | `team_task_list` | `src/magitech_harness/team/tools.py` | DONE | `test_team.py` |
| `team_task_get` | packages/team/tool-agent-team/src/index.ts:186 | `team_task_get` | `src/magitech_harness/team/tools.py` | DONE | `test_team.py` |
| `team_task_update` | packages/team/tool-agent-team/src/index.ts:202 | `team_task_update` | `src/magitech_harness/team/tools.py` | DONE | `test_team.py` |
| `plan` | packages/plan/plan-mode/src/index.ts:252 | `plan` | `src/magitech_harness/plan/tools.py` | DONE | `test_plan.py` |
| `exit_plan_mode` | packages/plan/plan-mode/src/index.ts | `exit_plan_mode` | `src/magitech_harness/plan/tools.py` | DONE | `test_plan.py` |
| `workflow` | packages/workflow/tool-workflow/src/index.ts | — | NONE | OUT OF SCOPE | Depends on out-of-scope `workflow-ptc` JS runtime (row 77). |
| `web_search` | packages/web/tool-web/src/search.ts:367 | `web_search` | `src/magitech_harness/web/tools.py` | DONE | `test_web.py` |
| `web_fetch` | packages/web/tool-web/src/fetch.ts:460 | `web_fetch` | `src/magitech_harness/web/tools.py` | DONE | `test_web.py` |
| `ask_user_question` | packages/interaction/tool-ask-user/src/index.ts:40 | `ask_user_question` | `src/magitech_harness/context/user_questions.py` | DONE | `test_pruner_and_default_model.py` |

## Missing capabilities that must be added to magitech-harness (in scope)

These rows/tools are in DSH base and have no Python implementation. Each needs an executable probe to reach DONE.

1. `job_output` / `job_list` / `job_kill` — `src/magitech_harness/tools/jobs.py` (new) — probe `test_job_tools.py`
7. `ask_user_question` — `src/magitech_harness/tools/ask_user.py` (new) — probe `test_ask_user.py`
10. tool-result pruner (`tool-result-pruner`) — wire into pipeline — probe `test_tool_result_pruner.py`

## Relocation history

ADR 0022 made the harness the owner of every base-bundle tool. The six initial
tools — `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` — were
implemented and probed in `mth-components`, then relocated here with their
tests and mounted by `base.yaml` (`tool-bash`, `tool-fs`, `tool-fs-search`).
mth-components' catalog and docs were updated in the same change; no duplicate
registrations remain.

## PARTIAL rows with accepted reasons

| Row | Reason | Accepted by |
|---|---|---|
| `tool-pwsh` / `pwsh-sandbox` | pwsh is Windows-only; CI runs Linux. No runtime behavior difference to port. | Malcom (platform parity decision) |
| `sandbox` / `sandbox-policy` | Windows ACL sandbox backend not implemented on Linux; `sandbox_policy` enforcement on all paths is a documented gap (finding #13). | Malcom (ratified plan C1) |
| `agent-instructions` | Row not mounted; instruction surface is via system-prompt sections. | Malcom (documented gap) |
| `upstream-skills` | Upstream skills download nothing until registered; no local files to pin. | Malcom (documented gap) |
| `image-offload` | Implemented as a durable message projection + `agent/request-error` retry (row 84 DONE). | Resolved by implementation |
| `commands` | Implemented as a durable registry with lifecycle events (row 59 DONE). | Resolved by implementation |
| `command-feedback` | Implemented as `feedback/record` plus `/feedback` (row 60 DONE). | Resolved by implementation |
| `goal-round-driver` | Automatic continuation rounds are not wired into the loop; entry/complete/block authority is implemented and probed. | **No acceptance recorded** — open gap |

## OUT OF SCOPE (recorded, not counted)

- GUI/web/desktop clients (`dsh-web-app`, `dsh-headless` runner apps, Electron).
- Windows ACL sandbox backend (`dsh-sandbox-windows-acl`).
- External connectors (linear, notion, feishu, dingtalk, wecom, tencent-*).
- Experimental bundles (forges, CUA, terminal, ralph, etc.).
- Plugin manager / config editor / settings / credentials / authorization / storage / otel telemetry as shipped rows (marketplace-owned, opt-in).
- DSH-specific packages (deepseek-account, deepseek-auth, typert, api-gateway, remote-access, session-log-deepseek, deepseek-llm-api-extensions, cli-account, ptc-runtime, workflow-ptc, tool-executor-ptc).
- `tool-workflow` (row 78) and its `workflow` tool: require the out-of-scope `workflow-ptc` JavaScript orchestration runtime.
- DeepSeek account-backed web providers `web-search-deepseek` / `web-deepseek-api` (rows 89–90): require the out-of-scope `deepseek-auth` / account-credential stack; the provider-neutral web seam is ported instead.

## Acceptance summary

- In-scope base-bundle rows: 101 counted; those mapped to Python are DONE/PARTIAL; DSH-specific and marketplace-owned rows are OUT OF SCOPE by rule.
- Tier 1 model-facing tools: DONE for `bash`, `read`, `write`, `edit`, `glob`, `grep`; MISSING for the rest (each named above with a probe).
- Every MISSING row above is the implementation backlog; none is silently dropped.
- PARITY rows are the gate: DONE requires an executable probe; PARTIAL requires a named acceptor.
