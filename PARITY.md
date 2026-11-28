# Parity map — DeepSeek Harness v0.2.3-alpha.2 (pin 4b399c39a6d1479cd84ca591b4d7c88e24d38a96)

This is the authoritative, evidence-backed 1:1 parity map between DeepSeek Harness (dsh) commit
`4b399c39a6d1479cd84ca591b4d7c88e24d38a96` (master head, 2026-11-22) and the Python stack
`animas` (kernel) / `magitech-harness` (host runtime, turn engine, Tier 1 substrate) /
`mth-components` (official marketplace for approved components outside the default bundle) /
`magitech-ui` (NiceGUI Web and Desktop hosts).

It replaces `fcc0ad46` (0.2.2-rc.1), which is now historical provenance only. Lineage is preserved
and attributed: see `animas/docs/adr/0012-lineage-disclosure.md`, `animas/NOTICE`, and
`agents-home` decision D-025. Nothing is scrubbed.

[ADR 0025](docs/adr/0025-full-surface-parity-and-the-divergence-gate.md) expands the target from the
base bundle to the whole harness surface and sets the gate: coverage plus behavioral equivalence
against pinned reference fixtures. Divergence starts only when that gate passes.

## Verdict vocabulary

- **DONE** — reference shape implemented and proven by an executable probe (test or CLI probe). Rows
  added under ADR 0025 additionally require behavioral equivalence against a pinned fixture.
- **PARTIAL** — implemented with a recorded gap, **or** in scope for a later phase and deferred with a
  named acceptor and phase. The reason names who accepted it (Malcom by name, or a pinned row), never
  an agent speaking for Malcom. A deferred row stays on the map and counts against full parity until closed.
- **DIVERGED** — a ratified decision is not met; DIVERGED in-scope fails the gate until fixed or accepted.
- **MISSING** — in-scope capability absent. Fails the gate until implemented and probed.
- **OUT OF SCOPE** — reserved for explicit user exclusions under ADR 0025: the five non-Notion
  connectors, the Windows ACL backend, Electron-specific packaging, the desktop release publisher,
  and unselected experimental bundles. Each names its acceptor. Nothing is silently dropped.

## Repository ownership rule (mechanical)

> Any tool in DSH's `packages/bundle/base/cordis.patch.yml` belongs in `magitech-harness`. Optional,
> on-demand and third-party tools belong in `mth-components`. Kernel semantics (events, effects,
> scopes, platform entry conditions) belong in `animas`. Application hosts (Web, Desktop) belong in
> `magitech-ui`, which depends on the harness and is never imported by it (ADR 0025).

The DSH base bundle at the pin has **101 rows**. Each Phase 0 row maps to exactly one owning repo and one
Python implementation location (or MISSING). Phase 1 adds the application and non-base surfaces that
ADR 0025 brought into scope. DSH file:line citations point at the pinned clone under
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
| 7 | `!!js process.env.DSH_PERMISSION_MODE ?? 'workspace-write'` | `mode: ${MAGITECH_PERMISSION_MODE:workspace-write}` | base.yaml `sandbox-policy` |
| 8 | `!!js process.cwd()` | `workspaceRoot: ${PWD}` | base.yaml `sandbox-policy` |
| 9 | `!!js process.platform === 'win32'` | `disabled_on: [ win32 ]` | base.yaml `tool-bash`, `bash-sandbox` |
| 10| `!!js process.platform !== 'win32'` | `enabled_on: [ win32 ]` for `pwsh-local`; `tool-pwsh` is mounted unconditionally so Linux pwsh 7 can run (dsh gates it win32) | base.yaml `pwsh-local`, `tool-pwsh` |
| 11| `!!js "(process.env.DSH_PERMISSION_MODE ?? 'workspace-write') === 'danger-full-access' ? 'never' : 'ask'"` | `policy: ask` (static default). The `never` behaviour is realised per-session through the sandbox policy: a `danger-full-access` session needs no escalation approval, and `approval.set_policy(agent, "never")` is available as a durable override. No global env-driven policy switch exists. | base.yaml `approval`; `approval/service.py`, `sandbox/policy.py` |
| 12| `!!js process.env.DSH_TELEMETRY_DISABLED` (outer config, not base) | `disabled: ${DSH_TELEMETRY_DISABLED:true}` | base.yaml `session-telemetry-otel` |

The translator golden tests pin each expression → declarative row. See `scripts/profile_translator.py`.

## Phase 0 — base-bundle tool rows (101 rows)

Each DSH row: package, DSH source, owning repo, Python implementation, verdict, probe.

| # | DSH id | DSH package | DSH file:line | Owner | Python impl (file:line) | Verdict | Probe |
|---|---|---|---|---|---|---|---|
| 1 | tool-plugin-manager | `@deepseek-ai/dsh-plugin-manager/tools` | packages/boot/plugin-manager/src/tools.ts:92 | mth-components (plugin manager is opt-in infra) | `src/mth_components/tools/plugin_manager/component.py:163` (unconditional `danger-full-access` escalation at :179, eight actions, approval/agent/cancellation guards, nextStep guidance) | DONE | mth-components: `tests/unit/mth_components/tools/plugin_manager/test_component.py` (registration, eight actions, pagination and meta stripping, nextStep guidance, unconditional escalation probe, argument guards), `tests/unit/mth_components/test_index.py` (catalog registration); CI 100% statement+branch coverage |
| 2 | plugin-manager | `@deepseek-ai/dsh-plugin-manager` | packages/boot/plugin-manager/src/index.ts | mth-components | `src/mth_components/plugin_manager/service.py:682` (install/enable/remove with pnpm registry fallback, build approval, GitHub/git transport probes, HMR exclusive apply and `overridden`, snapshot rollback, 0600 operation log, version listings, update/cancel, bundle detail/images, catalog and verified install target at :1094-1112); helpers `operations.py`, `versions.py`, `semver.py`, `compatibility.py`, `diagnostic.py`, `catalog.py`, `bundle_detail.py`, `git_transport.py`, `github_connection.py`, `build_approval.py`, `install_failure.py`, `official_install_target.py` | DONE | mth-components: `tests/unit/mth_components/plugin_manager/` (`test_core.py`, `test_edges.py`, `test_full_service.py`, `test_full_service_edges.py`, `test_full_service_edges2.py`, `test_helper_edges.py`, `test_helper_edges2.py`, `test_helpers.py`, `test_last_coverage.py`, `test_detail_fidelity.py`) probe spec parsing, registry plans, flat compatibility file, exact exemptions, staged install/rollback, git install, idempotent enable/disable/remove, disk-derived changed, stable codes, registry fallback, GitHub/git checks, build approval, version listings, update/cancel, cancellation kill, detail documents/images, and the cancellation watcher; `tests/unit/mth_components/test_index.py` (catalog registration); CI 100% statement+branch coverage |
| 3 | timer | `@deepseek-ai/cordis-plugin-timer` | `vendor/timer/src/index.ts` | animas | `src/animas/timer.py` | DONE | animas: `tests/unit/animas/test_timer.py` (kernel `timeout`, `interval`, `throttle`, `debounce` with fiber-scoped disposal; owned by animas, not the harness) |
| 4 | hmr | `@deepseek-ai/dsh-hmr` | packages/boot/hmr/src/index.ts | mth-components | `src/mth_components/hmr/component.py`, `src/mth_components/hmr/config.py` | DONE | mth-components: `tests/unit/mth_components/hmr/test_component.py` (dsh config mapping, multi-root watching, active ignore patterns, mount/inject), `tests/unit/mth_components/test_index.py` (catalog registration); core `animas.HmrService` multi-root support probed in animas `tests/unit/animas/test_hmr_roots.py` |
| 5 | llm | `@deepseek-ai/dsh-llm` | packages/llm/llm/src/index.ts | magitech-harness | `src/magitech_harness/llm/service.py` | DONE | `tests/unit/magitech_harness/llm/test_service.py` |
| 6 | deepseek-llm-api-extensions | `@deepseek-ai/dsh-deepseek-llm-api-extensions` | packages/llm/deepseek-llm-api-extensions/src/ | mth-components (marketplace phase) | NONE | PARTIAL | DeepSeek-specific extension DEFERRED: DeepSeek LLM API extension; Search/Fetch ship as opt-in `ctx.web` providers on the provider-neutral seam (row 88). Accepted by Malcom 2026-11-28 (ADR 0025). |
| 7 | session | `@deepseek-ai/dsh-session` | packages/core/session/src/index.ts | magitech-harness | `src/magitech_harness/session/service.py` | DONE | `tests/unit/magitech_harness/session/test_service.py` |
| 8 | session-log-deepseek | `@deepseek-ai/dsh-session-log-deepseek` | packages/session/session-log-deepseek/src/ | mth-components (marketplace phase) | NONE | PARTIAL | DSH-specific log contributor DEFERRED: DeepSeek log contributor tied to the Platform account stack. Accepted by Malcom 2026-11-28 (ADR 0025). |
| 9 | typert | `@deepseek-ai/dsh-typert-registry` | packages/typert/registry/src/ | magitech-harness (capability equivalent) | NONE | PARTIAL | type graph runtime SUBSTITUTION (decision 4): the type graph is a TypeScript/Cordis artifact; the host-facing API is an explicit versioned Python surface. Accepted by Malcom 2026-11-28 (ADR 0025). |
| 10 | typert-loader | `@deepseek-ai/dsh-typert-loader` | packages/typert/loader/src/ | magitech-harness (capability equivalent) | NONE | PARTIAL | SUBSTITUTION (decision 4): TypeScript build-time schema generation has no Python counterpart; controller behavior is the parity target. Accepted by Malcom 2026-11-28 (ADR 0025). |
| 11 | typert-gateway | `@deepseek-ai/dsh-api-gateway` | packages/api/gateway/src/ | magitech-harness (capability equivalent) | NONE | PARTIAL | API gateway SUBSTITUTION (decision 4): typed RPC descriptors and stream multiplexing collapse into the Python host API and NiceGUI transport. Accepted by Malcom 2026-11-28 (ADR 0025). |
| 12 | remote-access | `@deepseek-ai/dsh-api-remote-access` | packages/api/remote-access/src/ | magitech-harness (capability equivalent) | NONE | PARTIAL | remote access The single-owner fail-closed principal seam is the gate; the liveshare role model is a post-parity marketplace component (decision 4). Accepted by Malcom 2026-11-28 (ADR 0025). |
| 13 | session-title | `@deepseek-ai/dsh-session-title` | packages/session/session-title/src/index.ts | magitech-harness | `src/magitech_harness/session/title.py` | DONE | `tests/unit/magitech_harness/session/test_title.py` |
| 14 | session-title-llm | `@deepseek-ai/dsh-session-title-first-prompt-llm` | packages/session/session-title-first-prompt-llm/src/index.ts | magitech-harness | `src/magitech_harness/session/title_first_prompt.py` | DONE | `tests/unit/magitech_harness/session/test_title_first_prompt.py` |
| 15 | user-questions | `@deepseek-ai/dsh-user-questions` | packages/interaction/user-questions/src/index.ts | magitech-harness | `src/magitech_harness/context/user_questions.py` | DONE | `tests/unit/magitech_harness/compaction/test_pruner_and_default_model.py` |
| 16 | agent | `@deepseek-ai/dsh-agent` | packages/core/agent/src/index.ts | magitech-harness | `src/magitech_harness/agent/registry.py` | DONE | `tests/unit/magitech_harness/agent/test_registry.py` |
| 17 | plugin-package-inventory-deepseek | `@deepseek-ai/dsh-plugin-package-inventory-deepseek` | packages/llm/plugin-package-inventory-deepseek/src/ | mth-components (marketplace phase) | NONE | PARTIAL | DEFERRED: DeepSeek-specific plugin inventory. Accepted by Malcom 2026-11-28 (ADR 0025). |
| 18 | agent-default-model | `@deepseek-ai/dsh-agent-default-model` | packages/core/agent-default-model/src/index.ts | magitech-harness | `src/magitech_harness/agent/default_model.py` | DONE | `tests/unit/magitech_harness/compaction/test_pruner_and_default_model.py`; base row defaults to `kilo`/`kilo-auto/free`, verified live with `cost: 0` and no API key |
| 19 | jobs | `@deepseek-ai/dsh-jobs-local` | packages/jobs/jobs-local/src/index.ts | magitech-harness | `src/magitech_harness/jobs/local.py` | DONE | `tests/unit/magitech_harness/jobs/test_local.py` |
| 20 | llm-retry | `@deepseek-ai/dsh-llm-retry` | packages/llm/llm-retry/src/index.ts | magitech-harness | `src/magitech_harness/llm/retry.py` | DONE | `tests/unit/magitech_harness/llm/test_retry.py`, `tests/unit/magitech_harness/llm/test_retry_policy.py` |
| 21 | config-editor | `@deepseek-ai/dsh-config-editor` | packages/boot/config-editor/src/index.ts | mth-components | `src/mth_components/config_editor/service.py` | DONE | mth-components: `tests/unit/mth_components/config_editor/test_component.py` (whole-config edit, inherited layers, atomic 0600 patch write, rollback, HMR serialization, fiber/overlay guards), `tests/unit/mth_components/test_index.py` (catalog registration) |
| 22 | settings | `@deepseek-ai/dsh-settings` | packages/settings/settings/src/index.ts | mth-components | `src/mth_components/settings/service.py` | DONE | mth-components: `tests/unit/mth_components/settings/test_service.py` (secret redaction with concrete paths, defaults/projection, JSON-shape validation, array path ops, stale-write conflict, nested non-form config preservation, configurable autoGenerate), `tests/unit/mth_components/test_index.py` (catalog registration) |
| 23 | authorization | `@deepseek-ai/dsh-authorization` | packages/credentials/authorization/src/index.ts | mth-components | `src/mth_components/authorization/service.py` | DONE | mth-components: `tests/unit/mth_components/authorization/test_service.py` (flow registry, one attempt per key, validation, commit contract, cancellation/decline settlements, contained listeners, task/exception lifecycle), `tests/unit/mth_components/test_index.py` (catalog registration) |
| 24 | deepseek-account | `@deepseek-ai/dsh-deepseek-account` | packages/account/deepseek-account/src/index.ts | mth-components (marketplace phase) | NONE | PARTIAL | DeepSeek account DEFERRED: DeepSeek account sign-in; the UI account surface uses the abstract credential seam (row 26). Accepted by Malcom 2026-11-28 (ADR 0025). |
| 25 | deepseek-auth | `@deepseek-ai/dsh-deepseek-auth` | packages/account/deepseek-auth/src/index.ts | mth-components (marketplace phase) | NONE | PARTIAL | DEFERRED: DeepSeek auth; the API-key provider is ported (row 100). Accepted by Malcom 2026-11-28 (ADR 0025). |
| 26 | credentials | `@deepseek-ai/dsh-credentials-local` | packages/credentials/credentials-local/src/index.ts | mth-components | `src/mth_components/credentials_local/provider.py` | DONE | mth-components: `tests/unit/mth_components/credentials_local/test_provider.py` (versioned YAML, strict refs/records validation, env precedence, owner-only 0600/0700 atomic writes, foreign-key preservation, write-side validation, service hydration/lifecycle); harness: `tests/unit/magitech_harness/credentials/test_engine.py` (provider seam). Cross-process writer lock, hot-reload watcher and comment-preserving YAML edits remain unported and are documented in the component docstring. |
| 27 | llm-pi-ai | `@deepseek-ai/dsh-llm-pi-ai` | packages/llm/llm-pi-ai/src/index.ts | magitech-harness (ADR 0012 Tier 1) | `src/magitech_harness/llm/pi_ai/component.py` | DONE | `tests/unit/magitech_harness/llm/pi_ai/test_pi_ai_component.py` |
| 28 | session-persistence-jsonl | `@deepseek-ai/dsh-session-persistence-jsonl` | packages/session/session-persistence-jsonl/src/index.ts | magitech-harness | `src/magitech_harness/session/persistence_jsonl.py` | DONE | `tests/unit/magitech_harness/session/test_persistence_jsonl.py` |
| 29 | attachment-local | `@deepseek-ai/dsh-attachment-local` | packages/attachment/attachment-local/src/index.ts | mth-components | `src/mth_components/attachment_local/service.py` | DONE | mth-components: `tests/unit/mth_components/attachment_local/test_attachment_local.py`, `test_edges.py`, `test_edges2.py` (canonical base64 admission and batch limits, EXIF-aware image facts derived from mode, pre-decode pixel/dimension limits, normalization with quality ladder and alpha-preserving WebP, staged content-addressed publish with digest verification, EEXIST dedup and directory fsync, verbatim file streams without full retention, deterministic request-image variants with shrink loop and verified cache, config validation), `tests/unit/mth_components/test_index.py` (catalog registration) |
| 30 | session-query-sqlite | `@deepseek-ai/dsh-session-query-sqlite` | packages/session-query/session-query-sqlite/src/index.ts | mth-components | `src/mth_components/sessions/search/component.py` | DONE | mth-components: `tests/unit/mth_components/sessions/search/test_service.py` |
| 31 | working-directory | `@deepseek-ai/dsh-working-directory` | packages/session/working-directory/src/index.ts | magitech-harness | `src/magitech_harness/session/working_directory.py` | DONE | `tests/unit/magitech_harness/session/test_working_directory.py` |
| 32 | session-projection | `@deepseek-ai/dsh-session-projection` | packages/session/session-projection/src/index.ts | magitech-harness | `src/magitech_harness/session/projection_registry.py` | DONE | `tests/unit/magitech_harness/session/test_projection_registry.py` |
| 33 | storage | `@deepseek-ai/dsh-storage` | packages/storage/storage/src/index.ts | mth-components | `src/mth_components/storage/service.py` | DONE | mth-components: `tests/unit/mth_components/storage/test_service.py`, `tests/unit/mth_components/storage/test_component.py` (named backend registry, mountable forms, dsh error codes, stale-disposer guards), `tests/unit/mth_components/test_index.py` (catalog registration) |
| 34 | storage-json | `@deepseek-ai/dsh-storage-json` | packages/storage/storage-json/src/index.ts | mth-components | `src/mth_components/storage_json/backend.py` | DONE | mth-components: `tests/unit/mth_components/storage_json/test_backend.py` (both layouts, atomic publish, rollback, quarantine, legacy bootstrap, private modes, lifecycle) and `tests/unit/mth_components/test_index.py` (catalog registration) |
| 35 | storage-domain | `@deepseek-ai/dsh-storage-domain` | packages/storage/storage-domain/src/index.ts | mth-components | `src/mth_components/storage_domain/facility.py` | DONE | mth-components: `tests/unit/mth_components/storage_domain/test_facility.py` (spec validation, routing, record/global validation, backup-and-skip, write chain, events, quarantine, lifecycle), `tests/unit/mth_components/test_index.py` (catalog registration) |
| 36 | session-projection-cache | `@deepseek-ai/dsh-session-projection-cache` | packages/session/session-projection-cache/src/index.ts | magitech-harness | `src/magitech_harness/session/projection_cache.py` | DONE | `tests/unit/magitech_harness/session/test_projection_cache.py` |
| 37 | otel | `@deepseek-ai/dsh-otel` | packages/telemetry/otel/src/index.ts | mth-components | `src/mth_components/otel/service.py` | DONE | mth-components: `tests/unit/mth_components/otel/test_otel.py` (byte/count batching, single-record rejection, one in-flight request, watchdog, stopPending, bounded shutdown, limit validation, OTLP JSON wire shape, severity enum, retry/backoff, response ceiling, per-request headers, cancellation, event queue bound and export timeout), `tests/unit/mth_components/test_index.py` (catalog registration) |
| 38 | session-telemetry-otel | `@deepseek-ai/dsh-session-telemetry-otel` | packages/session/session-telemetry-otel/src/index.ts | mth-components | `src/mth_components/session_telemetry_otel/backend.py` | DONE | mth-components: `tests/unit/mth_components/session_telemetry_otel/test_backend.py` (mode/config validation, canonical feedback authorization, on-demand capture with handoff cursor, redaction waterfall, severity mapping, anonymous id, shutdown deadline and teardown effect, inspection restore, duplicate-load guard, no-op direct emit), `tests/unit/mth_components/test_index.py` (catalog registration) |
| 39 | subprocess | `@deepseek-ai/dsh-subprocess-local` | packages/subprocess/subprocess-local/src/index.ts | magitech-harness | `src/magitech_harness/subprocess/service.py` | DONE | `tests/unit/magitech_harness/subprocess/test_local.py` |
| 40 | cli-account | `@deepseek-ai/dsh-cli-account` | packages/connector/cli-account/src/index.ts | mth-components (marketplace phase) | NONE | PARTIAL | DEFERRED: CLI-account connector. Accepted by Malcom 2026-11-28 (ADR 0025). |
| 41 | sandbox | `@deepseek-ai/dsh-sandbox-local` | packages/sandbox/sandbox-local/src/index.ts | magitech-harness | `src/magitech_harness/sandbox/service.py` | DONE | `tests/unit/magitech_harness/sandbox/test_sandbox.py` (fail-closed `confine`, bwrap backend, denial/runner signatures, full enforcement) |
| 42 | sandbox-policy | `@deepseek-ai/dsh-sandbox-policy` | packages/sandbox/sandbox-policy/src/index.ts | magitech-harness | `src/magitech_harness/sandbox/policy.py` | DONE | `tests/unit/magitech_harness/sandbox/test_sandbox.py` (three modes, session `sandbox/mode` projection, precedence, writable roots, escalation) |
| 43 | bash-sandbox | `@deepseek-ai/dsh-bash-sandbox` | packages/shell/bash-sandbox/src/index.ts | magitech-harness | `src/magitech_harness/shell/bash_sandbox.py` | DONE | `tests/unit/magitech_harness/sandbox/test_sandbox.py` (real bwrap confinement, denial facts, runner-failure taxonomy, model-facing escalation; closes finding #13) |
| 44 | pwsh-sandbox | `@deepseek-ai/dsh-pwsh-sandbox` | packages/shell/pwsh-sandbox/src/index.ts | magitech-harness | `src/magitech_harness/tools/pwsh_tool.py` | DONE | `tests/unit/magitech_harness/tools/test_pwsh_tool.py` (argv confined through `ctx.sandbox`, bwrap backend, denial marker) |
| 45 | approval | `@deepseek-ai/dsh-user-approval` | packages/interaction/user-approval/src/index.ts | magitech-harness | `src/magitech_harness/approval/service.py` | DONE | `tests/unit/magitech_harness/approval/test_service.py` |
| 46 | permission | `@deepseek-ai/dsh-permission-presets` | packages/interaction/permission-presets/src/index.ts | magitech-harness | `src/magitech_harness/permission/service.py` | DONE | `tests/unit/magitech_harness/permission/test_service.py` (catalog, `permissions` projection, `/permission` command, knob writes, Auto hook, initial pinning; row disabled on win32 because the harness has no confining pwsh executor) |
| 47 | shell-env | `@deepseek-ai/dsh-shell-env` | packages/shell/shell-env/src/index.ts | magitech-harness | `src/magitech_harness/shell_env/service.py` | DONE | `tests/unit/magitech_harness/shell_env/test_service.py` |
| 48 | tool-bash | `@deepseek-ai/dsh-tool-bash` | packages/shell/tool-bash/src/index.ts:415 | magitech-harness (base tool) | `src/magitech_harness/tools/bash_tool.py` | DONE | `tests/unit/magitech_harness/tools/test_bash_tool.py` |
| 49 | tool-pwsh | `@deepseek-ai/dsh-tool-pwsh` | packages/shell/tool-pwsh/src/index.ts | magitech-harness | `src/magitech_harness/tools/pwsh_tool.py` | DONE | `tests/unit/magitech_harness/tools/test_pwsh_tool.py` (PowerShell dialect, `-Command` execution via local pwsh 7, timeout/spill/exit markers) |
| 50 | tool-jobs | `@deepseek-ai/dsh-tool-jobs` | packages/jobs/tool-jobs/src/index.ts:435 | magitech-harness (base tool) | `src/magitech_harness/tools/jobs.py` | DONE | `tests/unit/magitech_harness/tools/test_jobs.py` |
| 51 | fs-observation-policy | `@deepseek-ai/dsh-fs-observation-policy` | packages/fs/fs-observation-policy/src/index.ts | magitech-harness | `src/magitech_harness/fs/observation_policy.py` | DONE | `tests/unit/magitech_harness/fs/test_observation_policy.py` |
| 52 | tool-fs | `@deepseek-ai/dsh-tool-fs` | packages/fs/tool-fs/src/index.ts | magitech-harness (base tool) | `src/magitech_harness/tools/fs.py`, `src/magitech_harness/tools/fs_apply_patch.py` | DONE | `tests/unit/magitech_harness/tools/test_fs_read.py`, `tests/unit/magitech_harness/tools/test_fs_write.py`, `tests/unit/magitech_harness/tools/test_fs_edit.py`, `tests/unit/magitech_harness/tools/test_fs_apply_patch.py`, `tests/unit/magitech_harness/sandbox/test_sandbox.py` (tool-layer read-only fence matches the provider boundary; `apply_patch` is ported and config-gated, base leaves it off exactly as pinned dsh; full preflight/commit result strings, grouping, partial-move and first-op refusal paths are probed; parser and hunk applier are pure ports; observation intents, escalation schema fields, scope-aware guidance, canonical display paths, cancellation/unknown outcomes, affectedPaths/output schema, typed error codes/hunkLine and diff presentation metadata remain unported) |
| 53 | tool-fs-search | `@deepseek-ai/dsh-tool-fs-search` | packages/fs/tool-fs-search/src/index.ts | magitech-harness (base tool) | `src/magitech_harness/tools/fs_search.py` | DONE | `tests/unit/magitech_harness/tools/test_fs_search.py` |
| 54 | agent-instructions | `@deepseek-ai/dsh-agent-instructions` | packages/context/agent-instructions/src/index.ts | magitech-harness | `src/magitech_harness/agent_instructions/` | DONE | `tests/unit/magitech_harness/agent_instructions/test_agent_instructions.py` (AGENTS.md/CLAUDE.md discovery, two-chain budget, durable typed user messages, resume identity) |
| 55 | skill | `@deepseek-ai/dsh-skill` | packages/skill/skill/src/index.ts | mth-components | `src/mth_components/bridges/skills_registry/component.py` | DONE | mth-components: `tests/unit/mth_components/bridges/skills_registry/test_service.py` |
| 56 | skill-filesystem | `@deepseek-ai/dsh-skill-filesystem` | packages/skill/skill-filesystem/src/index.ts | mth-components | `src/mth_components/bridges/skills_filesystem/component.py` | DONE | mth-components: `tests/unit/mth_components/bridges/skills_filesystem/test_service.py` |
| 57 | tool-skill | `@deepseek-ai/dsh-tool-skill` | packages/skill/tool-skill/src/index.ts | mth-components | `src/mth_components/bridges/skills_loader/component.py` | DONE | mth-components: `tests/unit/mth_components/bridges/skills_loader/test_service.py` |
| 58 | upstream-skills | `@deepseek-ai/dsh-upstream-skills` | packages/connector/upstream-skills/src/index.ts | mth-components | `src/mth_components/upstream_skills/` | DONE | mth-components: `tests/unit/mth_components/test_upstream_skills.py` (verified pinned tarball download, SHA-256, atomic publish, registry integration) |
| 59 | commands | `@deepseek-ai/dsh-commands` | packages/interaction/commands/src/index.ts | magitech-harness | `src/magitech_harness/commands/` | DONE | `tests/unit/magitech_harness/commands/test_commands.py` (durable registry, alias resolution, `command/run`/`command/done` lifecycle, REPL wired through the registry; dsh's `zh` built-in tokens are intentionally omitted by project policy, and `parseCommand` cannot parse them anyway) |
| 60 | command-feedback | `@deepseek-ai/dsh-command-feedback` | packages/feedback/command-feedback/src/index.ts | magitech-harness | `src/magitech_harness/feedback/` | DONE | `tests/unit/magitech_harness/commands/test_commands.py` (`feedback/record` event, `sessionFeedback` service, `/feedback` command) |
| 61 | goal | `@deepseek-ai/dsh-goal` | packages/goal/goal/src/index.ts | magitech-harness (base tool) | `src/magitech_harness/goal/{types,projection,service}.py` | DONE | `tests/unit/magitech_harness/goal/test_goal.py` |
| 62 | goal-round-driver | `@deepseek-ai/dsh-goal-round-driver` | packages/goal/goal-round-driver/src/index.ts | magitech-harness | `src/magitech_harness/goal/round_driver.py` | DONE | `tests/unit/magitech_harness/goal/test_round_driver.py` (serialized driver, quiescence gates, round-limit block, checkpoint/subagent fences, disarm) |
| 63 | command-goal | `@deepseek-ai/dsh-command-goal` | packages/goal/command-goal/src/index.ts | magitech-harness | `src/magitech_harness/goal/command.py` | DONE | `tests/unit/magitech_harness/goal/test_goal.py`, `tests/unit/magitech_harness/test_cli.py` |
| 64 | plan-mode | `@deepseek-ai/dsh-plan-mode` | packages/plan/plan-mode/src/index.ts | magitech-harness (base tool) | `src/magitech_harness/plan/` | DONE | `tests/unit/magitech_harness/plan/test_plan.py` (durable `plan` projection, step-boundary admission, conditional `exit_plan_mode` with user review) |
| 65 | token-meter | `@deepseek-ai/dsh-token-meter` | packages/llm/token-meter/src/index.ts | magitech-harness | `src/magitech_harness/llm/token_meter.py` | DONE | `tests/unit/magitech_harness/llm/test_token_meter.py` |
| 66 | compaction-basic | `@deepseek-ai/dsh-compaction-basic` | packages/compaction/compaction-basic/src/index.ts | magitech-harness | `src/magitech_harness/compaction/service.py` | DONE | `tests/unit/magitech_harness/compaction/test_service.py` |
| 67 | command-compact | `@deepseek-ai/dsh-command-compact` | packages/compaction/command-compact/src/index.ts | magitech-harness | `src/magitech_harness/cli.py` (`/compact` → `session/compact` waterfall) | DONE | `tests/unit/magitech_harness/test_cli.py` |
| 68 | subagent | `@deepseek-ai/dsh-subagent` | packages/core/subagent/src/index.ts | magitech-harness | `src/magitech_harness/subagent/service.py` | DONE | `tests/unit/magitech_harness/subagent/test_manager.py`, `tests/unit/magitech_harness/subagent/test_surface.py`; lineage logged as `subagent/descriptor` |
| 69 | agent-team | `@deepseek-ai/dsh-agent-team` | packages/team/agent-team/src/index.ts | magitech-harness | `src/magitech_harness/team/` | DONE | `tests/unit/magitech_harness/team/test_team.py` (event-sourced board, CAS revisions, dependency graph, readiness/write-scope views) |
| 70 | tool-agent-team | `@deepseek-ai/dsh-tool-agent-team` | packages/team/tool-agent-team/src/index.ts:119 | magitech-harness (base tool) | `src/magitech_harness/team/tools.py` | DONE | `tests/unit/magitech_harness/team/test_team.py` (4 tools, conditional admission, Team policy section) |
| 71 | subagent-spawn-in-process | `@deepseek-ai/dsh-subagent-spawn-in-process` | packages/subagent/subagent-spawn-in-process/src/index.ts | magitech-harness | `src/magitech_harness/subagent/providers.py` | DONE | `tests/unit/magitech_harness/subagent/test_surface.py` |
| 72 | subagent-fork-in-process | `@deepseek-ai/dsh-subagent-fork-in-process` | packages/subagent/subagent-fork-in-process/src/index.ts | magitech-harness | `src/magitech_harness/subagent/providers.py` | DONE | `tests/unit/magitech_harness/subagent/test_surface.py` (replays parent events with surface ops) |
| 73 | tool-subagent-control | `@deepseek-ai/dsh-tool-subagent-control` | packages/subagent/tool-subagent-control/src/index.ts:45 | magitech-harness (base tool) | `src/magitech_harness/subagent/tools.py` | DONE | `tests/unit/magitech_harness/subagent/test_surface.py`, `tests/unit/magitech_harness/subagent/test_manager.py` |
| 74 | tool-subagent | `@deepseek-ai/dsh-tool-subagent` | packages/subagent/tool-subagent/src/index.ts:103 | magitech-harness (base tool) | `src/magitech_harness/subagent/tools.py` | DONE | `tests/unit/magitech_harness/subagent/test_surface.py`; dsh tool name `subagent` plus `list_subagent_models` |
| 75 | tool-subagent-fork | `@deepseek-ai/dsh-tool-subagent` (fork) | packages/subagent/tool-subagent/src/index.ts:103 | magitech-harness (base tool) | `src/magitech_harness/subagent/tools.py` | DONE | `tests/unit/magitech_harness/subagent/test_manager.py`; tool name `subagent_fork` |
| 76 | ptc-runtime | `@deepseek-ai/dsh-ptc-runtime-node` | packages/ptc-runtime/ptc-runtime-node/src/ | magitech-harness | NONE | MISSING | PTC Node runtime IN SCOPE FOR THE GATE (decision 5): sandboxed JS runtime driven from Python; not ported. |
| 77 | workflow-ptc | `@deepseek-ai/dsh-workflow-ptc` | packages/workflow/workflow-ptc/src/ | magitech-harness | NONE | MISSING | IN SCOPE FOR THE GATE (decision 5): workflow engine with JS/TS script authorship; not ported. |
| 78 | tool-workflow | `@deepseek-ai/dsh-tool-workflow` | packages/workflow/tool-workflow/src/index.ts | magitech-harness | NONE | MISSING | The `workflow` tool executes JavaScript orchestration scripts through `@deepseek-ai/dsh-workflow-ptc` (row 77). IN SCOPE FOR THE GATE (decision 5): `workflow` tool name, schema, hooks and envelope are model-facing; with no Python-driven JS engine yet, the tool is absent from the catalog. |
| 79 | timeout-policy | `@deepseek-ai/dsh-tool-call-timeout-policy` | packages/guard/timeout-policy/src/index.ts | magitech-harness | `src/magitech_harness/tools/timeout_guard.py` | DONE | `tests/integration/test_loop_hygiene_guards.py` |
| 80 | spill-local | `@deepseek-ai/dsh-spill-local` | packages/spill/spill-local/src/index.ts | magitech-harness | `src/magitech_harness/spill/local.py` | DONE | `tests/unit/magitech_harness/spill/test_local.py` |
| 81 | spill-policy | `@deepseek-ai/dsh-spill-policy` | packages/spill/spill-policy/src/index.ts | magitech-harness | `src/magitech_harness/spill/policy.py` | DONE | `tests/unit/magitech_harness/spill/test_policy.py` |
| 82 | session-checkpoint-policy | `@deepseek-ai/dsh-session-checkpoint-policy` | packages/session/session-checkpoint-policy/src/index.ts | magitech-harness | `src/magitech_harness/session/checkpoint_policy.py` | DONE | `tests/unit/magitech_harness/session/test_checkpoint_policy.py` |
| 83 | tool-result-pruner | `@deepseek-ai/dsh-compaction-tool-result-pruner` | packages/compaction/compaction-tool-result-pruner/src/index.ts | magitech-harness | `src/magitech_harness/compaction/pruner.py` | DONE | `tests/unit/magitech_harness/compaction/test_pruner_and_default_model.py` |
| 84 | image-offload | `@deepseek-ai/dsh-compaction-image-offload` | packages/compaction/compaction-image-offload/src/index.ts | magitech-harness | `src/magitech_harness/compaction/image_offload.py` | DONE | `tests/unit/magitech_harness/compaction/test_image_offload.py` (durable projection, oldest-image selection, `agent/request-error` retry wiring) |
| 85 | tool-todo | `@deepseek-ai/dsh-tool-todo` | packages/todo/tool-todo/src/index.ts:137 | magitech-harness (base tool) | `src/magitech_harness/tools/todo.py` | DONE | `tests/unit/magitech_harness/tools/test_todo.py` |
| 86 | tool-goal | `@deepseek-ai/dsh-tool-goal` | packages/goal/tool-goal/src/index.ts:197 | magitech-harness (base tool) | `src/magitech_harness/goal/tools.py` | DONE | `tests/unit/magitech_harness/goal/test_goal.py` |
| 87 | repeat-tool-reminder | `@deepseek-ai/dsh-repeat-tool-reminder` | packages/guard/repeat-tool-reminder/src/index.ts | magitech-harness | `src/magitech_harness/tools/repeat_guard.py` | DONE | `tests/integration/test_loop_hygiene_guards.py` |
| 88 | web | `@deepseek-ai/dsh-web` | packages/web/web/src/index.ts | magitech-harness | `src/magitech_harness/web/service.py` | DONE | `tests/unit/magitech_harness/web/test_web.py` (provider protocol, route selection, duplicate-id rejection, `list_providers`, `web/providers-changed`, `WEB_*` taxonomy, central result cap). Typert Remote RPC wrapper is out of scope (row 12). |
| 89 | web-search-deepseek | `@deepseek-ai/dsh-web-search-deepseek` | packages/web/web-search-deepseek/src/index.ts | mth-components (marketplace phase) | NONE | PARTIAL | DeepSeek account-sign-in provider: requires `dsh-deepseek-auth`, `dsh-credentials` and `dsh-launch-environment`, none of which exist in this harness (default route is Kilo). The provider-neutral `ctx.web` seam is ported (row 88) and accepts any registered search provider. DEFERRED: account-backed search provider; provider-neutral seam ported (row 88). Accepted by Malcom 2026-11-28 (ADR 0025). |
| 90 | web-deepseek-api | `@deepseek-ai/dsh-web-deepseek-api` | packages/web/web-deepseek-api/src/index.ts | mth-components (marketplace phase) | NONE | PARTIAL | DeepSeek Search/Fetch API provider with the same `dsh-deepseek-auth` account-credential dependency as row 89; no equivalent account service in this harness. DEFERRED: DeepSeek Search/Fetch API provider as an opt-in web provider. Accepted by Malcom 2026-11-28 (ADR 0025). |
| 91 | web-fetch-http | `@deepseek-ai/dsh-web-fetch-http` | packages/web/web-fetch-http/src/index.ts | magitech-harness | `src/magitech_harness/web/service.py` | DONE | `tests/unit/magitech_harness/web/test_web.py` (byte/char caps, content-type classification, charset, same-origin redirects, bounded timeout, User-Agent) |
| 92 | tool-web | `@deepseek-ai/dsh-tool-web` | packages/web/tool-web/src/search.ts:367 | magitech-harness (base tool) | `src/magitech_harness/web/tools.py` | DONE | `tests/unit/magitech_harness/web/test_web.py` (`web_search`, `web_fetch`; route-aware `when` admission, untrusted-data guidance) |
| 93 | mcp-resources | `@deepseek-ai/dsh-mcp-resources` | packages/mcp/mcp-resources/src/index.ts | mth-components | `src/mth_components/bridges/mcp/component.py` | DONE | mth-components: `tests/unit/mth_components/bridges/mcp/test_component.py`, `tests/unit/mth_components/bridges/mcp/test_protocol.py` |
| 94 | tools | `@deepseek-ai/dsh-tools` | packages/core/tools/src/index.ts | magitech-harness | `src/magitech_harness/tools/service.py` | DONE | `tests/unit/magitech_harness/tools/test_service.py` |
| 95 | tool-executor-native | `@deepseek-ai/dsh-tool-executor-native` | packages/core/tool-executor-native/src/index.ts | magitech-harness | `src/magitech_harness/tools/native_executor.py` | DONE | `tests/unit/magitech_harness/tools/test_native_executor.py` |
| 96 | tool-executor-ptc | `@deepseek-ai/dsh-tool-executor-ptc` | packages/core/tool-executor-ptc/src/index.ts | magitech-harness | NONE | MISSING | PTC executor IN SCOPE FOR THE GATE (decision 5): PTC `run_code` executor with JS/TS script authorship; not ported. |
| 97 | system-prompt | `@deepseek-ai/dsh-system-prompt` | packages/core/system-prompt/src/index.ts | magitech-harness | `src/magitech_harness/system_prompt/service.py` | DONE | `tests/unit/magitech_harness/agent/test_system_prompt.py` |
| 98 | agent-loop | `@deepseek-ai/dsh-agent-loop` | packages/core/agent-loop/src/index.ts | magitech-harness | `src/magitech_harness/agent/react_loop.py` | DONE | `tests/unit/magitech_harness/agent/test_agent_loop.py` |
| 99 | fs-sandbox | `@deepseek-ai/dsh-fs-sandbox` | packages/fs/fs-sandbox/src/index.ts | magitech-harness | `src/magitech_harness/fs/local.py` (sandboxed) | DONE | `tests/unit/magitech_harness/fs/test_local.py`, `tests/unit/magitech_harness/sandbox/test_sandbox.py` |
| 100 | llm-deepseek | `@deepseek-ai/dsh-llm-deepseek-api-key` | packages/llm/llm-deepseek-api-key/src/index.ts | magitech-harness (Tier 1 pi-ai) | `src/magitech_harness/llm/pi_ai/` | DONE | `tests/unit/magitech_harness/llm/pi_ai/test_pi_ai_provider_data.py`, `tests/unit/magitech_harness/llm/pi_ai/test_pi_ai_component.py` |
| 101 | llm-deepseek-account | `@deepseek-ai/dsh-llm-deepseek-account` | packages/llm/llm-deepseek-account/src/index.ts | mth-components (marketplace phase) | NONE | PARTIAL | DeepSeek account-token provider: injects `deepseekAuth` and depends on `dsh-deepseek-auth` / `dsh-credentials`, none of which exist in this harness. The API-key DeepSeek provider is ported through `pi_ai` (row 100). DEFERRED: account-token provider; API-key route ported (row 100). Accepted by Malcom 2026-11-28 (ADR 0025). |

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
| `workflow` | packages/workflow/tool-workflow/src/index.ts | — | NONE | MISSING | In scope for the gate: depends on the JS runtime at row 77 (ADR 0025 decision 5). |
| `run_code` | packages/core/tool-executor-ptc/src/index.ts | — | NONE | MISSING | In scope for the gate: PTC executor at row 96 (ADR 0025 decision 5). |
| `web_search` | packages/web/tool-web/src/search.ts:367 | `web_search` | `src/magitech_harness/web/tools.py` | DONE | `test_web.py` |
| `web_fetch` | packages/web/tool-web/src/fetch.ts:460 | `web_fetch` | `src/magitech_harness/web/tools.py` | DONE | `test_web.py` |
| `ask_user_question` | packages/interaction/tool-ask-user/src/index.ts:40 | `ask_user_question` | `src/magitech_harness/context/user_questions.py` | DONE | `test_pruner_and_default_model.py` |

## Phase 1 — application and non-base surfaces (ADR 0025)

These rows are outside the 101 base-bundle rows but inside the expanded parity target. Sources are
the pinned dsh clone; `magitech-ui` is the sibling NiceGUI host package. A MISSING row fails the gate
until it has a Python implementation and a behavioral-equivalence probe.

| # | Surface | DSH source | Owner | Python impl | Verdict | Notes / probe |
|---|---|---|---|---|---|---|
| 102 | CLI launcher | `apps/cli/src/args.ts`, `bin.ts`, `profile-boot.ts`, `dump-config.ts`, `dump-config-schema.ts`, `plugin.ts` | magitech-harness | `src/magitech_harness/cli.py`, `src/magitech_harness/profiles/launcher.py`, `src/magitech_harness/profiles/templates/` | DONE | Named profiles under `$MTH_HOME/profiles`, shipped-template initialization and `--from-default-profile`, three mutually exclusive boot-free dumps, `plugin list/add/remove/show`, `--version`, and an expected-help snapshot. Probes: `tests/unit/magitech_harness/profiles/test_launcher.py` (resolution, template reset, composition, schema, plugin lifecycle) and `tests/unit/magitech_harness/test_cli.py` (`test_cli_config_dumps_are_boot_free`, `test_cli_named_profile_component_and_template_reset`, `test_cli_named_profile_boot_initializes_template`, `test_cli_help_snapshot_and_version`). **[Python]** dumps are `mth-config-dump/v1` JSON rather than commented YAML, and `plugin` edits an explicit marketplace component list instead of forwarding to pnpm |
| 103 | Headless runner | `packages/bundle/headless/src/index.ts`, `startup.ts`, `json-stream.ts` | magitech-harness | `src/magitech_harness/cli.py`, `src/magitech_harness/headless_json.py` | DONE | Task (including stdin and `-`), `--session-id`, and 0/1 exit codes match. `--json` projects byte-bounded NDJSON (`session`, `status`, `thinking`, `text`, `tool_call`, `tool_result`, `final`) from committed events; reasoning goes to stderr. Probes: `tests/unit/magitech_harness/test_headless_json.py` (byte caps, argument parsing, usage accumulation, event mapping) and `tests/unit/magitech_harness/test_cli.py` (`test_cli_headless_json_streams_reasoning_and_final`, `test_cli_headless_reads_task_from_stdin`). **[Python]** diagnostic prefix is `mth:` and cwd comes from the Python process |
| 104 | SDK wire protocol | `packages/sdk/protocol/src/` | magitech-harness | `src/magitech_harness/sdk/protocol.py` | DONE | `JsonRpcLineTransport` implements one JSON-RPC 2.0 frame per newline over caller-owned async streams: request/response/notification classification, malformed-line tolerance, `-32601`/`-32603`, error-code and data preservation, AbortSignal cancellation, and non-destructive close. Named method maps and TypedDict payloads cover the pinned method set. Probe: `tests/unit/magitech_harness/sdk/test_protocol.py` (100% statement/branch coverage) |
| 105 | SDK JSON-RPC server | `packages/sdk/server/src/` | magitech-harness | NONE | MISSING | Stdio serving plugin; probe against the pinned wire fixture |
| 106 | SDK TypeScript client | `packages/sdk/client/src/` | magitech-harness | NONE | MISSING | Python client equivalent; dsh's `python/sdk` is reusable as the conformance client until ported |
| 107 | SDK application bundle | `packages/bundle/sdk-app/` | magitech-harness | NONE | MISSING | Stdio profile lifecycle and bounded shutdown |
| 108 | SDK minimal bundle | `packages/bundle/sdk-minimal/` | magitech-harness | NONE | MISSING | Standalone composition with one adapter, persistent shell, JSONL sessions |
| 109 | Python SDK | `python/sdk/src/` | magitech-harness | NONE | MISSING | Client library; conformance probe drives the mth SDK server |
| 110 | Python runtime carrier | `python/sdk-runtime/src/` | magitech-harness | NONE | MISSING | Packaging of the CLI and native sidecars |
| 111 | ACP server | `packages/acp/acp/src/` | magitech-harness | NONE | MISSING | Standard ACP v1 over stdio: sessions, model selection, MCP attach, prompt/cancel, close |
| 112 | ACP application bundle | `packages/bundle/acp-app/` | magitech-harness | NONE | MISSING | ACP stdio profile lifecycle and bounded shutdown |
| 113 | Session controller | `packages/api/session-controller/src/` | magitech-harness | NONE | MISSING | Host-facing API: create, resume, prompt, history, live state; `prompt/origin` for non-owner callers |
| 114 | Workspace controller | `packages/api/workspace-controller/src/` | magitech-harness | NONE | MISSING | Workspace navigation mutation and projection |
| 115 | Settings controller | `packages/api/settings-controller/src/` | magitech-harness | NONE | MISSING | Redacted settings/credential reads and writes |
| 116 | Job controller | `packages/api/job-controller/src/` | magitech-harness | NONE | MISSING | Job roster, retained output, kill on a human's behalf |
| 117 | Terminal controller | `packages/api/terminal-controller/src/` | magitech-harness | NONE | MISSING | Interactive terminals with bounded screen recovery |
| 118 | Workspace files | `packages/api/workspace-files/src/` | magitech-harness | NONE | MISSING | Bounded `stat`, paged `read`, `list`, and `changes` feed |
| 119 | Remote assembly | `packages/api/remotes/src/` | magitech-harness | NONE | MISSING | Typed capability selection and forwarded events for UI consumers |
| 120 | Web application bundle | `packages/bundle/web-app/` | magitech-ui | NONE | MISSING | Browser GUI composition over the base bundle |
| 121 | Web frontend entry | `apps/web/` | magitech-ui | NONE | MISSING | NiceGUI page tree replaces the Vite SPA |
| 122 | Web client boot kernel | `packages/client/web/src/` | magitech-ui | NONE | MISSING | Two-stage client boot and module table, in Python |
| 123 | Web host server | `packages/host/webserver/src/` | magitech-ui | NONE | MISSING | NiceGUI server replaces the Node HTTP/TLS host |
| 124 | Plugin-manager UI | `packages/client/ui-plugin-manager/src/client/` | magitech-ui | NONE | MISSING | Plugins page backed by the `mth-components` marketplace; Notion is its only connector |
| 125 | Desktop application | `apps/desktop/` | magitech-ui | NONE | MISSING | `mth-desktop` entry point on shared NiceGUI primitives |
| 126 | Desktop host process | `apps/desktop-host/` | magitech-ui | NONE | MISSING | Private Python host process behind the desktop entry point |
| 127 | Desktop authentication | `packages/host/authentication-desktop/src/` | magitech-ui | NONE | MISSING | In-memory credential for host requests, never exposed to the renderer |
| 128 | Liveshare bundle | `packages/experimental/liveshare/` | mth-components | NONE | PARTIAL | Deferred to post-parity marketplace phase: viewer roles and the separately deployed relay server. Accepted by Malcom 2026-11-28 (ADR 0025) |
| 129 | Liveshare protocol | `packages/experimental/liveshare-protocol/src/` | mth-components | NONE | PARTIAL | Deferred with row 128. Accepted by Malcom 2026-11-28 (ADR 0025) |
| 130 | Notion connector | `packages/connector/notion/src/` | mth-components | NONE | PARTIAL | The only connector ported; deferred to the marketplace phase, after the harness reaches parity. Accepted by Malcom 2026-11-28 (ADR 0025) |
| 131 | Plugin-market generator | `apps/plugin-market-generator/` | mth-components | NONE | PARTIAL | Reference index generator for the marketplace; deferred to the marketplace phase. Accepted by Malcom 2026-11-28 (ADR 0025) |
| 132 | Desktop publisher | `apps/desktop-publisher/` | magitech-ui | NONE | OUT OF SCOPE | Release tooling, not a runtime surface. Accepted by Malcom 2026-11-28 (ADR 0025) |
| 133 | Unselected experimental bundles | `packages/experimental/` (58 package dirs) | — | NONE | OUT OF SCOPE | See the explicit-exclusion table below; liveshare is tracked at rows 128–129. Accepted by Malcom 2026-11-28 (ADR 0025) |

`packages/web/tool-web` and `packages/boot/plugin-manager` are base-bundle rows (92, 1, 2) and are
not duplicated here.

## Gate status

The items formerly listed here as missing have all reached DONE with executable probes:
`job_output` / `job_list` / `job_kill` (row 50), `ask_user_question` (row 15), and the tool-result
pruner (row 83). The acceptance summary no longer treats them as open gaps.

ADR 0025 raises the proof standard to behavioral equivalence. Rows admitted under the older
shape-level standard are re-probed where their notes record unported behavior:

- row 52 `tool-fs` — observation intents, escalation schema fields, scope-aware guidance, canonical
  display paths, cancellation/unknown outcomes, `affectedPaths`/output schema, typed error codes with
  `hunkLine`, and diff presentation metadata.
- row 26 `credentials` — cross-process writer lock, hot-reload watcher, and comment-preserving YAML
  edits.

In-scope rows with no Python implementation are recorded as MISSING in Phase 0 (rows 76–78, 96) and
Phase 1. Deferred rows are PARTIAL with a named acceptor and phase in the table below.

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
| `pwsh-sandbox` | PowerShell tool ported and confined through the harness sandbox seam (rows 44 and 49 DONE). Windows ACL backend remains a platform gap. | Resolved by implementation |
| `sandbox` / `sandbox-policy` | Linux bwrap-backed confinement implemented (rows 41–43 DONE) with per-call policy, session mode, approval escalation, shell-executor enforcement and fs-provider/tool enforcement. finding #13 is closed for the shell and fs paths; Windows ACL backend remains a platform gap (row 44). | Resolved by implementation |
| `agent-instructions` | Implemented as a durable instruction loader (row 54 DONE). | Resolved by implementation |
| `upstream-skills` | Implemented as a verified pinned download component (row 58 DONE). | Resolved by implementation |
| `image-offload` | Implemented as a durable message projection + `agent/request-error` retry (row 84 DONE). | Resolved by implementation |
| `commands` | Implemented as a durable registry with lifecycle events (row 59 DONE). | Resolved by implementation |
| `command-feedback` | Implemented as `feedback/record` plus `/feedback` (row 60 DONE). | Resolved by implementation |
| `tool-plugin-manager` | Escalation gate and build-approval arguments/guidance fully mounted in the tool (row 1 DONE). | Resolved by implementation |
| `plugin-manager` | GitHub/git probes, pnpm fallback and build scripts, HMR live-apply/overridden, bundle detail images, version listings and update/cancel fully ported (row 2 DONE). | Resolved by implementation |
| `goal-round-driver` | Automatic continuation rounds are wired into the loop (row 62 DONE). | Resolved by implementation |

## Deferred rows (ADR 0025)

| Rows | Phase | Scope | Accepted by |
|---|---|---|---|
| 6, 8, 17, 24, 25, 40, 89, 90, 101 | marketplace phase | DeepSeek service-coupled extensions and connectors; the base bundle stays provider-neutral | Malcom 2026-11-28 |
| 9, 10, 11, 12 | accepted substitution at the gate | TypeScript/Cordis type and RPC artifacts replaced by the versioned Python host API and the fail-closed single-owner principal seam | Malcom 2026-11-28 |
| 128, 129 | post-parity marketplace | Liveshare viewer roles and the separately deployed peer-sharing server; the gate keeps a single-owner host | Malcom 2026-11-28 |
| 130 | marketplace phase | Notion, the only connector ported; the other five are explicit exclusions | Malcom 2026-11-28 |
| 131 | marketplace phase | Plugin-market index generator for the official marketplace | Malcom 2026-11-28 |

## OUT OF SCOPE (explicit exclusions, with acceptor)

| Exclusion | Reason | Accepted by |
|---|---|---|
| Linear, Feishu, DingTalk, WeCom, Tencent Meeting connectors | Only Notion is ported; the other five are dropped by user decision | Malcom 2026-11-28 |
| Windows ACL sandbox backend (`dsh-sandbox-windows-acl`) | Platform gap; Linux bwrap confinement is the gate (rows 41–44) | Malcom 2026-11-28 |
| Electron-specific packaging (`file://` dist, `app.asar` carrier) | NiceGUI replaces Electron; browser and desktop are both Python hosts | Malcom 2026-11-28 |
| Desktop release publisher (row 132) | Release tooling, not a runtime surface | Malcom 2026-11-28 |
| Experimental bundles not selected by ADR 0025 (row 133; forges, CUA, terminal, ralph, etc.) | Not named in the parity target; promotion requires a new decision | Malcom 2026-11-28 |

## Fidelity refinements

- `runnerCommand` wraps the same bwrap profile args as the built-in backend, advertises dsh's runner denial signatures, and a bounded `probeTimeoutMs` bwrap probe gates every unconfigured backend call (the verdict is cached).
- Policy-context prose, escalation messages, session writable-root keying and the filesystem provider's write boundary follow the pinned dsh wording and ownership.
- The PowerShell tool renders dsh-ordered markers (denial → escalation hint → exit status) with a single `[timed out after Nms]` marker and an `maxTimeoutMs` cap read from the mounted shell configuration.

## Acceptance summary

- Phase 0 covers the 101 base-bundle rows; Phase 1 adds 32 application and non-base rows. The tool
  inventory records model-facing names separately. Current verdict counts across the whole ledger:
  113 DONE, 17 PARTIAL, 29 MISSING, 0 DIVERGED, 2 OUT OF SCOPE.
- Phase 0: rows 9–12 are accepted substitutions (PARTIAL); rows 76–78 and 96 are in scope for the
  gate and MISSING; the marketplace-phase DeepSeek rows are PARTIAL with a named acceptor.
- Phase 1: rows 102–104 are DONE with executable probes (launcher, headless, SDK wire protocol);
  23 rows are MISSING and in scope; rows 128–131 are deferred; rows 132–133 are explicit exclusions.
- Tier 1 model-facing tools: every in-scope name is DONE with a probe except `workflow` and
  `run_code`, which are MISSING pending the JS-runtime gate work.
- The gate (ADR 0025): DONE requires an executable probe and, for rows added under that ADR,
  behavioral equivalence against a pinned fixture. MISSING fails the gate. PARTIAL requires a named
  acceptor and phase. Divergence opens only when the in-scope rows are DONE and the re-probe
  obligations in Gate status are closed.
