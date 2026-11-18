# Config catalog

Exhaustive reference for configuration keys, their shipped defaults, and where
each is set. Source of truth: the mounted rows in
[`profiles/base.yaml`](../src/magitech_harness/profiles/base.yaml) and each
service's config reader. This page is maintained by hand from source; the docs
review checks it against the code. For the resolution mechanisms, see
[configuration](subsystems/configuration.md).

Conventions: **YAML** means a value in `base.yaml`; **code** means a component
or service default with no YAML key. "Required" means the service raises
`ValueError` when the key is absent. Environment interpolation is noted where
the loader supplies the value.

## Profile YAML values

| Row | Key | Default | When absent |
|---|---|---|---|
| `approval` | `policy` | `ask` | code default `ask` |
| `sessions` | `base_storage_dir` | `${MAGITECH_SESSIONS_DIR}` | memory-only sessions |
| `session-persistence-jsonl` | `root` | `${MAGITECH_SESSIONS_DIR}` | hollow persistence service |
| `session-persistence-jsonl` | `live_batch_delay` | `0.2` | `0.2` |
| `session-title` | `fallback_max_words` | `5` | required |
| `session-title` | `fallback_max_bytes` | `40` | required |
| `session-title` | `max_title_bytes` | `80` | required; must be ≥ `fallback_max_bytes` |
| `session-title-first-prompt` | `target_words` | `5` | required |
| `session-title-first-prompt` | `target_cjk_characters` | `10` | required |
| `session-title-first-prompt` | `max_input_bytes` | `4096` | required |
| `session-title-first-prompt` | `max_output_tokens` | `4096` | required |
| `session-title-first-prompt` | `timeout_ms` | `60000` | required |
| `session-title-first-prompt` | `provider`, `model` | unset | must be supplied together as non-empty strings |
| `session-projection-cache` | `root` | `${MAGITECH_SESSIONS_DIR}` | hollow cache |
| `session-projection-cache` | `write_every_events` | `200` | `200` |
| `session-projection-cache` | `write_interval_ms` | `5000` | `5000` |
| `agent-loop` | `agents` | `[]` | `[]` |
| `agent-loop` | `maxParallelToolCalls` | `10` | `10` |
| `repeat-tool-reminder-guard` | `thresholds` | `[3, 5, 8]` | `[3, 5, 8]` |

## Component and code defaults

| Row / service | Key | Default |
|---|---|---|
| `system-prompt` | `tool_order`, `include_harness_identity`, `persona_prefix`, `persona_suffix`, `include_runtime_context` | none, true, empty, empty, true |
| `tool-executor-native` | `maxParallelToolCalls` / `maxParallel` | `10` |
| `tools` | `mode`, `maxParallelSubCalls` | rejected if present |
| `llm` | `provider` | unset (CLI registers `mock`) |
| LLM listing cache | `modelsTimeoutMs`, `modelsMaxAgeMs` | `5000`, `600000` |
| `llm-retry` | any key | rejected; retry config belongs under each provider |
| Retry policy (provider) | `mode`, `maxRetries` | `normal`, `5` |
| Retry policy (provider) | `backoff.initialDelayMs`, `backoff.maxDelayMs`, `backoff.jitterRatio` | `500`, `10000`, `0.1` |
| Retry policy (provider) | `retryableCodes` | empty-response, rate-limit, server, timeout, transport |
| Provider substrate | `timeoutMs`, `streamIdleTimeoutMs` | `300000`, `300000` |
| `fs` | `cwd`, `diff_basis_max_bytes` | process CWD, 10 MiB |
| `subprocess` | `spillDir` | computed default |
| Shell executors | `cwd`, `timeoutMs`, `maxTimeoutMs` | process CWD, `120000`, `600000` |
| Shell executors | `maxOutputBytes`, `maxSpillBytes`, `graceMs` | `64000`, 64 MiB, `3000` |
| Windows shell | `pwshPath` | probed |
| `jobs` | `maxConcurrentJobsPerOwner`, `retainBytes` | `10`, `262144` |
| `jobs` | `settledRetainBytes`, `pumpPollMs` | `16384`, `150` |
| `working-directory` | `defaultDirectory` | process CWD (must be absolute) |
| `approval-guard` | `read_only_tools` | built-in allow-list |
| `repeat-tool-reminder-guard` | `arguments_preview_chars` | `500` |
| `token-meter` | any key | rejected |
| `compaction` guard (plugin) | `max_tokens`, `target_ratio` | `8000`, `0.5` |
| Pruner (plugin, code) | `thresholdChars`, `headChars`, `tailChars` | `8192`, `4096`, `1024` |
| Truncation guard (plugin) | `max_lines`, `max_bytes` | `2000`, `51200` |
| `read_file` (plugin) | `default_limit`, `max_limit`, `workspace_root` | `2000`, `2000`, `ctx.workspace_dir` |
| `bash` tool (plugin, code) | `max_output_bytes` | `64 * 1024` |
| `execute_python` (plugin, code) | `timeout_s` | `30.0` |
| Security sandbox (plugin) | `max_level`, `enforce`, `bwrap_path` | `STRICT`, true, probed |

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `MAGITECH_SESSIONS_DIR` | loader interpolation | Session, persistence, and projection-cache roots; unset → memory-only |
| `MAGITECH_GLOBAL_DIR` | `layers.py` | Overrides `/etc/magitech` as the global layer |
| `MAGITECH_AGENTS_HOME` | `layers.py` | Overrides `~/.agents` as the shared agents root |
| `MAGITECH_HOME`, `MAGITECH_AGENTS_HOME`, `MAGITECH_SHELL`, `MAGITECH_SESSION_ID`, `MAGITECH_PROFILE`, `MAGITECH_PROFILE_DIR` | shell-env registry | Reserved overlay keys for child processes |
| provider `apiKeyEnv` names | provider substrate | Credential env var; missing → `MISSING_CREDENTIAL` |

Unresolved questions: none.
