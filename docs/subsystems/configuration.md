# Configuration

**Source:** `components/layers.py`, `profiles/merge.py`, `__init__.py`,
`docs/adr/0010`, and the [config catalog](../config-catalog.md).

Two distinct mechanisms are often both called "configuration": layer resolution
for `.agents` protocol files, and composition of a runtime from a base bundle
plus an optional profile. Keep them separate.

## Four-tier layer resolution

`ctx.layers` resolves files from lowest to highest precedence:

1. **global** — `/etc/magitech` or `$MAGITECH_GLOBAL_DIR`
2. **user** — `$MAGITECH_AGENTS_HOME` or `~/.agents/`
3. **agent** — `<shared root>/agents/<name>/`
4. **project** — `<project>/.agents/`

The two homes are the harness home (`$MAGITECH_HOME`, default `~/.magitech`)
and the shared agents root (`$MAGITECH_AGENTS_HOME`, default `~/.agents`). Both
resolvers treat a blank env value as unset and expand a configured path.

`find_first` searches highest to lowest; `find_all` returns base to highest. The
project layer is used only when an explicit `project_path` is supplied — an
ambient `Path.cwd()` is never used as a project layer. See
[adr/0010](../adr/0010-deterministic-four-tier-configuration-layer-resolution.md).

## Bundle and profile composition

`create_harness()` mounts the shipped base bundle and optionally overlays a
`profile` argument, then any `patches` in order. Overlay semantics
(`profiles/merge.py`): a known row id overlays the base row's keys; an unknown
id appends a row; duplicate ids, non-string ids, and rows without a resolvable
`name` fail at boot. Layers resolve as bundle → profile → each `--patch` in argv
order; a later layer wins. The `MAGITECH_SESSIONS_DIR` interpolation value is
computed from the caller's `storage_dir` and supplied to the loader.

## Where values live

- **Profile YAML** carries a small set of values: approval policy, session and
  persistence roots, live batch delay, title limits, title-first-prompt limits,
  projection-cache write cadence, the agent list, and repeat thresholds.
- **Component config and code defaults** carry the rest: system-prompt sections,
  tool-executor parallelism, filesystem and shell budgets, job limits, retry
  policy, and provider settings. Missing keys fall back to code defaults where
  the service declares one; keys declared required (for example title limits)
  raise `ValueError` when absent.
- **Environment** supplies `MAGITECH_GLOBAL_DIR` (read directly) and
  `MAGITECH_SESSIONS_DIR` (interpolated by the loader). Provider credentials use
  the env var named by each provider's `apiKeyEnv`. `MAGITECH_HOME`,
  `MAGITECH_AGENTS_HOME`, `MAGITECH_SHELL`, `MAGITECH_SESSION_ID`,
  `MAGITECH_PROFILE`, and `MAGITECH_PROFILE_DIR` are reserved overlay keys
  managed by the shell-env registry.

Unknown configuration keys fail loud at the owning service; there is no global
"ignore unknown keys" fallback.

Unresolved questions: none.
