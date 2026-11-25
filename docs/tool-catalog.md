# Tool catalog

Exhaustive reference for the tools that can be registered on `ctx.tools`.
Source of truth: registration sites in `src/magitech_harness/` and
`mth-components/src/mth_components/`, plus the canonical `define_tool` contract
in `src/magitech_harness/tools/schema.py`. This page is maintained by hand from
source; the docs review checks it against the registration calls.

Ownership follows the dsh base-bundle rule (ADR 0022): every tool in the
reference's `packages/bundle/base/cordis.patch.yml` ships in this repository and
is mounted by [`profiles/base.yaml`](../src/magitech_harness/profiles/base.yaml);
optional, on-demand, and third-party tools live in `mth-components`. The
per-row evidence map is [PARITY.md](../PARITY.md).

All shipped tools are registered with `define_tool` from
`magitech_harness.tools.schema`. Tools that need approval or a deadline declare
`requiresApproval` or `timeoutMs` in their spec; `run_code` is a reserved name
and cannot be registered.

## Filesystem and shell

Mounted by `tool-fs`, `tool-fs-search`, and `tool-bash` in
[`profiles/base.yaml`](../src/magitech_harness/profiles/base.yaml).

| Tool | Arguments | Behavior |
|---|---|---|
| `bash` | `command` (req), `timeout_ms` (default `30000`), `cwd` | Runs a shell command via `ctx.shell` with a 64 KiB in-memory cap; spills complete output to disk and renders `[output truncated; full output: …]`. Reports exit code and timeout. |
| `read_file` | `path` (req), `offset` (1-indexed), `limit` (default/max 2000) | Reads a workspace-confined text file with pagination; rejects binary files; a directory containing `SKILL.md` resolves to that file. |
| `write_file` | `path` (req), `content` (req) | Writes a workspace-confined file, creating parent directories. Blank path rejected; empty content writes an empty file. |
| `edit_file` | `path` (req), `target` (req), `replacement` (req) | Replaces a unique target string. Rejects blank paths, zero matches, or multiple matches. Not yet workspace-confined. |

**[Python]** Reference names are `read` / `write` / `edit`; this port uses the
suffixed names. The LSP bridge accepts both spellings in its post-execute hook.

## Background jobs

| Tool | Arguments | Behavior |
|---|---|---|
| `job_list` | none | Lists the caller's running and finished jobs with ids, kinds, labels, and statuses. |
| `job_output` | `job_id` (req), `wait` (bool), `timeout_ms` | Reads output since the previous read (stream jobs) or the finished result; `wait` blocks until settlement or the configured cap. |
| `job_kill` | `job_id` (req), `reason` | Requests cancellation; reports `cancellation-requested` or `already-finished`. |

## Planning and goals

| Tool | Arguments | Behavior |
|---|---|---|
| `todo_write` | `todos` (req: complete list of `{content, status}`) | Replaces the whole task list and appends one `todo/write` session event; the `todos` projection clears on the next `turn/start`. Rejects duplicate content and an empty status. |
| `get_goal` | none | Reads the current session goal, including the id and revision `update_goal` requires. |
| `create_goal` | `objective` (req), `max_goal_rounds` | Creates and arms a durable goal; requires a direct human turn on a top-level agent. A completed goal may be replaced; any other phase must be cleared or resumed. |
| `update_goal` | `goal_id` (req), `revision` (req), `action` (req), `objective`, `max_goal_rounds`, `blocked_reason` | Compare-and-set mutation. `edit`/`pause`/`resume` require a direct human turn; `complete`/`blocked` also accept the exact admitted goal round; `blocked` is refused before the configured round threshold. |

## Code execution

| Tool | Arguments | Behavior |
|---|---|---|
| `execute_python` | `code` (req) | CodeAct PTC runner with persistent `session_locals`, auto-dispatch of registered tools, and `final_answer(value)`. Runs in a subprocess with a scrubbed environment and an internal 30 s timeout. |

## Subagents

| Tool | Arguments | Behavior |
|---|---|---|
| `subagent` | `description` (req), `prompt` (req), `name`, `model`, `workspace` (`inherit`/`branch`), `allowed_tools` [], `wait` | Delegates to a managed child through the registered `spawn` provider; the child is registered in the live tree and logged as `subagent/descriptor`. |
| `subagent_fork` | same as `subagent` | Delegates through the `fork` provider, seeding the child session from the parent's completed events. |
| `send_message` | `agent` (req), `message` (req) | Delivers a follow-up instruction to an agent in the caller's tree. |
| `list_agents` | none | Lists the root and all reachable descendants with status and capacity. |
| `interrupt_agent` | `agent` (req) | Requests cooperative interruption of a live child. |
| `wait_agent` | `timeout_ms` (10000–3600000, default 30000) | Waits for pending next-step input; reports whether it timed out. |
| `list_subagent_models` | none | Lists the model routes a delegated child can use. |

## Web

| Tool | Arguments | Behavior |
|---|---|---|
| `web_search` | `queries` (req, array of strings) | Searches the web through a registered search provider selected by the current model route. Fails with a `WEB_PROVIDER_*` error when none is registered; results are untrusted data. |
| `web_fetch` | `url` (req, http(s)) | Fetches an HTTP(S) URL through the anonymous fetcher (which can reach intranet/local addresses), decodes HTML/text within byte/char caps, and treats the body as untrusted data. |
## Agent Teams

Team tools are admitted only while the shared task board is enabled for the session.
The board is event-sourced and compare-and-set: every mutation requires the latest
revision from `team_task_get` or `team_task_list`.

| Tool | Arguments | Behavior |
|---|---|---|
| `team_task_create` | `subject` (req), `description` (req), `blocked_by` [], `write_scopes` [] | Creates an unowned pending task with advisory write scopes and dependency ids. |
| `team_task_list` | `status`, `owner`, `unowned`, `ready`, `cursor` (0), `limit` (1–100, default 50) | Lists tasks with readiness, owner, revision, blockers and write-scope warnings; returns `nextCursor` when more pages exist. |
| `team_task_get` | `task_id` (req) | Reads the complete latest value of one task. |
| `team_task_update` | `task_id` (req), `expected_revision` (req), `action` (req), plus action-specific fields | Compare-and-set transition: `claim`, `release`, `edit`, `set_dependencies`, `complete`, `reopen`, `reassign`, `delete`. |

## Language and skills

| Tool | Arguments | Behavior |
|---|---|---|
| `lsp_definition` | `path`, `line`, `character` (zero-indexed) | Looks up a symbol definition via LSP. |
| `lsp_references` | `path`, `line`, `character` | Looks up all references to a symbol via LSP. |
| `activate_skill` | `name` (req) | Loads and returns a skill's full body. |
| `run_skill_script` | `skill_name` (req), `script_name` (req), `args` [] | Runs a skill-bundled script via the sandbox/shell; rejects `..` and absolute script names. |

## Sessions and workspace

| Tool | Arguments | Behavior |
|---|---|---|
| `session_search` | `query` (req), `limit` (default 10) | SQLite FTS5 search over sessions, messages, and tool calls; auto-indexes on `turn/finish`. |
| `snapshot_diff` | `target_snapshot` (default `latest`), `workspace` | Unified diff of the workspace against a snapshot. |
| `snapshot_restore` | `snapshot_id` (req), `workspace` | Restores a workspace from a snapshot. |

## Dynamic MCP tools

`mcp_<server>_<name>` proxies are registered per configured MCP server with the
server's `inputSchema` as the argument schema. They call `McpClient.call_tool`
and are disposed when the server is removed.

## Filesystem search

| Tool | Arguments | Behavior |
|---|---|---|
| `glob` | `pattern` (req), `path` | Finds files whose paths match a glob pattern under the workspace root, including hidden files, in modification-time order; caps results and reports the cap. A pattern with no slash matches the basename at any depth. |
| `grep` | `pattern` (req), `path`, `include` | Searches file contents with a regex, returning matching lines with line numbers grouped by file; caps matches and reports the cap. `include` filters files by glob. |

**[Python]** The reference drives ripgrep via subprocess; this port uses
`pathlib` and Python `re` so the wheel stays self-contained and cross-platform.
The observable contract (pattern/arguments, discovery, caps, markers) matches;
ripgrep-specific regex extensions are not supported.

Unresolved questions: none.
