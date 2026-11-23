# Tool catalog

Exhaustive reference for the tools that can be registered on `ctx.tools`.
Source of truth: registration sites in `mth-plugins/src/mth_plugins/` and the
canonical `define_tool` contract in `src/magitech_harness/tools/schema.py`. This
page is maintained by hand from source; the docs review checks it against the
registration calls. Domain tools live in `mth-plugins`; the harness ships none.

All shipped tools are registered with `define_tool` from
`magitech_harness.tools.schema`. Tools that need approval or a deadline declare
`requiresApproval` or `timeoutMs` in their spec; `run_code` is a reserved name
and cannot be registered.

## Filesystem and shell

| Tool | Arguments | Behavior |
|---|---|---|
| `bash` | `command` (req), `timeout_ms` (default `30000`), `cwd` | Runs a shell command via `ctx.shell` with a 64 KiB in-memory cap; spills complete output to disk and renders `[output truncated; full output: …]`. Reports exit code and timeout. |
| `read_file` | `path` (req), `offset` (1-indexed), `limit` (default/max 2000) | Reads a workspace-confined text file with pagination; rejects binary files; a directory containing `SKILL.md` resolves to that file. |
| `write_file` | `path` (req), `content` (req) | Writes a workspace-confined file, creating parent directories. Blank path rejected; empty content writes an empty file. |
| `edit_file` | `path` (req), `target` (req), `replacement` (req) | Replaces a unique target string. Rejects blank paths, zero matches, or multiple matches. Not yet workspace-confined. |

**[Python]** Reference names are `read` / `write` / `edit`; this port uses the
suffixed names. The LSP bridge accepts both spellings in its post-execute hook.

## Code execution

| Tool | Arguments | Behavior |
|---|---|---|
| `execute_python` | `code` (req) | CodeAct PTC runner with persistent `session_locals`, auto-dispatch of registered tools, and `final_answer(value)`. Runs in a subprocess with a scrubbed environment and an internal 30 s timeout. |

## Subagents

| Tool | Arguments | Behavior |
|---|---|---|
| `invoke_subagent` | `role` (req), `prompt` (req), `agent_name`, `model`, `workspace` (`inherit`/`branch`), `allowed_tools` [], `wait` | Spawns a child harness, optionally in a git worktree; copies providers; pins approval to `never`. |
| `send_message` | `subagent_id` (req), `message` (req) | Runs another turn on an existing subagent. |
| `manage_subagents` | `action` (req: `list`/`kill`/`kill_all`), `subagent_ids` [] | Inspects or terminates subagents, aborting tasks and removing worktrees. |

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
