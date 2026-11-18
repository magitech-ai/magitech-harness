# Filesystem

**Source:** `fs/local.py`, `mth-plugins/src/mth_plugins/tools/fs/path_guard.py`,
`mth-plugins/src/mth_plugins/tools/{read_file,write_file,edit_file}/`.

The harness provides a local filesystem service (`ctx.fs`). Domain file tools
live in `mth-plugins`. Their names are `read_file`, `write_file`, and
`edit_file`.

**[Python]** The reference tool names are `read` / `write` / `edit`. This port
uses the suffixed names; the LSP bridge accepts both spellings where it hooks
file edits.

## Path confinement

`read_file` and `write_file` confine every requested path to the workspace root.
`confined_path(root, raw)` canonicalizes the root with `resolve(strict=True)`,
resolves the candidate (collapsing `..` and following existing symlink
prefixes), and rejects any result outside the root with `PathEscapeError`. An
escape returns an error result and never discloses file content. Absolute paths
inside the root are accepted; absolute paths outside it are denied.

The workspace root is resolved per call from component config `workspace_root`
or `ctx.workspace_dir`, so a branch-worktree child confines to its own
workspace. `edit_file` does not yet apply this confinement.

## Tool behavior

- `read_file` — 1-indexed `offset`/`limit` pagination, capped at 2000 lines per
  call; rejects binary files; a directory containing `SKILL.md` resolves to it.
- `write_file` — writes `content`, creating parent directories; blank path
  rejected, empty content valid.
- `edit_file` — replaces a unique `target` string with `replacement`; rejects
  zero or multiple matches.

## Configuration

`read_file` component config: `default_limit` (`2000`), `max_limit` (`2000`),
`workspace_root` (`ctx.workspace_dir`). Harness filesystem service config: `cwd`
(process CWD), `diff_basis_max_bytes` (10 MiB). `fs-observation-policy` ignores
its config.

Unresolved questions: none.
