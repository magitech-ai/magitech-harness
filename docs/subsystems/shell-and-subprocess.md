# Shell and subprocess

**Source:** `shell/`, `subprocess/`, `shell_env/service.py`.

Shell execution is mounted through platform-gated executors in the base bundle:
`bash-local` on POSIX and `pwsh-local` on Windows. The `LocalSubprocessRuntime`
backs them with bounded output collection and spill-to-disk. See
[adr/0008](../adr/0008-process-tree-management-and-bounded-output-buffering.md).

## Bounded output and spill

The output collector keeps a head and tail of each stream, and spills the
complete stream to disk when the in-memory budget is exceeded. It never silently
drops bytes from the returned text: a truncated stream carries a marker that
names the spill path. A `ShellResult` exposes `truncated` plus
`stdout_spill_path` and `stderr_spill_path`, populated from the collector's
finalize data.

The `bash` tool renders the reference-parity marker:

```text
[output truncated; full output: <path-or-(unavailable)>]
```

If no collector spill exists (the direct-subprocess fallback path), the tool
spills the output it has and renders the same marker.

## Process management

Shell commands run under a process tree so termination reaches descendants.
Cancellation aborts the command's signal and kills the tree on expiry. An
expiry policy (`kill` or `none`) controls what happens to a running process when
its handle expires.

## Configuration

Shell defaults (component config, default in parentheses): `cwd` (process CWD),
`timeoutMs` (`120000`), `maxTimeoutMs` (`600000`), `maxOutputBytes` (`64000`),
`maxSpillBytes` (`64 MiB`), `graceMs` (`3000`), and `pwshPath` (probed, Windows
only). The subprocess runtime uses `spillDir` (computed default). The shell-env
registry provides reserved `MAGITECH_*` overlay keys to child processes.

The `bash` tool's `max_output_bytes` code default is `64 KiB`; a call may pass a
smaller `max_output_bytes` and a `timeout_ms` (default `30000`).

Unresolved questions: none.
