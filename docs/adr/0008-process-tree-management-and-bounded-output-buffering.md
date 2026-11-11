# ADR 0008: Process Tree Management and In-Memory Bounded Shell Execution

## Status
Accepted

## Date
2026-11-08

## Context
In agentic runtimes, subprocess execution is a primary operational vulnerability:
1. **Orphan / Zombie Subprocesses**: When a shell command spawns child processes (e.g. `npm run dev`, `find`, or background compilers) and is cancelled or times out, terminating only the parent process leaves detached descendant processes running indefinitely, leaking file locks, CPU, and ports.
2. **Memory Exhaustion (OOM)**: Commands that dump enormous output streams (e.g. `cat /dev/urandom` or scanning huge directories) will consume unbounded memory if collected naively into memory before return, crashing the entire harness process.
3. **Execution Seam Integrity**: The host engine must own low-level OS process management without exposing model-facing tools, leaving tool schema presentation and disk spilling to marketplace components.

## Decision
1. **Process Group Isolation**:
   All subprocesses spawned by `LocalShellService.run` (`ctx.shell`) set `start_new_session=True` on POSIX (mapping to distinct process groups) and `CREATE_NEW_PROCESS_GROUP` on Windows.
2. **Cross-Platform Tree Termination**:
   Process termination recursively eliminates descendant trees:
   - **POSIX**: `os.killpg(os.getpgid(proc.pid), signal.SIGKILL)`.
   - **Windows**: `taskkill /F /T /PID <pid>`.
   Triggered on execution timeout or when an associated `AbortSignal` is cancelled.
3. **In-Memory Bounded Output Buffering**:
   `LocalShellService.run` enforces a strict memory budget (default 64 KiB: 32 KiB head / 32 KiB tail). When output overflows the budget, the intermediate stream is discarded, inserting a `\n... [truncated X bytes] ...\n` marker to eliminate OOM vulnerabilities while preserving diagnostic context.
4. **Structured Shell Result**:
   Returns `ShellResult(exit_code, stdout, stderr, timed_out, aborted, bytes_read)` supporting both structured property inspection and backwards-compatible 3-tuple unpacking (`code, out, err`).

## Consequences
- **Positive**: Zero orphan processes on cancellation or timeout; immune to memory exhaustion from massive output streams.
- **Positive**: Clean architectural seam between the headless OS execution backend (`magitech-harness`) and domain tools with disk spilling (`mth-plugins`).
