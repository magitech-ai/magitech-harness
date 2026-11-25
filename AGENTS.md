# Agent Guidelines

## Harness Architecture & Invariants

- **Repository Role & Separation of Concerns**:
  - `magitech-harness` is the host runtime substrate and agent turn engine powered by `animas`.
  - **Everything is a Component**: every capability is an `apply(ctx, ...)` component or a service class mounted by a declarative row. Nothing is hard-wired into the kernel.
  - **Base-bundle rule (ADR 0022)**: every tool and service mounted by DeepSeek Harness's `packages/bundle/base/cordis.patch.yml` ships **here**, mounted by `src/magitech_harness/profiles/base.yaml` — `bash`, the filesystem tools, `glob`/`grep`, job control, `todo_write`, and the goal tools included.
  - Optional, on-demand, and third-party tools/providers/guards still live in `mth-components` (`openrouter-provider`, `pollinations-provider`, bridges, extra tools).
  - The multi-provider adapter `magitech_harness.llm.pi_ai` is Tier 1 substrate (ADR 0012, broadened by ADR 0022) and stays opt-in: no default bundle row mounts it.
  - Ownership evidence and every row's probe live in [`PARITY.md`](PARITY.md). Moving a row between repos requires a new ADR superseding 0022.
- **Session Event-Sourcing Invariants**:
  - **Model-Visible ⟺ Logged**: Any data or prompt fragment that reaches an LLM request must be reconstructable from the session event log. A new model-visible input requires an explicit session event.
  - **Append-Only History**: The raw session log is immutable and append-only. Compaction checkpoints fold historical context for model projection (`derive_messages`), preserving the full event trace.
  - **Tool-Pairing Integrity**: Turn cuts and compaction boundaries must never sever an assistant tool call from its matching tool result.
- **Cooperative Cancellation**: Long-running streaming turns and tool executions support cooperative cancellation via `AbortSignal`.

## Shared Development Guidance

Shared standards, workflow and quality gates:
[agents-home AGENTS](../agents-home/AGENTS.md) → `docs/ENGINEERING.md`,
`docs/WORKFLOWS.md`, `CODING_STANDARDS.md`, and `scripts/check_all.py`.

### Local checks
Run `python3 ../agents-home/scripts/check_all.py --fast` before committing;
CI runs the full gate.

## Harness-Specific References

### Issue tracker
GitHub Issues via `gh` CLI (`magitech-ai` organization). See `docs/agents/issue-tracker.md`.

### Triage labels
Canonical five-role triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs
Single-context layout (`CONTEXT.md` + `docs/adr/`). See `docs/agents/domain.md`.

