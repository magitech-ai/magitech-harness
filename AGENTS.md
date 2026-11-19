# Agent Guidelines

## Harness Architecture & Invariants

- **Repository Role & Separation of Concerns**:
  - `magitech-harness` is the host runtime substrate and agent turn engine powered by `anima`.
  - **Everything is a Component**: Contains **zero built-in domain tools** and **zero default-mounted model providers**.
  - All domain tools, guardrails, compaction policies, and single-vendor provider plugins (`openrouter-provider`, `pollinations-provider`) reside in `mth-plugins`.
  - Exception — Tier 1 substrate: the multi-provider adapter `magitech_harness.llm.pi_ai` ships in-tree (ADR 0012) but is opt-in: no default bundle row mounts it (see `docs/adr/0012-tier-1-provider-substrate-in-harness.md`).
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

