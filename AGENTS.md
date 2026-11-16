# Agent Guidelines

## Harness Architecture & Invariants

- **Repository Role & Separation of Concerns**:
  - `magitech-harness` is the host runtime substrate and agent turn engine powered by `anima`.
  - **Everything is a Component**: Contains **zero built-in domain tools** and **zero built-in model providers**.
  - All domain tools, model providers, guardrails, and compaction policies reside in `mth-plugins`.
- **Session Event-Sourcing Invariants**:
  - **Model-Visible ⟺ Logged**: Any data or prompt fragment that reaches an LLM request must be reconstructable from the session event log. A new model-visible input requires an explicit session event.
  - **Append-Only History**: The raw session log is immutable and append-only. Compaction checkpoints fold historical context for model projection (`derive_messages`), preserving the full event trace.
  - **Tool-Pairing Integrity**: Turn cuts and compaction boundaries must never sever an assistant tool call from its matching tool result.
- **Cooperative Cancellation**: Long-running streaming turns and tool executions support cooperative cancellation via `AbortSignal`.

## Shared Development Guidance

Follow [home AGENTS](../home/AGENTS.md) for cross-repository standards,
[uv skill](../home/skills/uv/SKILL.md) for package management and standard
test/type-check commands, and [Ruff skill](../home/skills/ruff/SKILL.md) for
linting and formatting.

## Harness-Specific References

### Issue tracker
GitHub Issues via `gh` CLI (`magitech-ai` organization). See `docs/agents/issue-tracker.md`.

### Triage labels
Canonical five-role triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs
Single-context layout (`CONTEXT.md` + `docs/adr/`). See `docs/agents/domain.md`.
