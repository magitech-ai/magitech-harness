# Magitech Harness (`mth`)

The composable autonomous agent runtime powering **Magitech**.

Built directly on top of **`animas`** ([`https://github.com/magitech-ai/animas`](https://github.com/magitech-ai/animas)), the micro-kernel.

[![Tests](https://img.shields.io/badge/coverage-100%25-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)

---

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │      Animas Context      │
                    └────────────┬────────────┘
         ┌───────────────┬───────┴───────┬───────────────┐
         ▼               ▼               ▼               ▼
   ctx.sessions      ctx.tools        ctx.llm      ctx.agent_loop
  (Event-Sourced   (Waterfall Tool  (Streaming      (Multi-Step
    JSONL Log)        Pipeline)       Seam)         State Machine)
         │               │               │               │
         └───────────────┴───────┬───────┴───────────────┘
                                 ▼
                     ctx.system_prompt / ctx.shell
```

### The base bundle

The default composition ([`profiles/base.yaml`](src/magitech_harness/profiles/base.yaml))
mounts the reference's shared base: sessions and persistence, prompt assembly, the
tool registry and pipeline, model streaming and retry, filesystem and shell,
jobs, compaction, approvals, goals, and the base tool set (`bash`, `read_file`,
`write_file`, `edit_file`, `glob`, `grep`, `job_list`, `job_output`, `job_kill`,
`todo_write`, `get_goal`, `create_goal`, `update_goal`). Optional, on-demand, and
third-party capabilities live in
[`mth-components`](https://github.com/magitech-ai/mth-components).

Ownership and per-row evidence against the pinned reference (dsh
`4b399c39`, v0.2.3-alpha.2) are in [PARITY.md](PARITY.md); the decision is
[ADR 0022](docs/adr/0022-supersede-zero-builtin-domain-tools-base-bundle-in-harness.md).

### The 6 Core Capability Seams
* **`ctx.sessions`**: Append-only event log (`turn/*`, `step/*`, `user/message`, `assistant/message`, `tool/*`). Follows the principle: **Model-visible ⟺ Logged**.
* **`derive_messages()`**: Deterministic projection reconstructing conversation history from session logs.
* **`ctx.tools`**: Scoped tool registry + guarded waterfall pipeline (`tools/pre-execute` ➔ execute ➔ `tools/post-execute`).
* **`ctx.system_prompt`**: Modular section assembly.
* **`ctx.llm`**: Pluggable provider seam with built-in mock and async streaming hooks.
* **`ctx.agent_loop`**: Full multi-step agent driver handling tool execution cycles.
* **`ctx.shell`**: Local async shell execution provider.

---

## Quickstart

### Prerequisites
* Python 3.12+ (tested with `uv`)
* The `animas` micro-kernel at `../animas` (or added with `uv add animas`)

### Running the Test Suite
`magitech-harness` maintains strict **100% statement and 100% branch test coverage**:

```bash
cd magitech-harness

# Run all unit and integration tests with coverage enforcement
uv run pytest
```

### Running the CLI
```bash
# Execute an agent turn using the mth CLI
uv run mth run "Calculate 15 * 8"

# Or use the aliases
uv run magitech run "Calculate 15 * 8"
uv run magitech-harness run "Calculate 15 * 8"
```
*(Session event logs persist to `.sessions/<session-id>.jsonl`)*

---

## License
MIT
