# Magitech Harness (`mth`)

The composable autonomous agent runtime powering **Magitech**.

Built directly on top of **`anima`** ([`https://github.com/magitech-ai/anima`](https://github.com/magitech-ai/anima)), the micro-kernel.

[![Tests](https://img.shields.io/badge/coverage-100%25-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)

---

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │      Anima Context      │
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
* The `anima` micro-kernel at `../anima` (or installed via pip)

### Running the Test Suite
`magitech-harness` maintains strict **100% statement and 100% branch test coverage**:

```bash
cd /mnt/storage/projects/AI/magitech-harness

# Run all unit and integration tests with coverage enforcement
uv run pytest
```

### Running the CLI
```bash
# Execute an agent turn using the mth CLI
uv run mth "Calculate 15 * 8"

# Or use the aliases
uv run magitech "Calculate 15 * 8"
uv run magitech-harness "Calculate 15 * 8"
```
*(Session event logs persist to `.sessions/<session-id>.jsonl`)*

---

## License
MIT
