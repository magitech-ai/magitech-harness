# Harness

The Agent Runtime and Capability Seams powering **Something From Nothing**.

Built on top of the standalone composable **`kernel`** ([`../kernel`](../kernel)).

---

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │      Kernel Context     │
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
* The sibling `kernel` repository at `../kernel`

### Installation & Test Suite
```bash
cd /mnt/storage/projects/AI/harness

# Run all integration tests
uv run pytest -v
```

### Running the CLI
```bash
# Execute an agent turn
uv run harness "Calculate 15 * 8"

# Or use the sfn alias
uv run sfn "Calculate 15 * 8"
```
*(Session event logs persist to `.sessions/<session-id>.jsonl`)*

---

## License
MIT
