# Magitech Harness

The autonomous agent runtime and capability seams by **Magitech**.

Built on top of **`anima`** ([`../anima`](../anima)), the composable micro-kernel.

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
* The sibling `anima` repository at `../anima`

### Installation & Test Suite
```bash
cd /mnt/storage/projects/AI/magitech-harness

# Run all integration tests
uv run pytest -v
```

### Running the CLI
```bash
# Execute an agent turn
uv run magitech "Calculate 15 * 8"

# Or use the full binary name
uv run magitech-harness "Calculate 15 * 8"
```
*(Session event logs persist to `.sessions/<session-id>.jsonl`)*

---

## License
MIT
