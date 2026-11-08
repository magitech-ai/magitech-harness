# Magitech Harness

The native Python agent runtime substrate and turn execution engine powered by the Anima micro-kernel.

## Language

### Core Entities

**Harness**:
The host runtime substrate that orchestrates agent lifecycles, session persistence, and turn loops over the micro-kernel.
_Avoid_: Framework, orchestrator, daemon

**Session**:
A durable, append-only chronological record of interaction events between users, models, and tools.
_Avoid_: Conversation, chat, thread

**Turn**:
A single unit of conversational execution starting with user input and resolving through reasoning, tool calls, and model response.
_Avoid_: Step, interaction, prompt

**Event**:
An immutable structured record appended to a session log representing a state transition or message exchange.
_Avoid_: Log line, message, action

**Profile**:
A declarative configuration bundle that specifies the set of plugins to load for an application run.
_Avoid_: Preset, blueprint, config file

### Context & Compaction

**Projection**:
The deterministic reduction of an event-sourced session log into the model-visible message list.
_Avoid_: Context serialization, history building

**Compaction**:
The folding of historical session events prior to a balanced cut point into a checkpoint summary for model projection.
_Avoid_: Pruning, truncation, context clearing

**Cut Point**:
A verified index in a session event log where historical events can be summarized without severing tool call and tool result pairs.
_Avoid_: Split index, boundary

### Execution & Control

**Abort Signal**:
A cooperative cancellation handle passed into streaming turns and tool runs to signal requested interruption.
_Avoid_: Cancellation token, interrupt flag, kill switch

**Token Chunk**:
An incremental text or deliberation delta yielded by a model provider during response streaming.
_Avoid_: Stream chunk, token packet, delta fragment
