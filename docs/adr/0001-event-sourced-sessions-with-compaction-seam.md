# 0001. Event-Sourced Sessions with Compaction Seam

Date: 2026-11-07
Status: accepted

## Context

Conversational AI agents execute over multiple turns with growing context. In-memory message arrays lose execution auditability, make time-travel or rehydration fragile, and couple session persistence directly to transient model prompt formats. Simultaneously, long-running agent loops eventually exceed LLM context window limits and require history summarization without severing critical tool call/result pairs.

## Decision

We model all session state as an append-only sequence of immutable events (`EventType.USER_PROMPT`, `EventType.MODEL_TURN`, `EventType.TOOL_CALL`, `EventType.TOOL_RESULT`, `EventType.COMPACTION`).

The core harness engine provides an event-sourcing projection function (`derive_messages`) and a tool-pairing boundary verifier (`is_tool_pairing_balanced`), but delegates all compaction policy, triggers, and LLM summarization to plugins:
1. When `EventType.COMPACTION` appears in the log, `derive_messages` folds historical events up to `cut_seq` into a checkpoint user message: `[Conversation summary up to checkpoint]:\n{summary}`.
2. Events preceding the cut point are excluded from the model-visible prompt projection while remaining permanently in the append-only session store on disk.
3. Compaction cuts are rejected unless `is_tool_pairing_balanced` confirms that no assistant tool call is severed from its corresponding tool result across the cut boundary.

## Consequences

- The session event log is strictly append-only, human-readable, and replayable.
- The core runtime remains free of hardcoded summarization prompts, token pressure heuristics, or vendor-specific compaction logic.
- Model-visible prompts are pure projections derived deterministically from the event log on each turn.
