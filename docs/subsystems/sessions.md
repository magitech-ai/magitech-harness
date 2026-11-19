# Sessions

**Source:** `session/events.py`, `session/store.py`, `session/surface.py`,
`session/service.py`, `session/format.py`.

A session is a durable, append-only chronological record of interaction events.
The log is the source of truth; the model-visible message list is a deterministic
projection over it. Nothing mutates or deletes a committed event.

## Event types

`SessionEvent.type` is a string. Known types are the union of `CORE_EVENT_TYPES`
(14), the owner-owned `OWNER_EVENT_TYPES` (49 across 30 owners), giving 63 at
format v4. Unknown `plugin:` records and `ignorable` events round-trip without
being known types.

Core types include the turn and step boundaries (`turn/start`, `turn/end`,
`step/start`, `step/end`), the role-bearing messages (`user/message`,
`assistant/message`, `system/message`, `developer/message`), `tool/call`,
`tool/result`, `assistant/attempt`, `request/context`, `request/header`, and
`session/end-seed`. Owner-owned types cover approval (`approval/asked`,
`approval/decided`, `approval/policy`), compaction (`compaction/start`,
`compaction/prune`, `compaction/end`, `compaction/summary`), jobs, model
selection, permission presets, sandbox mode, session title, todo, and more.

**[Python]** The retired event type `session/compaction` is deliberately not
a known type; new summaries are written as `compaction/summary`.

## Surface types and projection

Only five types are surface-eligible: `system/message`, `developer/message`,
`user/message`, `assistant/message`, and `tool/result`. A surface event carries
an `append` or `replace` operation and, for a replace, `source_event_seqs`
referencing earlier events. Non-surface events carry neither. `image/offload` is
a message-projection type: it affects the model-visible output only when a
matching `MessageProjection` is registered.

`derive_messages()` folds the surface in order into detached message copies.
A `user/message` surface `replace` over a range collapses that range into one
checkpoint message — the mechanism compaction uses to change the transcript.
A `tool/result` replace may change only its `content`. The system-head message
at node 0 cannot be shadowed. See [compaction](compaction.md).

Invariants checked at append: surface types require a surface op; non-surface
types forbid it; replace references must resolve to current surface nodes and
precede the replacing event. Violations raise `SurfaceError`.

## Format versions

`CURRENT_FORMAT_VERSION = 4`. The stored format is a frozen adjacent chain
v0→v1→v2→v3→v4. Each stage declares its transform; older logs migrate forward
at restore, and corrupt or foreign logs are refused with a direction-aware
error. v4 adds `developer/message`, `job/registered`, `job/settled`, `team/mode`,
and `working-directory/change` to v3's 58 known names.

## Durability

`SessionService` writes events through a JSONL persistence service rooted at
`MAGITECH_SESSIONS_DIR`. A projection cache reduces log reload cost; a
checkpoint policy flushes the log before a top-level tool body and before the
model stream so nested dispatches reuse the durable outer checkpoint. If the
storage directory is unset, sessions are memory-only for that run.

## Example

```python
from magitech_harness import create_harness
from magitech_harness.llm.message import create_user_message
from magitech_harness.session.events import EventType

ctx = create_harness(storage_dir=".sessions")
session = ctx.sessions.create("demo")
session.append(EventType.TURN_START, {"turn": 1})
session.append(
    EventType.USER_MESSAGE,
    create_user_message(content=[{"type": "text", "text": "Hello"}]),
    surface_op="append",
)
messages = session.derive_messages()  # detached, model-visible copies
```

Unresolved questions: none.
