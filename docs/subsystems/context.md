# Context

**Source:** `context/service.py`, `agent/cancellation.py`.

`ctx.context` is the context service mounted in the base bundle. It has no
configuration keys. The operationally important part of this subsystem is the
cooperative cancellation primitives in `agent/cancellation.py`, which every
dispatch uses.

## AbortSignal and AbortController

An `AbortController` owns a signal. `signal.is_aborted` and `signal.reason` read
its state; `controller.abort(reason)` transitions it once; `signal.on_abort(cb)`
registers a callback and returns a disposer. Signals are value-carriers for an
abort reason, not task handles.

## Deadlines

`deadline(upstream, timeout_ms, code)` fuses an upstream signal with a
loop-armed timer and stamps a `TimeoutReason(code, timeout_ms)`. It returns a
`Deadline` with `.signal`, `.dispose()`, and context-manager support.
`timeout_of(signal, code)` reads a `TimeoutReason` only when its code matches,
which lets a plugin classify its own timeout without mistaking an outer caller
abort for its deadline.

The timeout guard swaps a deadline signal onto a tool dispatch and restores the
caller's signal in `finally`; it replaces the result with a `TOOL_TIMEOUT` error
only if its own scoped deadline fired. This is cooperative: a tool that ignores
its signal runs to completion. See [tools](tools.md).

Unresolved questions: none.
