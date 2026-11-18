# System prompt

**Source:** `system_prompt/service.py`.

`ctx.system_prompt` assembles the invariant prompt scaffolding that is
recomputed for each model request. It is a set of ordered sections plus a
runtime context block, not a single editable string. See
[adr/0007](../adr/0007-two-layer-invariant-prompt-scaffolding.md).

## Sections

A section has a name, ordered content, and behavior flags. Registration returns
a disposer; removing a section is an effect with full cleanup. Built-in sections
include harness identity, deployment persona prefix and suffix, tool guidance,
and runtime context.

Configuration keys (component config, default in parentheses): `toolOrder` /
`tool_order` (none), `includeHarnessIdentity` / `include_harness_identity`
(true), `personaPrefix` / `persona_prefix` (empty), `personaSuffix` /
`persona_suffix` (empty), and `includeRuntimeContext` /
`include_runtime_context` (true).

## Model-visible effect

Sections are deterministic and order-stable so prompt caching remains valid. The
identity section is invariant across a run; per-step runtime context is appended
as the changing layer. Tool guidance is registered by tool components (for
example `tool:bash` tells the model to prefer the file tools for file access).
The resulting content appears as the `system/message` surface event.

Unresolved questions: none.
