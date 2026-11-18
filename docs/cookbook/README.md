# Cookbook

Step-by-step tasks for working with the harness. Each page states prerequisites,
the actions in order, the real entry path, and an observable verification. Design
rationale lives in [adr/](../adr/); current behavior lives in
[subsystems/](../subsystems/README.md).

## Getting started

- [Run your first turn](first-turn.md) — boot a harness and complete one user turn.
- [Resume a session](resume-session.md) — continue from persisted history.
- [Run headless with JSON output](headless-json.md) — one-shot non-interactive use.

## Configuring

- [Write a profile](write-profile.md) — overlay component rows on the base bundle.
- [Add a plugin](add-plugin.md) — register a new tool or service component.

## Working with turns

- [Approve a tool call](approve-tool-call.md) — wire an approval presenter.
- [Compact a long session](compact-long-session.md) — fold history into a checkpoint.

Unresolved questions: none.
