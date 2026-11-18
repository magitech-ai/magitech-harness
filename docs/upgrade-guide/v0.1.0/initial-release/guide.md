---
kind: upgrade-guide
description: The initial 0.1.0 harness baseline.
---

# Magitech Harness 0.1.0

## Change

The 0.1.0 baseline established the event-sourced session core, the composable
service mount model on the Anima kernel, the turn loop, and the strict quality
gates (100% statement and branch coverage, strict mypy). It introduced the
concepts documented in [CONTEXT.md](../../../../CONTEXT.md) and the first set of
[architecture decisions](../../../adr/).

## Migration

There is no earlier released version to migrate from. Start with the
[quickstart](../../../user/quickstart.md). If you are upgrading from an internal
pre-release build, export or discard old session logs: the on-disk format is a
versioned chain and only logs from a supported version migrate forward. Verify
the session format from the storage directory before opening it.
