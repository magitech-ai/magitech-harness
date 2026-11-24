---
kind: upgrade-guide
description: py.typed marker and strict mypy become part of the quality gate.
---

# Strict typing and py.typed

## Change

The package now ships a `py.typed` marker and is checked with strict mypy
(`disallow_untyped_defs`, `warn_return_any`, `warn_unused_configs`). At that
release the runtime was pinned to `animas>=0.1,<0.2`; the 0.3.0 release moved
this to `animas>=0.3,<0.4`. See
[adr/0004](../../../adr/0004-strict-quality-gates-100-coverage-and-mypy.md).

## Migration

1. Downstream type checkers now see inline types for the harness. Remove any
   local stub packages or `# type: ignore` comments that were suppressing
   missing-import errors for the harness.
2. Ensure your environment resolves a compatible `animas` (`>=0.1,<0.2`). A
   newer major version is not supported by this release.
3. Fix new strict-mypy findings in your own code that were previously hidden by
   untyped calls into the harness. Verify with `uv run --locked mypy src` and the
   full test suite.
