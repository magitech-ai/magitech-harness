# 0004. Strict Quality Gates: 100% Coverage and Mypy Typing

Date: 2026-11-07
Status: accepted

## Context

Agent runtimes execute complex asynchronous event loops, handle third-party JSON schemas, and coordinate multiple pluggable services. Subtle regressions, missing branches in error paths, or untyped `Any` returns quickly compound into elusive runtime failures.

## Decision

We enforce zero-tolerance quality gates across all codebase layers:
1. **100% Statement and Branch Coverage**: All unit and integration test runs enforce `--cov-fail-under=100`. Tests in `tests/unit/` mirror `src/` 1:1, flat named, and hermetic.
2. **Strict Static Typing**: The codebase enforces strict `mypy` with `disallow_untyped_defs = true`, `warn_return_any = true`, and `warn_unused_configs = true`. Every package exports a `py.typed` marker.
3. **Google Python Style & Ruff**: All code adheres to the Google Python Style Guide, gated by `ruff check` and `ruff format --check` at line length 88.
4. **Zero Ancestral Mentions Rule**: Complete prohibition against naming ancestral or upstream projects/brands anywhere in code, tests, docstrings, commits, or documentation.

## Consequences

- Every newly added module, branch, and edge-case error path must have a corresponding hermetic test before merging.
- Static typing ensures protocol conformance between core service interfaces and marketplace plugins.
- Eliminates technical debt and prevents regressions at compile and test time.
