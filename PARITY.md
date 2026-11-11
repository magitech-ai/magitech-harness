# Parity target pins

## Anima (verified in Phase 0a, 2026-11-10)

- Base ref: `ab83c6cae1cde2f496a3dc4059cec20eafda22ae` (PR #12 merge)
- Reflect-fix commit: `d3d13eaaf925daf535f0a31721700d93f3c5d609`
  (branch `phase0a/reflect-fix` in anima; fixes the reflect recursion on
  missing reserved-word attributes — see the anima commit message)
- Pinned dependency: `anima>=0.1,<0.2` in `pyproject.toml`
  (path source `../anima` for local development; the published package
  declares the version range, not a bare path-only dep)
- Verified suites against this ref (Python 3.12, sibling-layout clones):
  anima 538 passed (100% coverage), magitech-harness 118/118 (100%),
  mth-plugins 260/260 (100%)
