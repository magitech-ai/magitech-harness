# Parity target pins

## Anima (verified in Phase 0a, 2026-11-10)

- Base ref: `1a9384345024d36711d948f96d217f5d43cb7e56` (PR #12 merge)
- Reflect-fix commit: `898ac7d1f91308fb1290ce21e96537d48556e915`
  (branch `phase0a/reflect-fix` in anima; fixes the reflect recursion on
  missing reserved-word attributes — see the anima commit message)
- Pinned dependency: `anima>=0.1,<0.2` in `pyproject.toml`
  (path source `../anima` for local development; the published package
  declares the version range, not a bare path-only dep)
- Verified suites against this ref (Python 3.12, sibling-layout clones):
  anima 538 passed (100% coverage), magitech-harness 118/118 (100%),
  mth-plugins 260/260 (100%)
