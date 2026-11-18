# Jobs

**Source:** `jobs/local.py`, `jobs/types.py`.

`ctx.jobs` is the local background-job registry. It is mounted as `jobs` in the
base bundle; `LocalJobRegistry` runs jobs under an owner and publishes their
lifecycle to the session log.

## Lifecycle

A registered job is owned by an agent or owner id. The registry emits
`job/registered` when a job is accepted and `job/settled` when it reaches a
terminal state. Both are format-v4 event types.

## Configuration

Component config (default in parentheses): `maxConcurrentJobsPerOwner` (`10`),
`retainBytes` (`262144`), `settledRetainBytes` (`16384`), and `pumpPollMs`
(`150`). There are no profile-YAML keys for jobs.

Unresolved questions: none.
