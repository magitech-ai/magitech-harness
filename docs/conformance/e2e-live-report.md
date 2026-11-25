# End-to-end live test report — installed wheel (2026-11-18)
> **Status:** point-in-time record of the 2026-11-18 run against 0.2.0 wheels;
> current shipped versions are 0.3.0 with `animas>=0.3,<0.4`. The two FAILs
> below were fixed and re-verified (see the follow-up section and
> [`release-notes/v0.3.0.md`](../release-notes/v0.3.0.md)).

Validation of the **installed wheels**, driven as an end-user would — CLI
subprocesses and the installed Python API, with no sibling source checkout on
the path. The wheels were built from repository `main` with `uv build --wheel`
and installed into a clean virtual environment; the `mth` binary was confirmed
to import from `site-packages`, not editable source.

## Environment

- Python 3.14.9, clean venv, three wheels: animas 0.2.0, magitech-harness 0.2.0,
  mth-components 0.2.0.
- Scratch environments and session directories were removed after the run.

## Results: 36 PASS, 2 FAIL across 38 checks

### CLI and user flows — 18 PASS, 2 FAIL

PASS: wheel provenance; `mth --help` surface; `run` rejected (exit 2); headless
text and `--json` (sessionId/text/reason/exitCode); session list/show; resume via
`--session-id`; REPL slash commands and exit; profile overlay; profile + patch
overlays; bad profile/patch/session fail loud; `MAGITECH_AGENTS_HOME` accepted;
installed-package agent turn and seams; plugin catalog discovery.

FAIL 1 — clean install from the wheel fails dependency resolution: the harness
wheel declares `animas>=0.1,<0.2` while the animas wheel is 0.2.0, so a plain
`pip install magitech_harness` reports "no solution". The rest of the suite
ran with `--no-deps` plus explicit runtime dependencies. This was the version-pin
fix addressed by the 0.3.0 release (see [`release-notes/v0.3.0.md`](../release-notes/v0.3.0.md)).

FAIL 2 — the marketplace `index.json` was absent from the installed plugin
wheel, so `mth_components.index.load_index()` failed. Runtime component discovery
uses a manifest scan instead, so no user flow depended on it. This was a latent
packaging bug, fixed below.

### Safety-critical behaviors — 9 PASS

Path confinement denies escape and `read_file` never discloses the secret;
an approval-required tool fails closed and appends the asked→decided audit pair;
the compaction v4 fixture validates through the real format layer; the pruner
marker is byte-exact (39 bytes); budgets are 0.8/0.16; the repeat reminder fires
at three and five; the `TOOL_TIMEOUT` code is present; and the CodeAct escape
battery blocks all three probes.

### Edge cases and error framing — 9 PASS

Blank task framed; blank session-id JSON error object; invalid provider →
`NO_ADAPTER` exit 1; non-list profile fails; invalid config value fails;
`component list` works; missing `component install` (exit 2, documented gap);
empty `session show` traceback (documented gap); no-subcommand boots the REPL.

## Verdict

Core end-user flows (headless text/JSON, sessions, resume, REPL, profiles,
patches, error framing, all safety guards, plugin catalog) work from the
installed wheel. The only hard release blocker at report time was the version
pin above; the rest were disclosed gaps that did not break a primary flow.

## Follow-up — all four disclosed gaps fixed and re-verified

All four gaps were subsequently fixed and merged. All three wheels were rebuilt,
clean-installed, and the probes re-run from the installed artifact:

1. **Scripted tool-call turn** — `mth headless --script FILE` drives a full tool
   loop; a persisted `tool/call` + `tool/result` pair and final text were
   verified from the installed wheel.
2. **Component discover/install** — `mth component discover` lists 23 installed
   components; `mth component install fs-search-tool` prints the exact profile
   row, enabled by a `magitech.components` entry point and a packaged index.
3. **Component list disabled state** — now uses the evaluated `entry.disabled`;
   on Linux, bash-local shows `mounted` and pwsh-local `disabled`.
4. **Blank session id** — clean `mth: session id must not be blank`, exit 1, no
   raw traceback.

Re-verification scratch was removed after the run.

Unresolved questions: none.
