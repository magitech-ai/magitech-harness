---
kind: upgrade-guide
description: Single headless command, layered profiles, scripted turns, and component discovery CLI.
---

# CLI and component discovery

## Change

The CLI surface changed in four externally perceptible ways:

- The verb-first `mth run` subcommand is gone. `mth headless "task"` is the
  single one-shot command; no alias is kept.
- Config overlays now layer as shipped bundle → `--profile` → each `--patch`
  in argv order. `--patch PATH` is repeatable and overrides the same row ids.
- `mth headless --script FILE` drives an offline, scripted LLM turn. The file is
  a JSON object with `turns`, each a list of raw stream chunks, and optional
  `provider`/`model`. It is intended for end-to-end tool-loop testing from an
  installed wheel.
- `mth component discover` lists components from installed packages, and
  `mth component install NAME` prints the exact profile row needed to mount
  one. Installation is read-only; no file is mutated.
- The shared agents root honors `$MAGITECH_AGENTS_HOME` (default `~/.agents`),
  alongside the harness home `$MAGITECH_HOME` (default `~/.magitech`).

## Migration

1. Replace any `mth run "task"` call with `mth headless "task"`. Output
   differs: `run` printed decorative labels; `headless` prints only the result,
   or a JSON object with `--json`, and returns exit 1 for a non-completed turn.
2. If you used a custom profile, nothing changes. If you previously patched
   component rows by hand, move them into a `--patch FILE` file and pass it
   after `--profile`; the last layer wins for a repeated row id.
3. To drive a scripted tool loop, write a `turns.json` file and run
   `mth headless --script turns.json "task"`. A script that ends before the
   loop settles fails loud rather than hanging.
4. To find and mount an installed component, run `mth component discover`,
   then `mth component install <name>` and add the printed row to your profile.
5. If you relied on the shared agents root at `~/.agents`, set
   `$MAGITECH_AGENTS_HOME` to relocate it. A blank value is treated as unset.

Verify with `mth --help`, `mth component discover`, and a `mth headless --json`
turn before integrating the new CLI into scripts.

Unresolved questions: none.
