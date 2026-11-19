# Write a profile

## Prerequisites

- Familiarity with the base rows in
  [`profiles/base.yaml`](../../src/magitech_harness/profiles/base.yaml).

## Steps

1. A profile is a YAML list of component rows. Each row has `id`, `name`, and
   `config`:

   ```yaml
   # my-profile.yaml
   - id: approval
     name: magitech_harness.approval.service:ApprovalService
     config:
       policy: never
   ```

2. Overlay an existing row by repeating its `id`; unknown ids are appended as new
   rows. Here the approval policy is overridden without redefining any other
   service:

   ```python
   from magitech_harness import create_harness

   ctx = create_harness(storage_dir=".sessions", profile="my-profile.yaml")
   await ctx.flush()
   ```

3. From the CLI, pass the same file:

   ```sh
   uv run mth --storage .sessions --profile my-profile.yaml repl
   ```

4. A profile can also be supplied in code as a list of row dicts, which is useful
   for tests:

   ```python
   profile = [{"id": "approval", "name": "...:ApprovalService", "config": {"policy": "never"}}]
   ```

## Verify

`ctx.approval.policy` reflects the overridden value, and `mth component list`
shows the effective rows.

**Warning:** duplicate ids, non-string ids, and rows without a resolvable `name`
fail at boot; there is no silent fallback. Unknown config keys fail at the owning
service. See [configuration](../subsystems/configuration.md).

**[Python]** The composition overlay chain is shipped bundle → `--profile` →
each `--patch` in argv order; see [configuration](../subsystems/configuration.md#bundle-and-profile-composition-overlay-chain).
`--patch` is repeatable and applied after the profile; overlay flags precede the
subcommand (for example `mth --patch p.yaml headless "task"`).
