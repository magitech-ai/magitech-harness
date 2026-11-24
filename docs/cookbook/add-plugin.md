# Add a plugin

## Prerequisites

- A component module importable on the Python path with an `apply(ctx, config=None, **kwargs)` entry point.
- A domain tool component in `mth-plugins` (the harness core intentionally owns no domain tools).

## Steps

1. Define the component entry point. It registers on `ctx`, returns nothing, and
   mounts any listener with a disposer cleaned up via `ctx.effect` or by
   registering a disposer on the fiber:

   ```python
   # my_package/my_component.py
   from animas import Context, EventMode

   def apply(ctx: Context, config: dict | None = None, **kwargs) -> None:
       def on_result(view, result, next_fn):
           return next_fn(view, result)

       ctx.on("tools/post-execute", on_result, mode=EventMode.WATERFALL)
   ```

2. Declare a row for the component with a stable id and the `module:function`
   name:

   ```yaml
   - id: my-component
     name: my_package.my_component:apply
     config: {}
   ```

3. Load the row via a [profile](write-profile.md) (an unknown id appends it after
   the base rows):

   ```sh
   uv run mth --storage .sessions --profile my-profile.yaml repl
   ```

4. If the component declares `apply.inject = ["tools"]`, place its row after the
   `tools` row so the injection resolves.

## Verify

`mth component list` shows `my-component`, and the listener fires on a tool call
during a turn. A registration that fails to resolve its injection fails at boot
rather than at runtime.

**Warning:** every registration is an effect. A listener that leaks its disposer
will double-fire on reload. A misconfigured row fails loud; there is no silent
skip. See [architecture](../architecture.md#capability-seams-and-extension-points).
