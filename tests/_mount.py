"""Test helper: mount a component and return its fiber.

Animas's rc.13-compatible ``await fiber`` only observes a transition; it does
not drive one. Tests that need a component mounted must call ``load()``
explicitly. This helper keeps those call sites readable and preserves the
fiber as the return value.
"""

from __future__ import annotations

from typing import Any

from animas import Context


async def mount(ctx: Context, component: Any, *args: Any, **kwargs: Any) -> Any:
    fiber = ctx.use(component, *args, **kwargs)
    await fiber.load()
    return fiber
