"""
SystemPromptService (ctx.system_prompt): Dynamic prompt assembly.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable
from kernel import Context, Disposer, Service


class SystemPromptService(Service):
    """Service that manages modular system prompt sections."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "system_prompt")
        self._sections: dict[str, Callable[[], str] | str] = {}

    def register_section(
        self, name: str, section: Callable[[], str] | str
    ) -> Disposer:
        """Register a dynamic prompt section."""
        self._sections[name] = section

        def _unregister():
            if name in self._sections:
                del self._sections[name]

        self.ctx.effect(lambda: _unregister)
        return _unregister

    async def render(self) -> str:
        """Assemble all sections into final system prompt string."""
        rendered: list[str] = []
        for name, sec in self._sections.items():
            if callable(sec):
                res = sec()
                if inspect.isawaitable(res):
                    res = await res
                rendered.append(str(res))
            else:
                rendered.append(str(sec))
        return "\n\n".join(rendered)
