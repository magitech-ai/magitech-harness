"""CompactionService (ctx.compaction): context compaction and pruning seam."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from animas import Context, Disposer, Service
from animas.core import _access_context

logger = logging.getLogger(__name__)


class CompactionService(Service):
    """Service (ctx.compaction) coordinating context compaction and pruning."""

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "compaction")
        self._engines: dict[str, Callable[[Any], Any]] = {}
        self._pruners: dict[str, Callable[[Any], Any]] = {}

    def register_engine(self, name: str, engine: Callable[[Any], Any]) -> Disposer:
        """Register a compaction engine on the calling fiber."""
        caller_ctx = _access_context.get() or self.ctx

        def _setup() -> Callable[[], None]:
            if not name or not name.strip():
                raise ValueError("compaction engine name must be non-empty")
            if name in self._engines:
                raise ValueError(f'compaction engine "{name}" is already registered')
            self._engines[name] = engine

            def _teardown() -> None:
                self._engines.pop(name, None)

            return _teardown

        return caller_ctx.effect(_setup, label=f"compaction.registerEngine({name})")

    def register_pruner(self, name: str, pruner: Callable[[Any], Any]) -> Disposer:
        """Register a context pruner on the calling fiber."""
        caller_ctx = _access_context.get() or self.ctx

        def _setup() -> Callable[[], None]:
            if not name or not name.strip():
                raise ValueError("pruner name must be non-empty")
            if name in self._pruners:
                raise ValueError(f'pruner "{name}" is already registered')
            self._pruners[name] = pruner

            def _teardown() -> None:
                self._pruners.pop(name, None)

            return _teardown

        return caller_ctx.effect(_setup, label=f"compaction.registerPruner({name})")

    def list_engines(self) -> list[str]:
        """Return the names of all registered compaction engines."""
        return sorted(self._engines)

    def list_pruners(self) -> list[str]:
        """Return the names of all registered pruners."""
        return sorted(self._pruners)

    async def compact(self, session: Any, **kwargs: Any) -> dict[str, Any]:
        """Run all registered compaction engines over *session*."""
        self.ctx.emit("compaction/start", {"session": session})
        result: dict[str, Any] = {"compacted": False}
        for name, engine in self._engines.items():
            try:
                out = engine(session, **kwargs)
                if hasattr(out, "__await__"):
                    out = await out
                if isinstance(out, dict):
                    result.update(out)
            except Exception:
                logger.exception("compaction engine %s failed", name)
                continue
        self.ctx.emit("compaction/end", result)
        return result

    async def prune(self, session: Any, **kwargs: Any) -> dict[str, Any]:
        """Run all registered pruners over *session*."""
        self.ctx.emit("compaction/prune", {"session": session})
        result: dict[str, Any] = {"pruned": False}
        for name, pruner in self._pruners.items():
            try:
                out = pruner(session, **kwargs)
                if hasattr(out, "__await__"):
                    out = await out
                if isinstance(out, dict):
                    result.update(out)
            except Exception:
                logger.exception("pruner %s failed", name)
                continue
        return result
