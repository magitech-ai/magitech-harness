"""ContextContributor service (ctx.context): pluggable context contributors."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from animas import Context, Disposer, Service
from animas.core import _access_context

logger = logging.getLogger(__name__)


class ContextContributor(Protocol):
    """Plugin contribution to the agent request context."""

    @property
    def name(self) -> str:
        """Stable contributor name used in diagnostics."""

    def contribute(self, request: Mapping[str, Any]) -> str | None:
        """Return context text to inject, or None to skip."""


@dataclass
class _ContributorRecord:
    name: str
    contributor: ContextContributor


class ContextService(Service):
    """Service (ctx.context) managing pluggable context contributors."""

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "context")
        self._contributors: dict[str, _ContributorRecord] = {}

    def register(self, contributor: ContextContributor) -> Disposer:
        """Register one contributor on the calling fiber."""
        caller_ctx = _access_context.get() or self.ctx

        def _setup() -> Callable[[], None]:
            name = contributor.name
            if not name or not name.strip():
                raise ValueError("context contributor name must be non-empty")
            if name in self._contributors:
                raise ValueError(f'context contributor "{name}" is already registered')
            self._contributors[name] = _ContributorRecord(
                name=name, contributor=contributor
            )

            def _teardown() -> None:
                self._contributors.pop(name, None)

            return _teardown

        return caller_ctx.effect(_setup, label=f"context.register({contributor.name})")

    def list_contributors(self) -> list[str]:
        """Return the names of all registered contributors."""
        return sorted(self._contributors)

    def contribute(self, request: Mapping[str, Any]) -> list[str]:
        """Collect context text from all contributors."""
        out: list[str] = []
        for record in self._contributors.values():
            try:
                text = record.contributor.contribute(request)
            except Exception:
                logger.exception("context contributor %s failed", record.name)
                continue
            if isinstance(text, str) and text.strip():
                out.append(text)
        return out
