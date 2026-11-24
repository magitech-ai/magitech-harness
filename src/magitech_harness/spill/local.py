"""Local-filesystem spill backend.

Persists oversized text to a private, session-scoped file (see :mod:`store`
for the traversal-safe naming and exclusive owner-only write) and returns a
path locator plus local read/grep retrieval guidance.

After activation it runs one best-effort startup cleanup sweep that reclaims
spill files older than ``cleanupPeriodDays``.

Local filesystem spill backend.
"""

from __future__ import annotations

import logging
import os
from typing import TypedDict

from animas import Context

from magitech_harness.spill.service import (
    SaveTextSpill,
    SpillLocator,
    SpillRef,
    SpillStore,
)
from magitech_harness.spill.store import privateRoot

logger = logging.getLogger("magitech_harness.spill")

__all__ = [
    "Config",
    "LocalSpillStore",
    "ResolvedConfig",
]


class Config(TypedDict, total=False):
    """Plugin config (all optional)."""

    root: str
    cleanupPeriodDays: int


class ResolvedConfig(TypedDict):
    """The shape after schemastery-style defaults are applied."""

    root: str
    cleanupPeriodDays: int


class LocalSpillStore(SpillStore):
    """Local-filesystem spill backend.

    Files land under ``<root>/session-<hash>/…`` with unpredictable names, an
    exclusive owner-only (0600) write, and a private (0700) root — a spilled
    tool result must not be readable by other local users or redirectable via a
    planted symlink.

    After activation it launches one best-effort cleanup sweep (see
    ``cleanupPeriodDays``) that reclaims expired spill files without delaying
    service availability.
    """

    def __init__(self, ctx: Context, config: Config | None = None) -> None:
        super().__init__(ctx)
        cfg = config or {}
        root = cfg.get("root", "")
        cleanup_days = cfg.get("cleanupPeriodDays", 30)
        if (
            not isinstance(cleanup_days, int)
            or isinstance(cleanup_days, bool)
            or cleanup_days < 0
        ):
            raise ValueError(
                f"spill-local: cleanupPeriodDays must be a non-negative integer "
                f"(got {cleanup_days!r})"
            )
        self.config: ResolvedConfig = {
            "root": root,
            "cleanupPeriodDays": cleanup_days,
        }
        self.root = os.path.abspath(root) if root else privateRoot()

    async def saveText(self, input: SaveTextSpill) -> SpillRef:
        """Persist ``input.content`` to a session-scoped spill artifact."""
        from magitech_harness.spill.store import saveTextFile

        saved = saveTextFile(
            {
                "root": self.root,
                "sessionId": input["owner"]["sessionId"],
                "suggestedName": input["suggestedName"],
                "content": input["content"],
            }
        )
        return SpillRef(
            locator=SpillLocator(saved["path"]),
            bytes=saved["bytes"],
            retrievalHint=(
                "Use read with offset/limit, or grep this path to search within it."
            ),
        )
