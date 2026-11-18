"""Discover installed component packages via entry points.

The harness has no hard dependency on any plugin package. Installed plugins
advertise a ``magitech.components`` entry point whose callable returns an
iterable of component records. Discovery is fail-soft at that boundary: a
broken provider is logged and skipped, never crashing the CLI.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger("magitech.components.discovery")

GROUP = "magitech.components"


def discover_installed_components() -> list[dict[str, Any]]:
    """Return installed component records from registered entry points."""
    found: list[dict[str, Any]] = []
    for ep in entry_points(group=GROUP):
        try:
            items = ep.load()()
        except Exception as exc:  # noqa: BLE001 - fail-soft at plugin boundary
            logger.warning("Skipping component provider %s: %s", ep.name, exc)
            continue
        for item in items or ():
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                found.append(item)
    return found


__all__ = ["GROUP", "discover_installed_components"]
