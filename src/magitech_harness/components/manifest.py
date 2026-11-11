"""Component manifest schema and data structures.

This module defines the specification for portable Magitech component manifests.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ComponentManifest(BaseModel):
    """Specification schema for a Magitech component.

    Attributes:
        name: Unique name of the component (e.g. 'fs-tools').
        version: Semantic version string.
        description: Brief human-readable description of component capabilities.
        entry_point: Import target string (e.g. 'module:apply' or 'component.py:apply').
        types: List of capability categories provided (e.g. 'Tool', 'Provider').
        python_deps: List of required pip dependency specifications.
        hooks: List of lifecycle/waterfall hook names the component binds to.
        enabled: Whether the component is enabled by default.
        path: Optional directory path where the component and manifest reside.
    """

    name: str
    version: str
    entry_point: str
    description: str = ""
    types: list[str] = Field(default_factory=list)
    python_deps: list[str] = Field(default_factory=list)
    hooks: list[str] = Field(default_factory=list)
    requires_services: list[str] = Field(default_factory=list)
    recommended_components: list[str] = Field(default_factory=list)
    enabled: bool = True
    path: Path | None = None
