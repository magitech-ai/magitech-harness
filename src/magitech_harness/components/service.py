"""ComponentService (ctx.components): Discovery and mounting of portable Magitech components.

This service manages the discovery, lifecycle, and status tracking of components
within the Anima context.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

from anima import Context, Fiber, Service
from anima.loader import resolve_component_target
from pydantic import ValidationError

from magitech_harness.components.manifest import ComponentManifest

logger = logging.getLogger("magitech.components")


def _entry_file(file_part: str, manifest: ComponentManifest) -> Path | None:
    """Return the file an entry-point file part names, if it exists.

    Checks the manifest directory first, then the path as given.
    """
    if manifest.path and (manifest.path / file_part).is_file():
        return manifest.path / file_part
    if Path(file_part).is_file():
        return Path(file_part)
    return None


def _split_entry_point(
    target_str: str, manifest: ComponentManifest
) -> tuple[str, str | None]:
    """Split an entry point into (file part, attr name).

    The colon after a Windows drive letter (``C:\\...``) is part of the
    path, not the attr separator: only split on the last colon when the
    part before it names an existing file.
    """
    if ":" not in target_str:
        return target_str, None
    candidate, _, attr_name = target_str.rpartition(":")
    if _entry_file(candidate, manifest) is not None:
        return candidate, attr_name
    return target_str, None


def resolve_entry_point(manifest: ComponentManifest) -> Any:
    """Resolves a component target from manifest entry_point.

    Supports both file paths (relative to manifest.path or absolute)
    and standard Python module import paths.

    Args:
        manifest: ComponentManifest defining entry_point and optional path.

    Returns:
        Callable, class, or module target suitable for Context.use().

    Raises:
        ImportError: If the entry point file or module cannot be imported.
        AttributeError: If the specified attribute or default entry point is missing.
    """
    target_str = manifest.entry_point
    file_part, attr_name = _split_entry_point(target_str, manifest)

    file_path = _entry_file(file_part, manifest)

    if file_path is not None:
        clean_name = manifest.name.replace("-", "_").replace(".", "_")
        mod_name = f"magitech_components_{clean_name}"
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

        if attr_name:
            if not hasattr(mod, attr_name):
                raise AttributeError(
                    f"Module '{mod_name}' has no attribute '{attr_name}'"
                )
            return getattr(mod, attr_name)
        if hasattr(mod, "apply"):
            return mod.apply
        raise AttributeError(
            f"Component entry point in '{file_path}' must define 'apply(ctx)' "
            f"or specify an attribute via 'file.py:function_name'"
        )

    return resolve_component_target(target_str)


class ComponentService(Service):
    """Service managing discovery and mounting of portable components."""

    def __init__(self, ctx: Context) -> None:
        """Initializes ComponentService on the context."""
        super().__init__(ctx, "components")
        self._mounted_components: dict[str, tuple[ComponentManifest, Fiber]] = {}

    def discover(self, directory: Path | str) -> list[ComponentManifest]:
        """Discovers all valid components within a given directory.

        Scans immediate subdirectories for a ``manifest.json`` file,
        validating each against the ComponentManifest schema.

        Args:
            directory: Root directory path to scan.

        Returns:
            List of successfully validated ComponentManifest instances.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        discovered: list[ComponentManifest] = []
        for child in sorted(dir_path.iterdir()):
            if child.is_dir():
                manifest_file = child / "manifest.json"
                if manifest_file.exists():
                    try:
                        content = manifest_file.read_text(encoding="utf-8")
                        data = json.loads(content)
                        data["path"] = child
                        manifest = ComponentManifest.model_validate(data)
                        discovered.append(manifest)
                    except (json.JSONDecodeError, ValidationError, OSError) as e:
                        logger.warning(
                            "Skipping invalid component manifest at %s: %s",
                            manifest_file,
                            e,
                        )
        return discovered

    def mount_component(self, manifest: ComponentManifest, **config: Any) -> Fiber:
        """Mounts a validated component manifest into the Anima context.

        Args:
            manifest: The component manifest specification.
            **config: Configuration kwargs to pass to the component.

        Returns:
            The loaded Anima Fiber instance.
        """
        target = resolve_entry_point(manifest)
        # Mount on the root context, not this service's own context: as
        # a bundle row the service lives on a loader entry context, and
        # a component mounted there settles only when that entry
        # context flushes, which a root flush never awaits. The root is
        # the context callers flush, so mounting there restores the
        # settle contract: active after one root flush.
        fiber = self.ctx.root.use(target, **config)
        self._mounted_components[manifest.name] = (manifest, fiber)

        def _unregister() -> None:
            self._mounted_components.pop(manifest.name, None)

        fiber.effects.append(_unregister)
        return fiber

    async def unmount_component(self, name: str) -> None:
        """Unmounts and disposes a currently mounted component by name.

        Args:
            name: Unique name of the component to unmount.
        """
        mounted = self._mounted_components.get(name)
        if mounted is not None:
            _, fiber = mounted
            await fiber.dispose()

    @property
    def mounted_components(self) -> dict[str, tuple[ComponentManifest, Fiber]]:
        """Returns a snapshot dictionary of currently mounted components."""
        return dict(self._mounted_components)

    def get_component(self, name: str) -> tuple[ComponentManifest, Fiber] | None:
        """Returns the manifest and fiber for a mounted component, or None.

        Args:
            name: Unique name of the component.

        Returns:
            Tuple of (ComponentManifest, Fiber) or None if not mounted.
        """
        return self._mounted_components.get(name)

    def load_directory(self, directory: Path | str, **config: Any) -> list[Fiber]:
        """Discovers and mounts all enabled components found in directory.

        Args:
            directory: Root directory path containing component folders.
            **config: Shared kwargs to pass to mounted components.

        Returns:
            List of successfully mounted Fiber instances.
        """
        manifests = self.discover(directory)
        fibers: list[Fiber] = []
        for manifest in manifests:
            if manifest.enabled:
                fiber = self.mount_component(manifest, **config)
                fibers.append(fiber)
        return fibers

    def list_components(self) -> list[dict[str, Any]]:
        """Returns a list of status summaries for all currently mounted components."""
        result: list[dict[str, Any]] = []
        for name, (manifest, fiber) in self._mounted_components.items():
            result.append(
                {
                    "name": name,
                    "version": manifest.version,
                    "description": manifest.description,
                    "types": manifest.types,
                    "state": fiber.state.name,
                }
            )
        return result
