"""PluginService (ctx.plugins): Discovery and mounting of portable Magitech plugins.

This service manages the discovery, lifecycle, and status tracking of plugins
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
from anima.loader import resolve_plugin_target
from pydantic import ValidationError

from magitech_harness.plugins.manifest import PluginManifest

logger = logging.getLogger("magitech.plugins")


def _entry_file(file_part: str, manifest: PluginManifest) -> Path | None:
    """Return the file an entry-point file part names, if it exists.

    Checks the manifest directory first, then the path as given.
    """
    if manifest.path and (manifest.path / file_part).is_file():
        return manifest.path / file_part
    if Path(file_part).is_file():
        return Path(file_part)
    return None


def _split_entry_point(
    target_str: str, manifest: PluginManifest
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


def resolve_entry_point(manifest: PluginManifest) -> Any:
    """Resolves a plugin target from manifest entry_point.

    Supports both file paths (relative to manifest.path or absolute)
    and standard Python module import paths.

    Args:
        manifest: PluginManifest defining entry_point and optional path.

    Returns:
        Callable, class, or module target suitable for Context.plugin().

    Raises:
        ImportError: If the entry point file or module cannot be imported.
        AttributeError: If the specified attribute or default entry point is missing.
    """
    target_str = manifest.entry_point
    file_part, attr_name = _split_entry_point(target_str, manifest)

    file_path = _entry_file(file_part, manifest)

    if file_path is not None:
        clean_name = manifest.name.replace("-", "_").replace(".", "_")
        mod_name = f"magitech_plugins_{clean_name}"
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
        if hasattr(mod, "plugin"):
            return mod.plugin
        raise AttributeError(
            f"Plugin entry point in '{file_path}' must define 'apply(ctx)' or 'plugin(ctx)' "
            f"or specify an attribute via 'file.py:function_name'"
        )

    return resolve_plugin_target(target_str)


class PluginService(Service):
    """Service managing discovery and mounting of portable plugins."""

    def __init__(self, ctx: Context) -> None:
        """Initializes PluginService on the context."""
        super().__init__(ctx, "plugins")
        self._mounted_plugins: dict[str, tuple[PluginManifest, Fiber]] = {}

    def discover(self, directory: Path | str) -> list[PluginManifest]:
        """Discovers all valid plugins within a given directory.

        Scans immediate subdirectories for a ``manifest.json`` file,
        validating each against the PluginManifest schema.

        Args:
            directory: Root directory path to scan.

        Returns:
            List of successfully validated PluginManifest instances.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        discovered: list[PluginManifest] = []
        for child in sorted(dir_path.iterdir()):
            if child.is_dir():
                manifest_file = child / "manifest.json"
                if manifest_file.exists():
                    try:
                        content = manifest_file.read_text(encoding="utf-8")
                        data = json.loads(content)
                        data["path"] = child
                        manifest = PluginManifest.model_validate(data)
                        discovered.append(manifest)
                    except (json.JSONDecodeError, ValidationError, OSError) as e:
                        logger.warning(
                            "Skipping invalid plugin manifest at %s: %s",
                            manifest_file,
                            e,
                        )
        return discovered

    def mount_plugin(self, manifest: PluginManifest, **config: Any) -> Fiber:
        """Mounts a validated plugin manifest into the Anima context.

        Args:
            manifest: The plugin manifest specification.
            **config: Configuration kwargs to pass to the plugin.

        Returns:
            The loaded Anima Fiber instance.
        """
        target = resolve_entry_point(manifest)
        fiber = self.ctx.plugin(target, **config)
        self._mounted_plugins[manifest.name] = (manifest, fiber)

        def _unregister() -> None:
            self._mounted_plugins.pop(manifest.name, None)

        fiber.effects.append(_unregister)
        return fiber

    async def unmount_plugin(self, name: str) -> None:
        """Unmounts and disposes a currently mounted plugin by name.

        Args:
            name: Unique name of the plugin to unmount.
        """
        mounted = self._mounted_plugins.get(name)
        if mounted is not None:
            _, fiber = mounted
            await fiber.dispose()

    @property
    def mounted_plugins(self) -> dict[str, tuple[PluginManifest, Fiber]]:
        """Returns a snapshot dictionary of currently mounted plugins."""
        return dict(self._mounted_plugins)

    def get_plugin(self, name: str) -> tuple[PluginManifest, Fiber] | None:
        """Returns the manifest and fiber for a mounted plugin, or None.

        Args:
            name: Unique name of the plugin.

        Returns:
            Tuple of (PluginManifest, Fiber) or None if not mounted.
        """
        return self._mounted_plugins.get(name)

    def load_directory(self, directory: Path | str, **config: Any) -> list[Fiber]:
        """Discovers and mounts all enabled plugins found in directory.

        Args:
            directory: Root directory path containing plugin folders.
            **config: Shared kwargs to pass to mounted plugins.

        Returns:
            List of successfully mounted Fiber instances.
        """
        manifests = self.discover(directory)
        fibers: list[Fiber] = []
        for manifest in manifests:
            if manifest.enabled:
                fiber = self.mount_plugin(manifest, **config)
                fibers.append(fiber)
        return fibers

    def list_plugins(self) -> list[dict[str, Any]]:
        """Returns a list of status summaries for all currently mounted plugins."""
        result: list[dict[str, Any]] = []
        for name, (manifest, fiber) in self._mounted_plugins.items():
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
