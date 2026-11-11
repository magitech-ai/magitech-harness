"""Unit tests for PluginService discovery, mounting, and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from anima import Context, FiberState

from magitech_harness.plugins.manifest import PluginManifest
from magitech_harness.plugins.service import (
    PluginService,
    _split_entry_point,
    resolve_entry_point,
)


@pytest.mark.asyncio
async def test_plugin_service_discover_and_mount_file(tmp_path: Path) -> None:
    """Verifies discovering manifest in directory and mounting plugin from file with explicit attr."""
    plugin_dir = tmp_path / "mock_plugin"
    plugin_dir.mkdir()

    code_file = plugin_dir / "plugin.py"
    code_file.write_text(
        "def custom_apply(ctx):\n    ctx.on('custom/event', lambda: 'handled')\n",
        encoding="utf-8",
    )

    manifest_data = {
        "name": "mock-plugin",
        "version": "0.1.0",
        "description": "Mock plugin for testing",
        "entry_point": "plugin.py:custom_apply",
        "types": ["Tool"],
        "enabled": True,
    }
    manifest_file = plugin_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    ctx = Context()
    service = PluginService(ctx)
    discovered = service.discover(tmp_path)
    assert len(discovered) == 1
    assert discovered[0].name == "mock-plugin"
    assert discovered[0].path == plugin_dir

    fiber = service.mount_plugin(discovered[0])
    await ctx.flush()

    assert fiber.state == FiberState.ACTIVE
    assert service.get_plugin("mock-plugin") is not None
    assert "mock-plugin" in service.mounted_plugins
    assert service.get_plugin("nonexistent") is None

    plugins = service.list_plugins()
    assert len(plugins) == 1
    assert plugins[0]["name"] == "mock-plugin"
    assert plugins[0]["state"] == "ACTIVE"

    # Unmount via service
    await service.unmount_plugin("mock-plugin")
    assert service.get_plugin("mock-plugin") is None
    assert service.list_plugins() == []

    # Unmount nonexistent does nothing
    await service.unmount_plugin("nonexistent")


def test_split_entry_point_drive_letter_colon_is_path_not_attr(tmp_path: Path) -> None:
    """Verifies a Windows drive-letter colon is not treated as the attr separator."""
    manifest = PluginManifest(
        name="drive-plugin", version="1.0.0", entry_point="x", path=tmp_path
    )
    assert _split_entry_point("C:plugin.py", manifest) == ("C:plugin.py", None)
    assert _split_entry_point("C:\\Users\\x\\plugin.py", manifest) == (
        "C:\\Users\\x\\plugin.py",
        None,
    )


def test_split_entry_point_file_attr_and_module_paths(tmp_path: Path) -> None:
    """Verifies splitting file:attr and module:attr entry points."""
    (tmp_path / "plugin.py").write_text("def apply(ctx): pass\n", encoding="utf-8")
    manifest = PluginManifest(
        name="split-plugin", version="1.0.0", entry_point="x", path=tmp_path
    )
    assert _split_entry_point("plugin.py:custom_apply", manifest) == (
        "plugin.py",
        "custom_apply",
    )
    assert _split_entry_point("plugin.py", manifest) == ("plugin.py", None)
    assert _split_entry_point("anima.core:Context", manifest) == (
        "anima.core:Context",
        None,
    )


@pytest.mark.asyncio
async def test_plugin_service_mount_file_with_default_apply(tmp_path: Path) -> None:
    """Verifies mounting plugin from file without attr when module has apply()."""
    code_file = tmp_path / "auto_apply.py"
    code_file.write_text(
        "def apply(ctx):\n    ctx.test_mounted = True\n",
        encoding="utf-8",
    )

    manifest = PluginManifest(
        name="auto-apply-plugin",
        version="1.0.0",
        entry_point=str(code_file),
    )

    ctx = Context()
    service = PluginService(ctx)
    fiber = service.mount_plugin(manifest)
    await ctx.flush()

    assert fiber.state == FiberState.ACTIVE
    assert getattr(ctx, "test_mounted", False) is True


@pytest.mark.asyncio
async def test_plugin_service_mount_file_with_default_plugin(tmp_path: Path) -> None:
    """Verifies mounting plugin from file without attr when module has plugin()."""
    code_file = tmp_path / "auto_plugin.py"
    code_file.write_text(
        "def plugin(ctx):\n    ctx.plugin_mounted = True\n",
        encoding="utf-8",
    )

    manifest = PluginManifest(
        name="auto-func-plugin",
        version="1.0.0",
        entry_point=str(code_file),
    )

    ctx = Context()
    service = PluginService(ctx)
    fiber = service.mount_plugin(manifest)
    await ctx.flush()

    assert fiber.state == FiberState.ACTIVE
    assert getattr(ctx, "plugin_mounted", False) is True


def test_plugin_service_mount_file_missing_attr(tmp_path: Path) -> None:
    """Verifies error when explicit attr is missing from module."""
    code_file = tmp_path / "missing_attr.py"
    code_file.write_text("x = 10\n", encoding="utf-8")

    manifest = PluginManifest(
        name="missing-attr-plugin",
        version="1.0.0",
        entry_point=f"{code_file}:nonexistent_fn",
    )

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent_fn'"):
        resolve_entry_point(manifest)


def test_plugin_service_mount_file_missing_default_entry_point(tmp_path: Path) -> None:
    """Verifies error when module has neither apply nor plugin and no attr given."""
    code_file = tmp_path / "no_entry.py"
    code_file.write_text("SOME_DATA = 42\n", encoding="utf-8")

    manifest = PluginManifest(
        name="no-entry-plugin",
        version="1.0.0",
        entry_point=str(code_file),
    )

    with pytest.raises(AttributeError, match="must define 'apply\\(ctx\\)'"):
        resolve_entry_point(manifest)


@pytest.mark.asyncio
async def test_plugin_service_mount_module_path() -> None:
    """Verifies mounting plugin using standard Python module import path."""
    manifest = PluginManifest(
        name="standard-module-plugin",
        version="1.0.0",
        entry_point="anima.core:Context",
    )

    ctx = Context()
    service = PluginService(ctx)
    fiber = service.mount_plugin(manifest)
    await ctx.flush()
    assert fiber.state == FiberState.ACTIVE


def test_resolve_entry_point_spec_failure(tmp_path: Path) -> None:
    """Verifies resolve_entry_point raises ImportError if spec cannot be loaded."""
    dummy_file = tmp_path / "dummy.py"
    dummy_file.write_text("x = 1\n", encoding="utf-8")

    manifest = PluginManifest(
        name="bad-spec",
        version="0.1.0",
        entry_point=str(dummy_file),
    )

    with (
        patch("importlib.util.spec_from_file_location", return_value=None),
        pytest.raises(ImportError, match="Cannot load module spec"),
    ):
        resolve_entry_point(manifest)


@pytest.mark.asyncio
async def test_plugin_service_load_directory(tmp_path: Path) -> None:
    """Verifies load_directory discovers and mounts only enabled plugins."""
    # Plugin 1: enabled
    p1 = tmp_path / "p1"
    p1.mkdir()
    (p1 / "plugin.py").write_text("def apply(ctx): pass\n", encoding="utf-8")
    (p1 / "manifest.json").write_text(
        json.dumps(
            {
                "name": "p1",
                "version": "1.0",
                "entry_point": "plugin.py",
                "enabled": True,
            }
        ),
        encoding="utf-8",
    )

    # Plugin 2: disabled
    p2 = tmp_path / "p2"
    p2.mkdir()
    (p2 / "plugin.py").write_text("def apply(ctx): pass\n", encoding="utf-8")
    (p2 / "manifest.json").write_text(
        json.dumps(
            {
                "name": "p2",
                "version": "1.0",
                "entry_point": "plugin.py",
                "enabled": False,
            }
        ),
        encoding="utf-8",
    )

    ctx = Context()
    service = PluginService(ctx)
    mounted_fibers = service.load_directory(tmp_path)
    await ctx.flush()

    assert len(mounted_fibers) == 1
    assert service.get_plugin("p1") is not None
    assert service.get_plugin("p2") is None


def test_plugin_service_discover_empty_or_invalid(tmp_path: Path) -> None:
    """Verifies discovery ignores non-directories, folders without manifest, or invalid JSON."""
    ctx = Context()
    service = PluginService(ctx)

    # Non-directory file in scan folder
    (tmp_path / "plain_file.txt").write_text("not a directory", encoding="utf-8")

    (tmp_path / "empty_dir").mkdir()

    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir()
    (bad_dir / "manifest.json").write_text("{invalid json", encoding="utf-8")

    incomplete_dir = tmp_path / "incomplete_dir"
    incomplete_dir.mkdir()
    (incomplete_dir / "manifest.json").write_text("{}", encoding="utf-8")

    discovered = service.discover(tmp_path)
    assert discovered == []


def test_plugin_service_discover_nonexistent_dir() -> None:
    """Verifies discovering in a nonexistent directory returns empty list."""
    ctx = Context()
    service = PluginService(ctx)
    assert service.discover("/nonexistent/plugins/path") == []
