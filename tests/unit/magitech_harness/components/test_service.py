"""Unit tests for ComponentService discovery, mounting, and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from anima import Context, FiberState

from magitech_harness.components.manifest import ComponentManifest
from magitech_harness.components.service import (
    ComponentService,
    _split_entry_point,
    resolve_entry_point,
)


@pytest.mark.asyncio
async def test_component_service_discover_and_mount_file(tmp_path: Path) -> None:
    """Verifies discovering manifest in directory and mounting component from file with explicit attr."""
    component_dir = tmp_path / "mock_component"
    component_dir.mkdir()

    code_file = component_dir / "component.py"
    code_file.write_text(
        "def custom_apply(ctx):\n    ctx.on('custom/event', lambda: 'handled')\n",
        encoding="utf-8",
    )

    manifest_data = {
        "name": "mock-component",
        "version": "0.1.0",
        "description": "Mock component for testing",
        "entry_point": "component.py:custom_apply",
        "types": ["Tool"],
        "enabled": True,
    }
    manifest_file = component_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    ctx = Context()
    service = ComponentService(ctx)
    discovered = service.discover(tmp_path)
    assert len(discovered) == 1
    assert discovered[0].name == "mock-component"
    assert discovered[0].path == component_dir

    fiber = service.mount_component(discovered[0])
    await ctx.flush()

    assert fiber.state == FiberState.ACTIVE
    assert service.get_component("mock-component") is not None
    assert "mock-component" in service.mounted_components
    assert service.get_component("nonexistent") is None

    components = service.list_components()
    assert len(components) == 1
    assert components[0]["name"] == "mock-component"
    assert components[0]["state"] == "ACTIVE"

    # Unmount via service
    await service.unmount_component("mock-component")
    assert service.get_component("mock-component") is None
    assert service.list_components() == []

    # Unmount nonexistent does nothing
    await service.unmount_component("nonexistent")


def test_split_entry_point_drive_letter_colon_is_path_not_attr(tmp_path: Path) -> None:
    """Verifies a Windows drive-letter colon is not treated as the attr separator."""
    manifest = ComponentManifest(
        name="drive-component", version="1.0.0", entry_point="x", path=tmp_path
    )
    assert _split_entry_point("C:component.py", manifest) == ("C:component.py", None)
    assert _split_entry_point("C:\\Users\\x\\component.py", manifest) == (
        "C:\\Users\\x\\component.py",
        None,
    )


def test_split_entry_point_file_attr_and_module_paths(tmp_path: Path) -> None:
    """Verifies splitting file:attr and module:attr entry points."""
    (tmp_path / "component.py").write_text("def apply(ctx): pass\n", encoding="utf-8")
    manifest = ComponentManifest(
        name="split-component", version="1.0.0", entry_point="x", path=tmp_path
    )
    assert _split_entry_point("component.py:custom_apply", manifest) == (
        "component.py",
        "custom_apply",
    )
    assert _split_entry_point("component.py", manifest) == ("component.py", None)
    assert _split_entry_point("anima.core:Context", manifest) == (
        "anima.core:Context",
        None,
    )


@pytest.mark.asyncio
async def test_component_service_mount_file_with_default_apply(tmp_path: Path) -> None:
    """Verifies mounting component from file without attr when module has apply()."""
    code_file = tmp_path / "auto_apply.py"
    code_file.write_text(
        "def apply(ctx):\n    ctx.test_mounted = True\n",
        encoding="utf-8",
    )

    manifest = ComponentManifest(
        name="auto-apply-component",
        version="1.0.0",
        entry_point=str(code_file),
    )

    ctx = Context()
    service = ComponentService(ctx)
    fiber = service.mount_component(manifest)
    await ctx.flush()

    assert fiber.state == FiberState.ACTIVE
    assert getattr(ctx, "test_mounted", False) is True


def test_component_service_mount_file_rejects_component_entry_point(
    tmp_path: Path,
) -> None:
    """Verifies a module defining only component() is not a valid entry point.

    The old module-level entry point was removed by the clean rename with
    no alias: only apply() (or an explicit file.py:attr) resolves.
    """
    code_file = tmp_path / "auto_component.py"
    code_file.write_text(
        "def component(ctx):\n    ctx.component_mounted = True\n",
        encoding="utf-8",
    )

    manifest = ComponentManifest(
        name="auto-func-component",
        version="1.0.0",
        entry_point=str(code_file),
    )

    with pytest.raises(AttributeError, match="must define 'apply\\(ctx\\)'"):
        resolve_entry_point(manifest)


def test_component_service_mount_file_missing_attr(tmp_path: Path) -> None:
    """Verifies error when explicit attr is missing from module."""
    code_file = tmp_path / "missing_attr.py"
    code_file.write_text("x = 10\n", encoding="utf-8")

    manifest = ComponentManifest(
        name="missing-attr-component",
        version="1.0.0",
        entry_point=f"{code_file}:nonexistent_fn",
    )

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent_fn'"):
        resolve_entry_point(manifest)


def test_component_service_mount_file_missing_default_entry_point(
    tmp_path: Path,
) -> None:
    """Verifies error when module has no apply() and no attr given."""
    code_file = tmp_path / "no_entry.py"
    code_file.write_text("SOME_DATA = 42\n", encoding="utf-8")

    manifest = ComponentManifest(
        name="no-entry-component",
        version="1.0.0",
        entry_point=str(code_file),
    )

    with pytest.raises(AttributeError, match="must define 'apply\\(ctx\\)'"):
        resolve_entry_point(manifest)


@pytest.mark.asyncio
async def test_component_service_mount_module_path() -> None:
    """Verifies mounting component using standard Python module import path."""
    manifest = ComponentManifest(
        name="standard-module-component",
        version="1.0.0",
        entry_point="anima.core:Context",
    )

    ctx = Context()
    service = ComponentService(ctx)
    fiber = service.mount_component(manifest)
    await ctx.flush()
    assert fiber.state == FiberState.ACTIVE


def test_resolve_entry_point_spec_failure(tmp_path: Path) -> None:
    """Verifies resolve_entry_point raises ImportError if spec cannot be loaded."""
    dummy_file = tmp_path / "dummy.py"
    dummy_file.write_text("x = 1\n", encoding="utf-8")

    manifest = ComponentManifest(
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
async def test_component_service_load_directory(tmp_path: Path) -> None:
    """Verifies load_directory discovers and mounts only enabled components."""
    # Component 1: enabled
    p1 = tmp_path / "p1"
    p1.mkdir()
    (p1 / "component.py").write_text("def apply(ctx): pass\n", encoding="utf-8")
    (p1 / "manifest.json").write_text(
        json.dumps(
            {
                "name": "p1",
                "version": "1.0",
                "entry_point": "component.py",
                "enabled": True,
            }
        ),
        encoding="utf-8",
    )

    # Component 2: disabled
    p2 = tmp_path / "p2"
    p2.mkdir()
    (p2 / "component.py").write_text("def apply(ctx): pass\n", encoding="utf-8")
    (p2 / "manifest.json").write_text(
        json.dumps(
            {
                "name": "p2",
                "version": "1.0",
                "entry_point": "component.py",
                "enabled": False,
            }
        ),
        encoding="utf-8",
    )

    ctx = Context()
    service = ComponentService(ctx)
    mounted_fibers = service.load_directory(tmp_path)
    await ctx.flush()

    assert len(mounted_fibers) == 1
    assert service.get_component("p1") is not None
    assert service.get_component("p2") is None


def test_component_service_discover_empty_or_invalid(tmp_path: Path) -> None:
    """Verifies discovery ignores non-directories, folders without manifest, or invalid JSON."""
    ctx = Context()
    service = ComponentService(ctx)

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


def test_component_service_discover_nonexistent_dir() -> None:
    """Verifies discovering in a nonexistent directory returns empty list."""
    ctx = Context()
    service = ComponentService(ctx)
    assert service.discover("/nonexistent/components/path") == []
