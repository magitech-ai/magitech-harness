"""Probes for named profiles, boot-free dumps, and profile component lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from magitech_harness.profiles import launcher


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "home"
    monkeypatch.setenv(launcher.HOME_ENV, str(root))
    return root / launcher.PROFILE_DIR


def test_profiles_root_and_profile_dir_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    assert launcher.profiles_root() == home
    assert launcher.profile_dir("work") == home / "work"
    monkeypatch.delenv(launcher.HOME_ENV)
    assert launcher.profiles_root() == Path.home() / ".mth" / launcher.PROFILE_DIR
    with pytest.raises(ValueError, match="must not be blank"):
        launcher.profile_dir("   ")
    with pytest.raises(ValueError, match="reserved"):
        launcher.profile_dir("Desktop")


def test_prepare_profile_initializes_and_resets(home: Path) -> None:
    destination = launcher.prepare_profile("headless")
    assert (destination / launcher.PROFILE_FILE).read_text(
        encoding="utf-8"
    ).strip() == "[]"
    (destination / launcher.COMPONENTS_FILE).write_text("[]\n", encoding="utf-8")
    # First-use initialization does not overwrite an existing component list.
    destination = launcher.prepare_profile("headless")
    assert (destination / launcher.PROFILE_FILE).exists()

    custom = home / "custom"
    custom.mkdir(parents=True)
    (custom / launcher.PROFILE_FILE).write_text("[]\n", encoding="utf-8")
    (custom / launcher.COMPONENTS_FILE).write_text('["old"]\n', encoding="utf-8")
    launcher.prepare_profile("custom", from_default="base")
    assert (custom / launcher.COMPONENTS_FILE).read_text(
        encoding="utf-8"
    ).strip() == "[]"

    with pytest.raises(ValueError, match="needs a name"):
        launcher.prepare_profile("custom", from_default="  ")
    with pytest.raises(ValueError, match="unknown default profile template"):
        launcher.prepare_profile("custom", from_default="nope")
    with pytest.raises(ValueError, match="profile not found"):
        launcher.prepare_profile("missing")


def test_resolve_profile_source(tmp_path: Path, home: Path) -> None:
    assert launcher.resolve_profile_source(None) is None
    path = tmp_path / "overlay.yaml"
    path.write_text("[]\n", encoding="utf-8")
    assert launcher.resolve_profile_source(str(path)) == str(path)
    assert launcher.resolve_profile_source("relative/patch.yml") == "relative/patch.yml"
    resolved = launcher.resolve_profile_source("headless")
    assert resolved == home / "headless" / launcher.PROFILE_FILE
    assert Path(str(resolved)).exists()


def test_compose_and_dump_rows(tmp_path: Path, home: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "- id: commands\n  name: custom:Commands\n  config: {}\n", encoding="utf-8"
    )
    composed = launcher.compose_rows(str(overlay), [str(overlay)])
    commands = [row for row in composed if row.get("id") == "commands"]
    assert commands[0]["name"] == "custom:Commands"

    full = launcher.dump_config(str(overlay), [str(overlay)])
    assert full["schema"] == "mth-config-dump/v1"
    assert len(full["layers"]) == 3
    assert full["composed"] == composed

    defaults = launcher.dump_config(str(overlay), [str(overlay)], default_only=True)
    assert len(defaults["layers"]) == 1
    assert defaults["composed"] == launcher._base_rows()

    schema = launcher.dump_config_schema()
    assert schema["x-mth"]["complete"] is True
    assert "name" in schema["row"]["required"]
    assert json.dumps(schema)


def test_component_list_lifecycle(home: Path) -> None:
    assert launcher.read_components("work") == []
    assert launcher.edit_components("work", "add", "notion") == ["notion"]
    assert launcher.edit_components("work", "add", "notion") == ["notion"]
    assert launcher.edit_components("work", "add", "sample") == ["notion", "sample"]
    assert launcher.edit_components("work", "list") == ["notion", "sample"]
    assert launcher.edit_components("work", "show", "sample") == ["sample"]
    assert launcher.edit_components("work", "remove", "notion") == ["sample"]
    assert launcher.edit_components("work", "remove", "missing") == ["sample"]
    assert launcher.component_path("work") == home / "work" / launcher.COMPONENTS_FILE

    empty = home / "empty"
    empty.mkdir(parents=True)
    (empty / launcher.COMPONENTS_FILE).write_text("", encoding="utf-8")
    assert launcher.read_components("empty") == []
    (empty / launcher.COMPONENTS_FILE).write_text("- 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list of component names"):
        launcher.read_components("empty")

    with pytest.raises(ValueError, match="needs a component name"):
        launcher.edit_components("work", "add", "  ")
    with pytest.raises(ValueError, match="not installed"):
        launcher.edit_components("work", "show", "absent")
    with pytest.raises(ValueError, match="unknown component action"):
        launcher.edit_components("work", "bogus", "notion")


def test_prepare_profile_preserves_existing_files_on_first_use(home: Path) -> None:
    destination = home / "headless"
    destination.mkdir(parents=True)
    marker = destination / launcher.COMPONENTS_FILE
    marker.write_text('["keep"]\n', encoding="utf-8")
    launcher.prepare_profile("headless")
    assert marker.read_text(encoding="utf-8").strip() == '["keep"]'
    assert (destination / launcher.PROFILE_FILE).exists()


def test_copy_template_skips_directories(
    tmp_path: Any, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "templates" / "odd"
    source.mkdir(parents=True)
    (source / "nested").mkdir()
    (source / launcher.PROFILE_FILE).write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "TEMPLATES", tmp_path / "templates")
    monkeypatch.setattr(launcher, "SHIPPED_TEMPLATES", frozenset({"odd"}))
    destination = home / "odd"
    launcher.prepare_profile("odd")
    assert (destination / launcher.PROFILE_FILE).exists()
    assert not (destination / "nested").exists()


def test_prepare_profile_completes_a_component_created_directory(home: Path) -> None:
    destination = home / "component-first"
    destination.mkdir(parents=True)
    components = destination / launcher.COMPONENTS_FILE
    components.write_text('["notion"]\n', encoding="utf-8")
    prepared = launcher.prepare_profile("component-first")
    assert prepared == destination
    assert (destination / launcher.PROFILE_FILE).exists()
    assert components.read_text(encoding="utf-8").strip() == '["notion"]'
