"""Unit tests for LayerResolutionService (.agents protocol hierarchy)."""

import os
from pathlib import Path
from unittest.mock import patch

from anima import Context

from magitech_harness.components.layers import LayerResolutionService


def test_layer_resolution_defaults(tmp_path: Path) -> None:
    """Verifies default directory paths without overrides."""
    ctx = Context()
    service = LayerResolutionService(ctx)

    with patch.dict(os.environ, {}, clear=True):
        assert service.get_global_dir() == Path("/etc/magitech")

    assert service.get_user_dir() == Path.home() / ".agents"
    assert service.get_agent_dir(None) is None
    assert service.get_agent_dir("") is None
    assert (
        service.get_agent_dir("reviewer")
        == Path.home() / ".agents" / "agents" / "reviewer"
    )

    # Security invariant: project_path None or empty returns None, never ambient cwd
    assert service.get_project_dir(None) is None
    assert service.get_project_dir("") is None
    assert service.get_project_dir(tmp_path) == tmp_path / ".agents"


def test_layer_resolution_global_dir_env_override(tmp_path: Path) -> None:
    """Verifies MAGITECH_GLOBAL_DIR overrides default global directory."""
    ctx = Context()
    service = LayerResolutionService(ctx)
    custom_global = tmp_path / "custom_global"

    with patch.dict(os.environ, {"MAGITECH_GLOBAL_DIR": str(custom_global)}):
        assert service.get_global_dir() == custom_global


def test_layer_resolution_get_layers_tiers(tmp_path: Path) -> None:
    """Verifies get_layers returns appropriate ConfigLayer tiers in base-to-highest order."""
    ctx = Context()
    service = LayerResolutionService(ctx)

    # Base layers only (global + user)
    layers_base = service.get_layers()
    assert len(layers_base) == 2
    assert [layer.name for layer in layers_base] == ["global", "user"]

    # Base + Agent
    layers_agent = service.get_layers(agent_name="tester")
    assert len(layers_agent) == 3
    assert [layer.name for layer in layers_agent] == ["global", "user", "agent"]

    # Base + Agent + Project
    layers_all = service.get_layers(project_path=tmp_path, agent_name="tester")
    assert len(layers_all) == 4
    assert [layer.name for layer in layers_all] == [
        "global",
        "user",
        "agent",
        "project",
    ]
    assert layers_all[3].path == tmp_path / ".agents"


def test_layer_resolution_find_first_and_find_all(tmp_path: Path) -> None:
    """Verifies find_first (highest override) and find_all (all matches) precedence."""
    ctx = Context()
    service = LayerResolutionService(ctx)

    global_dir = tmp_path / "global"
    user_dir = tmp_path / "user"
    agent_dir = user_dir / "agents" / "coder"
    project_dir = tmp_path / "project" / ".agents"

    # Create directories
    global_dir.mkdir(parents=True)
    user_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)

    # Patch dirs on service
    with (
        patch.object(service, "get_global_dir", return_value=global_dir),
        patch.object(service, "get_user_dir", return_value=user_dir),
    ):
        # 1. Initially no files exist
        assert (
            service.find_first(
                "settings.json", project_path=tmp_path / "project", agent_name="coder"
            )
            is None
        )
        assert (
            service.find_all(
                "settings.json", project_path=tmp_path / "project", agent_name="coder"
            )
            == []
        )

        # 2. Add file to global
        (global_dir / "settings.json").write_text(
            '{"level": "global"}', encoding="utf-8"
        )
        found1 = service.find_first(
            "settings.json", project_path=tmp_path / "project", agent_name="coder"
        )
        assert found1 == global_dir / "settings.json"
        assert (
            len(
                service.find_all(
                    "settings.json",
                    project_path=tmp_path / "project",
                    agent_name="coder",
                )
            )
            == 1
        )

        # 3. Add file to user (overrides global)
        (user_dir / "settings.json").write_text('{"level": "user"}', encoding="utf-8")
        found2 = service.find_first(
            "settings.json", project_path=tmp_path / "project", agent_name="coder"
        )
        assert found2 == user_dir / "settings.json"
        assert (
            len(
                service.find_all(
                    "settings.json",
                    project_path=tmp_path / "project",
                    agent_name="coder",
                )
            )
            == 2
        )

        # 4. Add file to agent (overrides user)
        (agent_dir / "settings.json").write_text('{"level": "agent"}', encoding="utf-8")
        found3 = service.find_first(
            "settings.json", project_path=tmp_path / "project", agent_name="coder"
        )
        assert found3 == agent_dir / "settings.json"
        assert (
            len(
                service.find_all(
                    "settings.json",
                    project_path=tmp_path / "project",
                    agent_name="coder",
                )
            )
            == 3
        )

        # 5. Add file to project (highest precedence, overrides agent)
        (project_dir / "settings.json").write_text(
            '{"level": "project"}', encoding="utf-8"
        )
        found4 = service.find_first(
            "settings.json", project_path=tmp_path / "project", agent_name="coder"
        )
        assert found4 == project_dir / "settings.json"

        # find_all returns in base-to-highest order [global, user, agent, project]
        all_matches = service.find_all(
            "settings.json", project_path=tmp_path / "project", agent_name="coder"
        )
        assert all_matches == [
            global_dir / "settings.json",
            user_dir / "settings.json",
            agent_dir / "settings.json",
            project_dir / "settings.json",
        ]
