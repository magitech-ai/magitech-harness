"""Layer resolution engine for the .agents protocol hierarchy.

Enforces 4-tier resolution: global -> user -> agent -> project.
Security invariant: Project layer is omitted when unspecified, never defaulting to ambient CWD.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from anima import Context, Service


@dataclass(frozen=True)
class ConfigLayer:
    """Represents a specific configuration tier in the resolution hierarchy."""

    name: str  # "global", "user", "agent", "project"
    path: Path
    exists: bool


class LayerResolutionService(Service):
    """Resolves and queries the 4-tier configuration hierarchy."""

    DEFAULT_GLOBAL_DIR: ClassVar[Path] = Path("/etc/magitech")

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "layers")

    def get_global_dir(self) -> Path:
        """Returns the global configuration directory path honoring $MAGITECH_GLOBAL_DIR."""
        env_val = os.environ.get("MAGITECH_GLOBAL_DIR")
        if env_val:
            return Path(env_val)
        return self.DEFAULT_GLOBAL_DIR

    def get_user_dir(self) -> Path:
        """Returns the user configuration directory path (~/.agents)."""
        return Path.home() / ".agents"

    def get_agent_dir(self, agent_name: str | None = None) -> Path | None:
        """Returns the agent profile directory path (~/.agents/agents/<name>)."""
        if not agent_name:
            return None
        return self.get_user_dir() / "agents" / agent_name

    def get_project_dir(self, project_path: Path | str | None = None) -> Path | None:
        """Returns the project layer directory (<project>/.agents).

        Security invariant: Omitted if project_path is None or empty.
        Never defaults to ambient Path.cwd().
        """
        if not project_path:
            return None
        p = Path(project_path)
        return p / ".agents"

    def get_layers(
        self,
        project_path: Path | str | None = None,
        agent_name: str | None = None,
    ) -> list[ConfigLayer]:
        """Returns active configuration layers in base-to-highest precedence order:
        global -> user -> agent -> project.
        """
        layers: list[ConfigLayer] = []

        global_p = self.get_global_dir()
        layers.append(ConfigLayer("global", global_p, global_p.exists()))

        user_p = self.get_user_dir()
        layers.append(ConfigLayer("user", user_p, user_p.exists()))

        agent_p = self.get_agent_dir(agent_name)
        if agent_p is not None:
            layers.append(ConfigLayer("agent", agent_p, agent_p.exists()))

        proj_p = self.get_project_dir(project_path)
        if proj_p is not None:
            layers.append(ConfigLayer("project", proj_p, proj_p.exists()))

        return layers

    def find_first(
        self,
        relative_path: str | Path,
        project_path: Path | str | None = None,
        agent_name: str | None = None,
    ) -> Path | None:
        """Finds the highest-precedence existing file matching relative_path.
        Search order: project -> agent -> user -> global.
        """
        for layer in reversed(self.get_layers(project_path, agent_name)):
            target = layer.path / relative_path
            if target.exists():
                return target
        return None

    def find_all(
        self,
        relative_path: str | Path,
        project_path: Path | str | None = None,
        agent_name: str | None = None,
    ) -> list[Path]:
        """Finds all existing files matching relative_path across active layers.
        Returned in base-to-highest precedence order (global -> user -> agent -> project).
        """
        found: list[Path] = []
        for layer in self.get_layers(project_path, agent_name):
            target = layer.path / relative_path
            if target.exists():
                found.append(target)
        return found
