"""Local bash executor over ``ctx.subprocess``."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

from magitech_harness.shell._subprocess_base import (
    SubprocessShellExecutor,
    assert_serviceable_config,
)
from magitech_harness.shell.types import ShellExecSpec, ShellExecution

ENV_OVERRIDES: Mapping[str, str] = MappingProxyType(
    {
        "NO_COLOR": "1",
        "TERM": "dumb",
        "PAGER": "cat",
        "GIT_PAGER": "cat",
    }
)


def assert_serviceable_bash_config(config: Mapping[str, Any]) -> None:
    """Validate live bash-local configuration budgets."""
    assert_serviceable_config("bash-local", config)


class LocalBashExecutor(SubprocessShellExecutor):
    """Subprocess-backed ``bash -c`` provider for ``ctx.shell``."""

    _config_prefix: ClassVar[str] = "bash-local"
    _env_overrides: ClassVar[Mapping[str, str]] = ENV_OVERRIDES

    async def execute(self, spec: ShellExecSpec) -> ShellExecution:
        """Execute *spec* through ``bash -c`` via ``ctx.subprocess``."""
        return await self.execute_argv(spec, ["bash", "-c", spec["command"]])


__all__ = [
    "ENV_OVERRIDES",
    "LocalBashExecutor",
    "assert_serviceable_bash_config",
]
