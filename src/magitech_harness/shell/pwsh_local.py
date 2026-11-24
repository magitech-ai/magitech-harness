"""Local PowerShell executor over ``ctx.subprocess`` and executable probe."""

from __future__ import annotations

import ntpath
import os
import stat
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

from animas import Context

from magitech_harness.shell._subprocess_base import (
    SubprocessShellExecutor,
    assert_serviceable_config,
)
from magitech_harness.shell.types import ShellExecSpec, ShellExecution

ENV_OVERRIDES: Mapping[str, str] = MappingProxyType(
    {
        "NO_COLOR": "1",
        "PAGER": "cat",
        "GIT_PAGER": "cat",
    }
)

ENCODING_PREAMBLE = (
    "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
    "$OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
)


def candidate_pwsh_paths(env: Mapping[str, str] | None = None) -> list[str]:
    """Return candidate PowerShell executable paths on Windows in priority order."""
    source = os.environ if env is None else env
    program_files = source.get("ProgramFiles") or r"C:\Program Files"
    system_root = source.get("SystemRoot") or r"C:\Windows"
    candidates = [ntpath.join(program_files, "PowerShell", "7", "pwsh.exe")]
    for entry in (source.get("PATH") or "").split(";"):
        trimmed = entry.strip().strip('"')
        if not trimmed:
            continue
        candidates.append(ntpath.join(trimmed, "pwsh.exe"))
    candidates.append(
        ntpath.join(
            system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
        )
    )
    return candidates


def _candidate_exists(candidate: str) -> bool:
    try:
        st = os.lstat(candidate)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)


def resolve_pwsh_path(
    configured: str | None = None,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> str:
    """Resolve the PowerShell executable matching *configured*, Windows probes, or PATH."""
    if configured:
        return configured
    target_platform = sys.platform if platform is None else platform
    if target_platform == "win32":
        for candidate in candidate_pwsh_paths(env):
            if _candidate_exists(candidate):
                return candidate
    return "pwsh"


def assert_serviceable_pwsh_config(config: Mapping[str, Any]) -> None:
    """Validate live pwsh-local configuration budgets."""
    assert_serviceable_config("pwsh-local", config)


class LocalPwshExecutor(SubprocessShellExecutor):
    """Subprocess-backed ``pwsh -NoLogo -NoProfile -NonInteractive -Command`` provider."""

    _config_prefix: ClassVar[str] = "pwsh-local"
    _env_overrides: ClassVar[Mapping[str, str]] = ENV_OVERRIDES

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, config)
        self._declared_pwsh_path: str | None = self.config.get("pwshPath")
        self._resolved_pwsh_path = resolve_pwsh_path(self._declared_pwsh_path)

    @property
    def pwsh_path(self) -> str:
        """Resolved executable path; re-probes whenever ``config['pwshPath']`` changes."""
        declared = self.config.get("pwshPath")
        if declared != self._declared_pwsh_path:
            self._resolved_pwsh_path = resolve_pwsh_path(declared)
            self._declared_pwsh_path = declared
        return self._resolved_pwsh_path

    def argv(self, spec: ShellExecSpec) -> list[str]:
        """Construct the PowerShell command invocation argv for *spec*."""
        return [
            self.pwsh_path,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"{ENCODING_PREAMBLE}{spec['command']}",
        ]

    async def execute(self, spec: ShellExecSpec) -> ShellExecution:
        """Execute *spec* through PowerShell via ``ctx.subprocess``."""
        return await self.execute_argv(spec, self.argv(spec))


PwshLocalExecutor = LocalPwshExecutor

__all__ = [
    "ENCODING_PREAMBLE",
    "ENV_OVERRIDES",
    "LocalPwshExecutor",
    "PwshLocalExecutor",
    "assert_serviceable_pwsh_config",
    "candidate_pwsh_paths",
    "resolve_pwsh_path",
]
