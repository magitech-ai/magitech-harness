"""Trusted per-execution MAGITECH_* environment registry package."""

from __future__ import annotations

from magitech_harness.shell_env.service import (
    MAGITECH_AGENTS_HOME_ENV,
    MAGITECH_HOME_ENV,
    MAGITECH_PROFILE_DIR_KEY,
    MAGITECH_PROFILE_KEY,
    MAGITECH_SESSION_ID_KEY,
    MAGITECH_SHELL_KEY,
    RESERVED_SHELL_ENV_KEYS,
    ShellEnvContributor,
    ShellEnvRegistry,
    ShellEnvVariable,
    ShellEnvVariableInfo,
    apply,
    resolve_magitech_agents_home,
    resolve_magitech_home,
)

__all__ = [
    "MAGITECH_AGENTS_HOME_ENV",
    "MAGITECH_HOME_ENV",
    "MAGITECH_PROFILE_DIR_KEY",
    "MAGITECH_PROFILE_KEY",
    "MAGITECH_SESSION_ID_KEY",
    "MAGITECH_SHELL_KEY",
    "RESERVED_SHELL_ENV_KEYS",
    "ShellEnvContributor",
    "ShellEnvRegistry",
    "ShellEnvVariable",
    "ShellEnvVariableInfo",
    "apply",
    "resolve_magitech_agents_home",
    "resolve_magitech_home",
]
