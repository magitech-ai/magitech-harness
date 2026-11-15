"""Abstract SubprocessRuntime service definition (ctx.subprocess) and env scrubbing."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping

from anima import Context, Service

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.error import HarnessError
from magitech_harness.subprocess.types import (
    SENSITIVE_ENV_PATTERN,
    SubprocessHandle,
    SubprocessSpawnSpec,
)


class SubprocessExecutableNotFoundError(HarnessError):
    """Raised when an executable fails PATH or filesystem validation."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(
            message,
            "SUBPROCESS_EXECUTABLE_NOT_FOUND",
            cause=cause,
        )


def scrubbed_parent_env(
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy parent environment removing harness and provider secret variables."""
    source = os.environ if base_env is None else base_env
    return {
        key: value
        for key, value in source.items()
        if not SENSITIVE_ENV_PATTERN.match(key)
    }


def child_env(
    overrides: Mapping[str, str] | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build child environment by overlaying explicit *overrides* on scrubbed parent env."""
    env = scrubbed_parent_env(base_env)
    if overrides is not None:
        env.update(overrides)
    return env


class SubprocessRuntime(Service, ABC):
    """Abstract subprocess execution seam registering on ``ctx.subprocess``."""

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "subprocess")

    @abstractmethod
    async def resolve_executable(
        self,
        command: str,
        env: Mapping[str, str] | None = None,
        signal: AbortSignal | None = None,
    ) -> str:
        """Resolve *command* to an executable path or fail loud."""
        ...

    async def resolveExecutable(
        self,
        command: str,
        env: Mapping[str, str] | None = None,
        signal: AbortSignal | None = None,
    ) -> str:
        """CamelCase alias for :meth:`resolve_executable`."""
        return await self.resolve_executable(command, env, signal)

    @abstractmethod
    def spawn(self, spec: SubprocessSpawnSpec) -> SubprocessHandle:
        """Validate *spec* and spawn a managed subprocess handle."""
        ...


__all__ = [
    "SubprocessExecutableNotFoundError",
    "SubprocessRuntime",
    "child_env",
    "scrubbed_parent_env",
]
