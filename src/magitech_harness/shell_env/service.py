"""Trusted per-execution MAGITECH_* environment variable registry."""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypedDict, cast

from anima import Context, Service
from anima.core import _access_context

from magitech_harness.subprocess.types import MAGITECH_ENV_PREFIX

MAGITECH_HOME_ENV = "MAGITECH_HOME"
MAGITECH_AGENTS_HOME_ENV = "MAGITECH_AGENTS_HOME"
MAGITECH_SHELL_KEY = f"{MAGITECH_ENV_PREFIX}SHELL"
MAGITECH_SESSION_ID_KEY = f"{MAGITECH_ENV_PREFIX}SESSION_ID"
MAGITECH_PROFILE_KEY = f"{MAGITECH_ENV_PREFIX}PROFILE"
MAGITECH_PROFILE_DIR_KEY = f"{MAGITECH_ENV_PREFIX}PROFILE_DIR"

RESERVED_SHELL_ENV_KEYS = frozenset(
    {
        MAGITECH_HOME_ENV,
        MAGITECH_AGENTS_HOME_ENV,
        MAGITECH_SHELL_KEY,
        MAGITECH_SESSION_ID_KEY,
        MAGITECH_PROFILE_KEY,
        MAGITECH_PROFILE_DIR_KEY,
    }
)
_ENV_KEY_SUFFIX_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ShellEnvVariable(TypedDict):
    """Model-visible metadata for one managed environment variable."""

    description: str


class ShellEnvVariableInfo(TypedDict):
    """Enumerable declaration returned by :meth:`ShellEnvRegistry.list`."""

    contributor: str
    key: str
    description: str


class ShellEnvContributor(Protocol):
    """Plugin contribution to the managed per-execution shell environment."""

    @property
    def name(self) -> str:
        """Stable contributor name used in diagnostics and conflict errors."""

    @property
    def variables(self) -> Mapping[str, ShellEnvVariable]:
        """Complete map of managed keys this contributor may produce."""

    def resolve(self, execution: Any) -> Mapping[str, str]:
        """Resolve available key values for *execution*."""


@dataclass(slots=True)
class _ContributorRecord:
    name: str
    contributor: ShellEnvContributor
    variables: dict[str, ShellEnvVariable]


def resolve_magitech_home(configured: str | None = None) -> str:
    """Resolve explicit config, ``$MAGITECH_HOME``, or ``~/.magitech``."""
    raw = configured if configured is not None else os.environ.get(MAGITECH_HOME_ENV)
    if raw is not None:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("shell-env: magitechHome must be a non-empty string")
        return str(Path(raw).expanduser().resolve())
    return str((Path.home() / ".magitech").resolve())


def resolve_magitech_agents_home(configured: str | None = None) -> str:
    """Resolve explicit config, ``$MAGITECH_AGENTS_HOME``, or ``~/.agents``."""
    raw = (
        configured
        if configured is not None
        else os.environ.get(MAGITECH_AGENTS_HOME_ENV)
    )
    if raw is not None:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("shell-env: agentsHome must be a non-empty string")
        return str(Path(raw).expanduser().resolve())
    return str((Path.home() / ".agents").resolve())


def _extract_session_id(execution: Any) -> str | None:
    agent = (
        execution.get("agent")
        if isinstance(execution, Mapping)
        else getattr(execution, "agent", None)
    )
    if agent is None:
        return None
    session = getattr(agent, "session", None)
    if session is not None:
        header = getattr(session, "header", None)
        if isinstance(header, Mapping) and isinstance(
            (raw_id := header.get("id")), str
        ):
            return raw_id
        sid = getattr(session, "session_id", None)
        if isinstance(sid, str):
            return sid
    aid = getattr(agent, "id", None)
    return aid if isinstance(aid, str) else None


class ShellEnvRegistry(Service):
    """Service (``ctx.shellEnv``) computing trusted ``MAGITECH_*`` overlays."""

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "shellEnv")
        raw = config or {}
        if "magitechHome" in raw and not isinstance(raw["magitechHome"], str):
            raise ValueError("shell-env: magitechHome must be a string")
        if "agentsHome" in raw and not isinstance(raw["agentsHome"], str):
            raise ValueError("shell-env: agentsHome must be a string")
        self._magitech_home = resolve_magitech_home(raw.get("magitechHome"))
        self._agents_home = resolve_magitech_agents_home(raw.get("agentsHome"))
        self._contributors: dict[str, _ContributorRecord] = {}
        self._key_owners: dict[str, str] = {}

    def register(
        self, contributor: ShellEnvContributor
    ) -> Callable[[], Awaitable[None] | None]:
        """Register one contributor on the calling fiber with conflict validation."""
        caller_ctx = _access_context.get() or self.ctx

        def _setup() -> Callable[[], None]:
            cname = contributor.name
            if not cname.strip():
                raise ValueError("shell env contributor name must be non-empty")
            if cname in self._contributors:
                raise ValueError(
                    f'shell env contributor "{cname}" is already registered'
                )

            variables = dict(contributor.variables)
            for key, var in variables.items():
                if not key.startswith(
                    MAGITECH_ENV_PREFIX
                ) or not _ENV_KEY_SUFFIX_RE.match(key[len(MAGITECH_ENV_PREFIX) :]):
                    raise ValueError(
                        f'shell env contributor "{cname}" declared invalid key "{key}"'
                    )
                if key in RESERVED_SHELL_ENV_KEYS:
                    raise ValueError(
                        f'shell env contributor "{cname}" cannot own reserved key "{key}"'
                    )
                if not var["description"].strip():
                    raise ValueError(
                        f'shell env contributor "{cname}" must describe "{key}"'
                    )
                owner = self._key_owners.get(key)
                if owner is not None:
                    raise ValueError(
                        f'shell env key "{key}" is already owned by contributor '
                        f'"{owner}"; contributor "{cname}" cannot also own it'
                    )

            record = _ContributorRecord(
                name=cname, contributor=contributor, variables=variables
            )
            self._contributors[cname] = record
            for key in variables:
                self._key_owners[key] = cname

            def _teardown() -> None:
                self._contributors.pop(cname, None)
                for key in variables:
                    self._key_owners.pop(key, None)

            return _teardown

        return cast(
            Callable[[], Awaitable[None] | None],
            caller_ctx.effect(_setup, "shellEnv.register()"),
        )

    def collect(self, execution: Any = None) -> Mapping[str, str]:
        """Compute immutable sorted ``MAGITECH_*`` environment map for *execution*."""
        values: dict[str, str] = {
            MAGITECH_HOME_ENV: self._magitech_home,
            MAGITECH_AGENTS_HOME_ENV: self._agents_home,
            MAGITECH_SHELL_KEY: "1",
        }
        session_id = _extract_session_id(execution)
        if session_id is not None:
            values[MAGITECH_SESSION_ID_KEY] = session_id

        profile = self.ctx.get("profileContext", None, strict=False)
        if profile is not None:
            p_name = (
                profile["name"]
                if isinstance(profile, Mapping)
                else getattr(profile, "name", None)
            )
            p_dir = (
                profile["dir"]
                if isinstance(profile, Mapping)
                else getattr(profile, "dir", None)
            )
            if isinstance(p_name, str) and isinstance(p_dir, str):
                values[MAGITECH_PROFILE_KEY] = p_name
                values[MAGITECH_PROFILE_DIR_KEY] = p_dir

        for record in sorted(self._contributors.values(), key=lambda r: r.name):
            resolved = record.contributor.resolve(execution)
            for key, value in resolved.items():
                if key not in record.variables:
                    raise ValueError(
                        f'shell env contributor "{record.name}" returned '
                        f'undeclared key "{key}"'
                    )
                if not isinstance(value, str):
                    raise TypeError(
                        f'shell env contributor "{record.name}" returned a '
                        f'non-string value for "{key}"'
                    )
                values[key] = value

        return MappingProxyType(dict(sorted(values.items())))

    def list(self) -> list[ShellEnvVariableInfo]:
        """List plugin-contributed variables sorted by key."""
        items: list[ShellEnvVariableInfo] = [
            {
                "contributor": record.name,
                "key": key,
                "description": var["description"],
            }
            for record in self._contributors.values()
            for key, var in record.variables.items()
        ]
        items.sort(key=lambda item: item["key"])
        return items


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> ShellEnvRegistry:
    """Install ``ctx.shellEnv`` service onto *ctx*."""
    return ShellEnvRegistry(ctx, config)


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
