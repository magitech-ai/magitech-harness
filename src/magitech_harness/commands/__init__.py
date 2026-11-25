"""Plugin-owned slash-command registry."""

from __future__ import annotations

from magitech_harness.commands.service import (
    CommandsService,
    apply,
    parse_command,
    resolve_command,
)
from magitech_harness.commands.types import CommandDefinition, CommandResult

__all__ = [
    "CommandDefinition",
    "CommandResult",
    "CommandsService",
    "apply",
    "parse_command",
    "resolve_command",
]
