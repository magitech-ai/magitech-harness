"""Durable slash-command vocabulary for interactive command adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from magitech_harness.llm.error import HarnessError

CommandKind = Literal["success", "error"]


class CommandError(HarnessError):
    """Stable failure taxonomy for the command registry."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


@dataclass(frozen=True)
class CommandInputDescriptor:
    """Immutable metadata for a command's optional unstructured input."""

    hint: str
    attachments: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint": self.hint,
            **({"attachments": True} if self.attachments else {}),
        }


@dataclass(frozen=True)
class CommandResult:
    """Expected command outcome rendered directly by the dispatching UI."""

    kind: CommandKind
    text: str | None = None
    source_event_seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            **({} if self.text is None else {"text": self.text}),
            **(
                {}
                if self.kind != "success" or self.source_event_seq is None
                else {"sourceEventSeq": self.source_event_seq}
            ),
        }

    @classmethod
    def success(
        cls, text: str | None = None, source_event_seq: int | None = None
    ) -> CommandResult:
        return cls("success", text, source_event_seq)

    @classmethod
    def error(cls, text: str) -> CommandResult:
        return cls("error", text)


@dataclass(frozen=True)
class CommandInvocation:
    """One resolved command invocation passed to its handler."""

    command_id: str
    agent: Any
    raw_input: str
    signal: Any = None
    attachments: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CommandDefinition:
    """Producer for one registered command."""

    name: str
    description: str
    handler: Any
    definition_id: str | None = None
    input: CommandInputDescriptor | None = None
    record_input: bool = True


@dataclass(frozen=True)
class CommandDescriptor:
    """Handler-free immutable command view returned to UI adapters."""

    name: str
    description: str
    definition_id: str | None = None
    input: CommandInputDescriptor | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            **(
                {}
                if self.definition_id is None
                else {"definitionId": self.definition_id}
            ),
            **({} if self.input is None else {"input": self.input.to_dict()}),
        }


@dataclass(frozen=True)
class CommandExecution:
    """One settled command execution with its lifecycle pairing id."""

    command_id: str
    result: CommandResult
    agent: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"commandId": self.command_id, "result": self.result.to_dict()}


__all__ = [
    "CommandDefinition",
    "CommandDescriptor",
    "CommandError",
    "CommandExecution",
    "CommandInputDescriptor",
    "CommandInvocation",
    "CommandKind",
    "CommandResult",
]
