"""``ctx.commands``: plugin-owned slash-command registry with durable lifecycle.

Ports ``packages/interaction/commands/src/index.ts``. A resolved command appends
``command/run`` before its handler and ``command/done`` after settlement; an
admission miss logs nothing. Observer failures on ``commands/change`` cannot
veto a registry mutation.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from animas import Context, Service

from magitech_harness.commands.types import (
    CommandDefinition,
    CommandDescriptor,
    CommandError,
    CommandExecution,
    CommandInvocation,
    CommandResult,
)

COMMAND_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
BUILTINS = {
    "goal": "@deepseek-ai/dsh-command-goal",
    "plan": "@deepseek-ai/dsh-plan-mode",
    "feedback": "@deepseek-ai/dsh-command-feedback",
    "compact": "@deepseek-ai/dsh-command-compact",
    "permission": "@deepseek-ai/dsh-permission-presets",
    "export": "@deepseek-ai/dsh-session-log-export",
}
TOKEN_ALIASES = {
    "goal": "goal",
    "目标": "goal",
    "plan": "plan",
    "计划": "plan",
    "feedback": "feedback",
    "反馈": "feedback",
    "compact": "compact",
    "压缩": "compact",
    "permission": "permission",
    "权限": "permission",
    "export": "export",
    "导出": "export",
}


def parse_command(line: str) -> tuple[str, str] | None:
    """Split a slash line into its command name and verbatim remainder."""
    if not isinstance(line, str) or not line.startswith("/"):
        return None
    body = line[1:]
    match = re.match(r"([a-z][a-z0-9_-]*)(?=\s|$)", body, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).lower(), body[match.end() :]


def resolve_command(
    token: str, descriptors: list[CommandDescriptor]
) -> CommandDescriptor | None:
    """Resolve an exact name, then a first-party alias by definition id."""
    exact = next((item for item in descriptors if item.name == token), None)
    if exact is not None:
        return exact
    builtin = TOKEN_ALIASES.get(token)
    if builtin is None:
        return None
    return next(
        (item for item in descriptors if item.definition_id == BUILTINS[builtin]), None
    )


def normalize_result(name: str, value: Any) -> CommandResult:
    """Normalize a handler's return value into a CommandResult."""
    if isinstance(value, CommandResult):
        return value
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind in ("success", "error"):
            text = value.get("text")
            return CommandResult(
                kind,
                str(text) if text is not None else None,
                value.get("sourceEventSeq") if kind == "success" else None,
            )
    raise CommandError(
        f'command "{name}" returned an invalid result', "COMMAND_INVALID_RESULT"
    )


class CommandsService(Service):
    """Plugin-owned command registry with log-only lifecycle events."""

    inject: ClassVar[list[str]] = []

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "commands")
        del config
        self._definitions: dict[str, CommandDefinition] = {}
        self._seq = 0

    def _emit_change(self) -> None:
        try:
            self.ctx.emit("commands/change", {})
        except Exception:  # noqa: BLE001, S110 -- observers cannot veto a registry mutation
            pass

    def register(self, definition: CommandDefinition) -> Callable[[], None]:
        """Register one command; reject invalid names and duplicates."""
        if not isinstance(definition.name, str) or not COMMAND_NAME.match(
            definition.name
        ):
            raise CommandError(
                f'invalid command name "{definition.name}"', "COMMAND_INVALID_NAME"
            )
        if definition.name in self._definitions:
            raise CommandError(
                f'command "{definition.name}" is already registered',
                "COMMAND_DUPLICATE",
            )
        if not callable(definition.handler):
            raise CommandError(
                f'command "{definition.name}" needs a callable handler',
                "COMMAND_INVALID_HANDLER",
            )
        self._definitions[definition.name] = definition
        self._emit_change()

        def dispose() -> None:
            if self._definitions.get(definition.name) is definition:
                self._definitions.pop(definition.name)
                self._emit_change()

        return dispose

    def descriptors(self) -> list[CommandDescriptor]:
        """Return handler-free descriptors in registration order."""
        return [
            CommandDescriptor(
                name=definition.name,
                description=definition.description,
                definition_id=definition.definition_id,
                input=definition.input,
            )
            for definition in self._definitions.values()
        ]

    async def execute(
        self,
        agent: Any,
        line: str,
        signal: Any = None,
    ) -> CommandExecution | None:
        """Parse, resolve and execute one command line, logging its lifecycle."""
        parsed = parse_command(line)
        if parsed is None:
            return None
        name, raw_input = parsed
        descriptor = resolve_command(name, self.descriptors())
        if descriptor is None:
            return None
        if signal is not None and signal.aborted:
            raise CommandError("command execution was aborted", "COMMAND_ABORTED")
        definition = self._definitions[descriptor.name]
        self._seq += 1
        command_id = f"cmd-{id(self)}-{self._seq}"
        session = getattr(agent, "session", None)
        if session is not None:
            session.append(
                "command/run",
                {
                    "commandId": command_id,
                    "name": descriptor.name,
                    **({} if not definition.record_input else {"args": raw_input}),
                    "source": {"kind": "user"},
                },
            )

        def settle(result: CommandResult) -> CommandExecution:
            if session is not None:
                session.append(
                    "command/done", {"commandId": command_id, **result.to_dict()}
                )
            return CommandExecution(command_id, result, agent)

        attachments: tuple[Any, ...] = ()
        if raw_input and definition.input is not None and definition.input.attachments:
            # The registry admits attachment-capable definitions; a concrete
            # adapter supplies durable blocks. No attachment store is mounted
            # in this harness, so submissions carry none.
            attachments = ()
        invocation = CommandInvocation(
            command_id, agent, raw_input, signal, attachments
        )
        try:
            output = definition.handler(invocation)
            if inspect.isawaitable(output):
                output = await output
            result = normalize_result(descriptor.name, output)
        except Exception as error:
            if session is not None:
                try:
                    session.append(
                        "command/done",
                        {
                            "commandId": command_id,
                            "kind": "error",
                            "text": str(error),
                        },
                    )
                except Exception:  # noqa: BLE001, S110 -- handler error stays the reported failure
                    pass
            raise
        return settle(result)


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> CommandsService:
    """Mount ``ctx.commands``."""
    return CommandsService(ctx, config)


apply.inject = []  # type: ignore[attr-defined]

__all__ = [
    "BUILTINS",
    "TOKEN_ALIASES",
    "CommandsService",
    "apply",
    "normalize_result",
    "parse_command",
    "resolve_command",
]
