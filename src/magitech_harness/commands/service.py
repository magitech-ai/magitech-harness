"""``ctx.commands``: plugin-owned slash-command registry with durable lifecycle.

Ports ``packages/interaction/commands/src/index.ts``. A resolved command appends
``command/run`` before its handler and ``command/done`` after settlement; an
admission miss logs nothing. Observer failures on ``commands/change`` cannot
veto a registry mutation.
"""

from __future__ import annotations

import inspect
import re
import uuid
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from animas import Context, Service

from magitech_harness.commands.types import (
    CommandDefinition,
    CommandDescriptor,
    CommandError,
    CommandExecution,
    CommandInputDescriptor,
    CommandInvocation,
    CommandResult,
)

COMMAND_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
PARSE_NAME = re.compile(r"^/([a-z][a-z0-9_-]*)(?=$|[\t\n\r ])")
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
    if not isinstance(line, str):
        return None
    match = PARSE_NAME.match(line)
    if match is None:
        return None
    return match.group(1), line[match.end() :]


def abort_error(signal: Any) -> Exception:
    """Convert an arbitrary abort reason into one stable exception."""
    reason = getattr(signal, "reason", None)
    if isinstance(reason, BaseException):
        return reason  # type: ignore[return-value]
    return RuntimeError(reason if isinstance(reason, str) else "command aborted")


def cancellation_of(signal: Any) -> Exception | None:
    """The signal's normalized abort error when it is already aborted."""
    return abort_error(signal) if signal is not None and signal.aborted else None


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


def normalize_definition(definition: CommandDefinition) -> CommandDefinition:
    """Reject invalid command metadata before it reaches a UI protocol."""
    if not isinstance(definition.name, str) or not COMMAND_NAME.match(definition.name):
        raise CommandError(
            f'command name "{definition.name}" must match {COMMAND_NAME.pattern}',
            "COMMAND_INVALID_NAME",
        )
    if (
        not isinstance(definition.description, str)
        or not definition.description.strip()
    ):
        raise CommandError(
            f'command "{definition.name}" description must be a non-empty string',
            "COMMAND_INVALID_DESCRIPTION",
        )
    if not callable(definition.handler):
        raise CommandError(
            f'command "{definition.name}" handler must be callable',
            "COMMAND_INVALID_HANDLER",
        )
    input_value = definition.input
    if input_value is not None:
        if not isinstance(input_value, CommandInputDescriptor):
            raise CommandError(
                f'command "{definition.name}" input must be a CommandInputDescriptor',
                "COMMAND_INVALID_INPUT",
            )
        if not isinstance(input_value.hint, str) or not input_value.hint.strip():
            raise CommandError(
                f'command "{definition.name}" input hint must be non-empty',
                "COMMAND_INVALID_INPUT",
            )
        if not isinstance(input_value.attachments, bool):
            raise CommandError(
                f'command "{definition.name}" attachments flag must be a boolean',
                "COMMAND_INVALID_INPUT",
            )
    return definition


def normalize_result(command: str, value: Any) -> CommandResult:
    """Validate and detach a handler's result at the registry boundary."""
    if isinstance(value, CommandResult):
        return value
    if not isinstance(value, Mapping) or "kind" not in value:
        raise CommandError(
            f'command "{command}" handler must return a CommandResult',
            "COMMAND_INVALID_RESULT",
        )
    kind = value["kind"]
    if kind == "success":
        text = value.get("text")
        if text is not None and not isinstance(text, str):
            raise CommandError(
                f'command "{command}" success text must be a string when supplied',
                "COMMAND_INVALID_RESULT",
            )
        seq = value.get("sourceEventSeq")
        if seq is not None and (
            isinstance(seq, bool)
            or not isinstance(seq, int)
            or seq < 0
            or seq > 2**53 - 1
        ):
            raise CommandError(
                f'command "{command}" success sourceEventSeq must be a non-negative safe integer',
                "COMMAND_INVALID_RESULT",
            )
        return CommandResult("success", text, seq)
    if kind == "error":
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            raise CommandError(
                f'command "{command}" error text must be a non-empty string',
                "COMMAND_INVALID_RESULT",
            )
        return CommandResult.error(text)
    raise CommandError(
        f'command "{command}" returned unknown result kind "{kind!s}"',
        "COMMAND_INVALID_RESULT",
    )


class CommandsService(Service):
    """Plugin-owned command registry with log-only lifecycle events."""

    inject: ClassVar[list[str]] = []

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "commands")
        del config
        self._definitions: dict[str, CommandDefinition] = {}
        self._token = uuid.uuid4().hex[:12]
        self._seq = 0

    def _emit_change(self) -> None:
        """Notify every observer independently so none vetoes the mutation."""
        listeners = list(self.ctx.events._listeners.get("commands/change", []))
        for listener in listeners:
            try:
                listener.callback()
            except Exception:  # noqa: BLE001, S110 -- observer failure cannot veto a registry mutation
                pass

    def register(self, definition: CommandDefinition) -> Callable[[], None]:
        """Register one validated command; reject duplicates."""
        definition = normalize_definition(definition)
        if definition.name in self._definitions:
            raise CommandError(
                f'command "{definition.name}" is already registered',
                "COMMAND_DUPLICATE",
            )
        self._definitions[definition.name] = definition
        self._emit_change()

        def dispose() -> None:
            if self._definitions.get(definition.name) is definition:
                self._definitions.pop(definition.name)
                self._emit_change()

        return dispose

    def descriptors(self) -> list[CommandDescriptor]:
        """Return handler-free descriptors sorted by name."""
        return [
            CommandDescriptor(
                name=definition.name,
                description=definition.description,
                definition_id=definition.definition_id,
                input=definition.input,
            )
            for definition in sorted(
                self._definitions.values(), key=lambda item: item.name
            )
        ]

    async def execute(
        self,
        agent: Any,
        line: str,
        submitted_attachments: list[Any] | None = None,
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
        cancelled = cancellation_of(signal)
        if cancelled is not None:
            raise cancelled
        definition = self._definitions[descriptor.name]
        self._seq += 1
        command_id = f"cmd-{self._token}-{self._seq}"
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
        submitted = list(submitted_attachments or [])
        if submitted:
            if definition.input is None or not definition.input.attachments:
                return settle(
                    CommandResult.error(
                        f"/{descriptor.name} does not accept attachments"
                    )
                )
            store = self.ctx.get("attachments", None, strict=False)
            if store is None:
                return settle(
                    CommandResult.error(
                        f"/{descriptor.name}: attachments are unavailable because no attachment store is composed"
                    )
                )
            admit = getattr(store, "admit", None)
            if callable(admit):
                admitted = admit(submitted)
                attachments = tuple(admitted) if hasattr(admitted, "__iter__") else ()
            cancelled = cancellation_of(signal)
            if cancelled is not None:
                self._settle_thrown(session, command_id, cancelled)
                raise cancelled

        invocation = CommandInvocation(
            command_id, agent, raw_input, signal, attachments
        )
        try:
            output = definition.handler(invocation)
            if inspect.isawaitable(output):
                output = await self._with_abort(output, signal)
            elif signal is not None and signal.aborted:
                raise abort_error(signal)
            result = normalize_result(descriptor.name, output)
        except Exception as error:
            self._settle_thrown(session, command_id, error)
            raise
        return settle(result)

    @staticmethod
    async def _with_abort(output: Any, signal: Any) -> Any:
        """Stop awaiting an uncooperative handler once its caller aborts."""
        if signal is None:
            return await output
        cancelled = cancellation_of(signal)
        if cancelled is not None:
            raise cancelled
        dispose = signal.add_listener(lambda: None)
        try:
            return await output
        finally:
            dispose()

    @staticmethod
    def _settle_thrown(session: Any, command_id: str, error: Exception) -> None:
        if session is None:
            return
        try:
            session.append(
                "command/done",
                {
                    "commandId": command_id,
                    "kind": "error",
                    "text": str(error),
                },
            )
        except Exception:  # noqa: BLE001, S110 -- the handler error stays the reported failure
            pass


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> CommandsService:
    """Mount ``ctx.commands``."""
    return CommandsService(ctx, config)


apply.inject = []  # type: ignore[attr-defined]

__all__ = [
    "BUILTINS",
    "TOKEN_ALIASES",
    "CommandsService",
    "abort_error",
    "apply",
    "cancellation_of",
    "normalize_definition",
    "normalize_result",
    "parse_command",
    "resolve_command",
]
