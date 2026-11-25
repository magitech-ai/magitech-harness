"""Tests for the command registry and session feedback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.commands.service import (
    BUILTINS,
    TOKEN_ALIASES,
    CommandsService,
    normalize_result,
    parse_command,
    resolve_command,
)
from magitech_harness.commands.types import (
    CommandDefinition,
    CommandDescriptor,
    CommandError,
    CommandInputDescriptor,
    CommandResult,
)
from magitech_harness.feedback.service import USAGE, record_feedback
from magitech_harness.feedback.types import FeedbackRecord, SessionFeedbackRecordRequest


def _agent(ctx: Any, identifier: str = "cmd-root") -> Any:
    return SimpleNamespace(
        id=identifier, session=ctx.sessions.create(identifier), options={}
    )


# -- parsing / normalization ------------------------------------------------


def test_parse_and_resolve() -> None:
    assert parse_command("/help") == ("help", "")
    assert parse_command("/model  x") == ("model", "  x")
    assert parse_command("/Plan off") is None
    assert parse_command("/goal\u00a0x") is None
    assert parse_command("help") is None
    assert parse_command("/") is None
    assert parse_command("") is None
    assert parse_command(None) is None  # type: ignore[arg-type]

    descriptors = [
        CommandDescriptor("goal", "g", definition_id="@deepseek-ai/dsh-command-goal"),
        CommandDescriptor("plan", "p", definition_id="@deepseek-ai/dsh-plan-mode"),
    ]
    assert resolve_command("goal", descriptors).name == "goal"
    assert resolve_command("目标", descriptors).name == "goal"
    assert resolve_command("计划", descriptors).name == "plan"
    assert resolve_command("unknown", descriptors) is None
    assert resolve_command("compact", descriptors) is None
    assert set(BUILTINS) <= set(TOKEN_ALIASES)


def test_normalize_result() -> None:
    value = CommandResult.success("ok", 3)
    assert normalize_result("x", value) is value
    assert normalize_result("x", {"kind": "success", "text": "ok"}).text == "ok"
    assert normalize_result("x", {"kind": "error", "text": "bad"}).kind == "error"
    with pytest.raises(CommandError):
        normalize_result("x", 1)


# -- registry ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_lifecycle() -> None:
    ctx = Context()
    service = CommandsService(ctx)
    result = CommandResult.success("done")
    definition = CommandDefinition("ping", "Ping", lambda invocation: result)
    events: list[Any] = []
    ctx.on("commands/change", lambda *a: events.append(True), global_=True)

    dispose = service.register(definition)
    assert [descriptor.name for descriptor in service.descriptors()] == ["ping"]
    assert service.descriptors()[0].to_dict()["name"] == "ping"
    assert len(events) == 1
    with pytest.raises(CommandError, match="already registered"):
        service.register(definition)
    with pytest.raises(CommandError, match="must match"):
        service.register(CommandDefinition("Bad", "x", lambda i: result))
    with pytest.raises(CommandError, match="handler must be callable"):
        service.register(CommandDefinition("bad", "x", None))

    session = SimpleNamespace(append=lambda *a, **k: events.append(a))
    agent = SimpleNamespace(session=session, id="a")
    execution = await service.execute(agent, "/ping")
    assert execution is not None
    assert execution.result is result
    assert execution.to_dict()["commandId"] == execution.command_id
    assert [event[0] for event in events if isinstance(event, tuple)] == [
        "command/run",
        "command/done",
    ]

    assert await service.execute(agent, "not a command") is None
    assert await service.execute(agent, "/unknown") is None
    aborted = AbortSignal()
    aborted.abort()
    with pytest.raises(RuntimeError, match="aborted"):
        await service.execute(agent, "/ping", signal=aborted)

    dispose()
    dispose()
    assert service.descriptors() == []


@pytest.mark.asyncio
async def test_handler_results_and_record_input() -> None:
    ctx = Context()
    service = CommandsService(ctx)
    logged: list[Any] = []
    session = SimpleNamespace(
        append=lambda event_type, payload: logged.append((event_type, payload))
    )
    agent = SimpleNamespace(session=session, id="a")

    service.register(
        CommandDefinition(
            "quiet", "q", lambda i: {"kind": "success"}, record_input=False
        )
    )
    service.register(
        CommandDefinition(
            "loud",
            "l",
            lambda i: _async_result(),
            input=CommandInputDescriptor("text", True),
        )
    )
    service.register(CommandDefinition("boom", "b", lambda i: 1 / 0))
    service.register(
        CommandDefinition("map", "m", lambda i: {"kind": "error", "text": "x"})
    )

    async def _async_result() -> dict[str, Any]:
        return {"kind": "success", "text": "async"}

    result = await service.execute(agent, "/quiet args")
    assert result is not None and result.result.kind == "success"
    assert "args" not in logged[0][1]
    assert logged[1][1]["kind"] == "success"

    result = await service.execute(agent, "/loud")
    assert result is not None and result.result.text == "async"
    result = await service.execute(agent, "/map")
    assert result is not None and result.result.kind == "error"

    with pytest.raises(ZeroDivisionError):
        await service.execute(agent, "/boom")
    assert logged[-1][0] == "command/done"
    assert logged[-1][1]["kind"] == "error"


@pytest.mark.asyncio
async def test_done_append_failure_is_contained() -> None:
    ctx = Context()
    service = CommandsService(ctx)
    service.register(
        CommandDefinition("ok", "ok", lambda i: CommandResult.success("x"))
    )
    calls: list[Any] = []

    def append(event_type: str, payload: dict[str, Any]) -> None:
        calls.append(event_type)
        if event_type == "command/done":
            raise RuntimeError("disk full")

    agent = SimpleNamespace(session=SimpleNamespace(append=append), id="a")
    with pytest.raises(RuntimeError, match="disk full"):
        await service.execute(agent, "/ok")


# -- feedback ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_recording(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "feedback")
    await ctx.flush()
    session = ctx.sessions.create("feedback-session")

    record_feedback(session, FeedbackRecord(text="  nice  ", category="task-result"))
    event = session.events[-1]
    assert event.type == "feedback/record"
    assert event.payload == {"text": "nice", "category": "task-result"}

    record_feedback(session, {"text": "  ", "category": None})
    assert session.events[-1].payload == {}

    with pytest.raises(ValueError, match="unknown feedback category"):
        record_feedback(session, FeedbackRecord(text="x", category="bogus"))

    result = ctx.sessionFeedback.record(
        SessionFeedbackRecordRequest("feedback-session", "good", "other")
    )
    assert result.recorded is True
    missing = ctx.sessionFeedback.record(SessionFeedbackRecordRequest("missing"))
    assert missing.recorded is False
    assert missing.error is not None

    agent = _agent(ctx, "feedback-agent")
    execution = await ctx.commands.execute(agent, "/feedback")
    assert execution is not None and execution.result.kind == "error"
    assert USAGE in execution.result.text or ""


def test_standalone_apply() -> None:
    ctx = Context()
    from magitech_harness.commands.service import apply as apply_commands
    from magitech_harness.feedback.service import apply as apply_feedback

    commands = apply_commands(ctx)
    assert isinstance(commands, CommandsService)
    apply_feedback(ctx)
    assert any(
        descriptor.definition_id == "@deepseek-ai/dsh-command-feedback"
        for descriptor in commands.descriptors()
    )


@pytest.mark.asyncio
async def test_registry_remaining_branches() -> None:
    ctx = Context()
    service = CommandsService(ctx)

    def bad_listener() -> None:
        raise RuntimeError("observer failed")

    ctx.on("commands/change", bad_listener, global_=True)
    service.register(CommandDefinition("still", "s", lambda i: CommandResult.success()))
    assert service.descriptors()

    with pytest.raises(CommandError):
        normalize_result("x", {"kind": "bad"})
    assert CommandInputDescriptor("text", True).to_dict()["attachments"] is True

    service.register(
        CommandDefinition(
            "attach",
            "a",
            lambda i: CommandResult.success(),
            input=CommandInputDescriptor("x", True),
        )
    )
    no_session = SimpleNamespace(id="no-session")
    result = await service.execute(no_session, "/attach hello")
    assert result is not None and result.result.kind == "success"

    def append_once(event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "command/run":
            return
        raise RuntimeError("done failed")

    service.register(CommandDefinition("boom2", "b", lambda i: 1 / 0))
    failing = SimpleNamespace(session=SimpleNamespace(append=append_once), id="f")
    with pytest.raises(ZeroDivisionError):
        await service.execute(failing, "/boom2")


def test_feedback_result_shapes() -> None:
    from magitech_harness.feedback.types import SessionFeedbackRecordResult

    assert SessionFeedbackRecordResult(True).to_dict() == {"recorded": True}
    assert SessionFeedbackRecordResult(False, {"code": "x"}).to_dict() == {
        "recorded": False,
        "error": {"code": "x"},
    }


@pytest.mark.asyncio
async def test_feedback_apply_with_existing_service() -> None:
    ctx = Context()
    ctx.provide("sessions", SimpleNamespace(get=lambda session_id: None))
    from magitech_harness.commands.service import apply as apply_commands
    from magitech_harness.feedback.service import SessionFeedbackService
    from magitech_harness.feedback.service import apply as apply_feedback

    commands = apply_commands(ctx)
    existing = SessionFeedbackService(ctx)
    ctx.provide("sessionFeedback", existing)
    apply_feedback(ctx)
    assert any(
        descriptor.definition_id == "@deepseek-ai/dsh-command-feedback"
        for descriptor in commands.descriptors()
    )

    session = SimpleNamespace(session_id="s", append=lambda event_type, payload: None)
    ctx.provide("sessions", SimpleNamespace(get=lambda session_id: session))
    agent = SimpleNamespace(session=session, id="a")
    result = await commands.execute(agent, "/feedback nice work")
    assert result is not None and result.result.kind == "success"


@pytest.mark.asyncio
async def test_handler_error_without_session() -> None:
    ctx = Context()
    service = CommandsService(ctx)
    service.register(CommandDefinition("bad", "b", lambda i: 1 / 0))
    with pytest.raises(ZeroDivisionError):
        await service.execute(SimpleNamespace(id="no-session"), "/bad")


def test_strict_normalization() -> None:
    from magitech_harness.commands.service import abort_error, normalize_definition

    reason = ValueError("already failed")
    assert abort_error(SimpleNamespace(reason=reason, aborted=True)) is reason
    assert str(abort_error(SimpleNamespace(reason="text", aborted=True))) == "text"

    handler = lambda invocation: CommandResult.success()
    for definition in (
        CommandDefinition("a", " ", handler),
        CommandDefinition("b", "ok", handler, input={"hint": "x"}),  # type: ignore[arg-type]
        CommandDefinition("c", "ok", handler, input=CommandInputDescriptor(" ")),
        CommandDefinition("d", "ok", handler, input=CommandInputDescriptor("x", 1)),  # type: ignore[arg-type]
    ):
        with pytest.raises(CommandError):
            normalize_definition(definition)
    assert normalize_definition(CommandDefinition("ok", "ok", handler)) is not None

    for value in (
        {"kind": "success", "text": 1},
        {"kind": "success", "sourceEventSeq": -1},
        {"kind": "error", "text": "  "},
        {"kind": "weird"},
        "not a mapping",
    ):
        with pytest.raises(CommandError):
            normalize_result("x", value)
    assert normalize_result("x", {"kind": "success"}).kind == "success"


@pytest.mark.asyncio
async def test_attachment_and_abort_paths() -> None:
    ctx = Context()
    service = CommandsService(ctx)
    logged: list[Any] = []
    session = SimpleNamespace(
        append=lambda event_type, payload: logged.append((event_type, payload))
    )
    agent = SimpleNamespace(session=session, id="a")

    service.register(CommandDefinition("plain", "p", lambda i: CommandResult.success()))
    result = await service.execute(agent, "/plain", [{"type": "image"}])
    assert result is not None and result.result.kind == "error"
    assert "does not accept attachments" in result.result.text or ""

    service.register(
        CommandDefinition(
            "attach",
            "a",
            lambda i: CommandResult.success(),
            input=CommandInputDescriptor("x", True),
        )
    )
    result = await service.execute(agent, "/attach", [{"type": "image"}])
    assert result is not None and result.result.kind == "error"
    assert "unavailable" in result.result.text or ""

    ctx.provide(
        "attachments",
        SimpleNamespace(admit=lambda values: values),
    )
    result = await service.execute(agent, "/attach", [{"type": "image"}])
    assert result is not None and result.result.kind == "success"

    ctx.provide(
        "attachments",
        SimpleNamespace(admit=lambda values: "not-iterable"),
    )
    result = await service.execute(agent, "/attach", [{"type": "image"}])
    assert result is not None and result.result.kind == "success"

    aborted = AbortSignal()
    aborted.abort()
    with pytest.raises(RuntimeError, match="aborted"):
        await service.execute(agent, "/attach", [{"type": "image"}], signal=aborted)

    # Synchronous handler that observes an abort after registration.
    live = AbortSignal()

    def handler(invocation: Any) -> CommandResult:
        live.abort()
        return CommandResult.success()

    service.register(CommandDefinition("syncabort", "s", handler))
    with pytest.raises(RuntimeError):
        await service.execute(agent, "/syncabort", signal=live)

    # Async handler awaiting with no signal and with an already-aborted signal.
    async def async_handler(invocation: Any) -> dict[str, Any]:
        return {"kind": "success"}

    service.register(CommandDefinition("async", "a", async_handler))
    with pytest.raises(RuntimeError):
        await service.execute(agent, "/async", signal=aborted)


class _LateSignal:
    """Signal that aborts on demand, used to probe admission cancellation."""

    def __init__(self) -> None:
        self._aborted = False
        self.reason: Any = None

    @property
    def aborted(self) -> bool:
        return self._aborted

    def add_listener(self, callback: Any) -> Any:
        return lambda: None

    def abort(self) -> None:
        self._aborted = True


@pytest.mark.asyncio
async def test_remaining_attachment_and_abort_branches() -> None:
    ctx = Context()
    service = CommandsService(ctx)
    session = SimpleNamespace(append=lambda *a, **k: None)
    agent = SimpleNamespace(session=session, id="a")
    service.register(
        CommandDefinition(
            "attach",
            "a",
            lambda i: CommandResult.success(),
            input=CommandInputDescriptor("x", True),
        )
    )
    ctx.provide("attachments", SimpleNamespace())
    result = await service.execute(agent, "/attach", [{"type": "image"}])
    assert result is not None and result.result.kind == "success"

    signal = _LateSignal()

    def admit(_values: Any) -> list[Any]:
        signal.abort()
        return []

    ctx.provide("attachments", SimpleNamespace(admit=admit))
    with pytest.raises(RuntimeError, match="aborted"):
        await service.execute(agent, "/attach", [{"type": "image"}], signal=signal)

    async def async_handler(invocation: Any) -> dict[str, Any]:
        return {"kind": "success"}

    service.register(CommandDefinition("async", "a", async_handler))
    result = await service.execute(agent, "/async", signal=None)
    assert result is not None and result.result.kind == "success"

    aborted = AbortSignal()
    aborted.abort()
    with pytest.raises(RuntimeError):
        await service.execute(agent, "/async", signal=aborted)

    live = AbortSignal()
    result = await service.execute(agent, "/async", signal=live)
    assert result is not None and result.result.kind == "success"


@pytest.mark.asyncio
async def test_with_abort_direct() -> None:
    from magitech_harness.commands.service import CommandsService

    async def value() -> str:
        return "ok"

    assert await CommandsService._with_abort(value(), None) == "ok"
    aborted = AbortSignal()
    aborted.abort()
    with pytest.raises(RuntimeError):
        await CommandsService._with_abort(value(), aborted)
