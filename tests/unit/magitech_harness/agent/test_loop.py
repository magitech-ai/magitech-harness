"""Unit tests for AgentLoopService."""

import pytest
from anima import Context, EventMode

from magitech_harness.agent.loop import AgentLoopService, _RequestState
from magitech_harness.llm.service import LLMService, MockLLMProvider, NoProviderError
from magitech_harness.llm.types import ModelChunk, ModelResponse, UsageData
from magitech_harness.session.events import EventType, SessionEvent, ToolCallData
from magitech_harness.session.service import SessionService


@pytest.mark.asyncio
async def test_agent_loop_single_step_no_tools() -> None:
    """Verifies a 1-step turn when the LLM returns plain text with no tool calls."""
    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    mock_llm = MockLLMProvider(default_model="mock-model")
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="Direct answer")

    session = ctx.sessions.create("sess_1")
    reply = await ctx.agent_loop.run_turn(session, "Hello agent")

    assert reply == "Direct answer"
    types = [e.type for e in session.events]
    assert types == [
        EventType.TURN_START,
        EventType.STEP_START,
        EventType.SYSTEM_MESSAGE,
        EventType.USER_MESSAGE,
        EventType.REQUEST_HEADER,
        EventType.REQUEST_CONTEXT,
        EventType.ASSISTANT_MESSAGE,
        EventType.STEP_END,
        EventType.TURN_END,
    ]
    by_type = {e.type: e for e in session.events}
    assert by_type[EventType.TURN_START].payload == {"turn": 1}
    system = by_type[EventType.SYSTEM_MESSAGE].payload
    assert system["turn"] == 1 and system["step"] == 1
    # No prompt service registered: the empty prompt still reserves node 0.
    assert system["message"]["role"] == "system"
    assert system["message"]["content"] == []
    assert system["message"]["source"] == {"kind": "system-prompt"}
    user = by_type[EventType.USER_MESSAGE].payload
    assert user["role"] == "user"
    assert user["content"] == [{"type": "text", "text": "Hello agent"}]
    assert user["source"] == {"kind": "user"}
    header = by_type[EventType.REQUEST_HEADER].payload
    # The header carries the provider's declared default, not the
    # registration name: the default is deliberately distinct here.
    assert header == {
        "header": {"config": {"provider": "mock", "model": "mock-model"}},
        "reason": "initial",
    }
    assert by_type[EventType.REQUEST_CONTEXT].payload == {
        "provider": "mock",
        "model": "mock-model",
    }
    assistant = by_type[EventType.ASSISTANT_MESSAGE].payload
    assert assistant["turn"] == 1 and assistant["step"] == 1
    assert assistant["message"]["role"] == "assistant"
    assert assistant["message"]["content"] == [
        {"type": "text", "text": "Direct answer"}
    ]
    assert assistant["message"]["source"] == {
        "kind": "model",
        "provider": "mock",
        "model": "mock-model",
    }
    # Non-streaming settlement: exactly one chunk record, no usage field
    # when the adapter reported no accounting.
    assert len(assistant["stream"]) == 1
    record = assistant["stream"][0]
    assert record["type"] == "chunk"
    assert isinstance(record["time"], int)
    assert record["chunk"]["text"] == "Direct answer"
    assert record["chunk"]["tool_calls"] == []
    assert record["chunk"]["finish_reason"] == "stop"
    assert "usage" not in assistant
    assert "tool_calls" not in assistant
    assert by_type[EventType.STEP_END].payload == {"turn": 1, "step": 1}
    assert by_type[EventType.TURN_END].payload == {
        "turn": 1,
        "reason": {"kind": "completed"},
    }
    # The model call receives the derived history (block-structured) and
    # no separate system prompt; the resolved default is dispatched as
    # a provider/model slug.
    assert mock_llm.call_history[0]["system_prompt"] is None
    assert mock_llm.call_history[0]["kwargs"]["model"] == "mock/mock-model"
    # At request time the derived history holds only the user message: the
    # empty system node projects to nothing and the assistant reply is
    # not yet logged.
    assert mock_llm.call_history[0]["messages"] == [user]
    # The empty system node projects to nothing: the derived history is
    # exactly the user and assistant messages, verbatim.
    derived = session.derive_messages()
    assert [message["role"] for message in derived] == ["user", "assistant"]
    assert derived[0] == user
    assert derived[1] == assistant["message"]


@pytest.mark.asyncio
async def test_agent_loop_pre_step_reject_short_circuit() -> None:
    """Verifies that pre-step waterfall veto immediately stops the turn."""
    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    async def guard(decision: dict, next_fn):
        return {"action": "reject"}

    ctx.on("agent/pre-step", guard, mode=EventMode.WATERFALL)

    session = ctx.sessions.create("sess_veto")
    reply = await ctx.agent_loop.run_turn(session, "Prompt")

    assert reply == ""
    types = [e.type for e in session.events]
    # The user message is logged inside the step attempt, after the
    # pre-step waterfall, so a veto logs no user message; the entered
    # step still ends exactly once.
    assert types == [
        EventType.TURN_START,
        EventType.STEP_START,
        EventType.STEP_END,
        EventType.TURN_END,
    ]
    turn_end = next(e for e in session.events if e.type == EventType.TURN_END)
    assert turn_end.payload == {"turn": 1, "reason": {"kind": "completed"}}


@pytest.mark.asyncio
async def test_agent_loop_max_steps_termination() -> None:
    """Verifies that reaching max_steps terminates the loop even if tool calls continue."""
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(ToolService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    ctx.tools.register(
        define_tool(name="ping", description="ping", execute=lambda args: "pong")
    )

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    # Queue infinite tool calls
    for i in range(10):
        mock_llm.queue_response(
            text=f"step_{i}",
            tool_calls=[ToolCallData(call_id=f"c_{i}", name="ping", arguments={})],
        )

    session = ctx.sessions.create("sess_max")
    reply = await ctx.agent_loop.run_turn(session, "Run infinite", max_steps=2)

    assert reply == "step_1"
    step_starts = [e for e in session.events if e.type == EventType.STEP_START]
    assert len(step_starts) == 2


@pytest.mark.asyncio
async def test_agent_loop_stream_turn_single_step() -> None:
    """Verifies stream_turn yielding token chunks for a single step turn."""
    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="Streaming direct answer")

    session = ctx.sessions.create("sess_stream_1")
    yielded_events = []

    async for ev in ctx.agent_loop.stream_turn(session, "Prompt"):
        yielded_events.append(ev)

    assert len(yielded_events) == 2
    assert yielded_events[0] == {"type": "chunk", "text": "Streaming direct answer"}
    assert yielded_events[1] == {"type": "turn_end", "text": "Streaming direct answer"}

    types = [e.type for e in session.events]
    assert EventType.TURN_START in types
    assert EventType.USER_MESSAGE in types
    assert EventType.ASSISTANT_MESSAGE in types
    assert EventType.TURN_END in types


@pytest.mark.asyncio
async def test_agent_loop_stream_turn_with_usage() -> None:
    """Verifies stream_turn yields usage event and persists usage telemetry."""
    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(
        text="With usage",
        usage=UsageData(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    session = ctx.sessions.create("sess_usage")
    yielded = []
    async for ev in ctx.agent_loop.stream_turn(session, "Prompt with usage"):
        yielded.append(ev)

    types = [y["type"] for y in yielded]
    assert "chunk" in types
    assert "usage" in types
    assert "turn_end" in types

    asst_msg = next(e for e in session.events if e.type == EventType.ASSISTANT_MESSAGE)
    assert asst_msg.payload["usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_agent_loop_stream_turn_with_tool_calls() -> None:
    """Verifies stream_turn executing tools and yielding tool_start / tool_end events."""
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(ToolService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    ctx.tools.register(
        define_tool(
            name="greet",
            description="Greets a person",
            execute=lambda args: f"Hello {args.get('name', 'friend')}",
        )
    )

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    # Step 1: Call tool
    mock_llm.queue_response(
        text="",
        tool_calls=[
            ToolCallData(
                call_id="call_greet_1", name="greet", arguments={"name": "Alice"}
            )
        ],
    )
    # Step 2: Final text
    mock_llm.queue_response(text="Greeted Alice successfully.")

    session = ctx.sessions.create("sess_stream_tool")
    yielded = []

    async for ev in ctx.agent_loop.stream_turn(session, "Greet Alice"):
        yielded.append(ev)

    types = [ev["type"] for ev in yielded]
    assert "tool_start" in types
    assert "tool_end" in types
    assert "chunk" in types
    assert "turn_end" in types

    tool_start_ev = next(ev for ev in yielded if ev["type"] == "tool_start")
    assert tool_start_ev["name"] == "greet"
    assert tool_start_ev["arguments"] == {"name": "Alice"}

    tool_end_ev = next(ev for ev in yielded if ev["type"] == "tool_end")
    assert tool_end_ev["name"] == "greet"
    assert tool_end_ev["result"] == "Hello Alice"
    assert not tool_end_ev["is_error"]


@pytest.mark.asyncio
async def test_agent_loop_abort_signal_cancellation() -> None:
    """Verifies that AbortSignal aborts both run_turn and stream_turn."""
    from magitech_harness.agent.cancellation import AbortError, AbortSignal

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="response")

    session = ctx.sessions.create("sess_abort")
    signal = AbortSignal()
    signal.abort()

    with pytest.raises(AbortError, match="Operation was aborted"):
        await ctx.agent_loop.run_turn(session, "hi", signal=signal)

    with pytest.raises(AbortError, match="Operation was aborted"):
        async for _ in ctx.agent_loop.stream_turn(session, "hi", signal=signal):
            pass


@pytest.mark.asyncio
async def test_agent_loop_stream_turn_pre_step_reject() -> None:
    """Verifies that pre-step waterfall veto in stream_turn stops execution."""
    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    async def guard(decision: dict, next_fn):
        return {"action": "reject"}

    ctx.on("agent/pre-step", guard, mode=EventMode.WATERFALL)

    session = ctx.sessions.create("sess_stream_veto")
    yielded = []
    async for ev in ctx.agent_loop.stream_turn(session, "Prompt"):
        yielded.append(ev)

    assert len(yielded) == 1
    assert yielded[0] == {"type": "turn_end", "text": ""}


@pytest.mark.asyncio
async def test_agent_loop_stream_turn_max_steps_termination() -> None:
    """Verifies that reaching max_steps terminates stream_turn."""
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(ToolService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    ctx.tools.register(
        define_tool(name="ping", description="ping", execute=lambda args: "pong")
    )

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    for i in range(5):
        mock_llm.queue_response(
            text=f"step_{i}",
            tool_calls=[ToolCallData(call_id=f"c_{i}", name="ping", arguments={})],
        )

    session = ctx.sessions.create("sess_stream_max")
    yielded = []
    async for ev in ctx.agent_loop.stream_turn(session, "Infinite", max_steps=2):
        yielded.append(ev)

    step_starts = [e for e in session.events if e.type == EventType.STEP_START]
    assert len(step_starts) == 2


@pytest.mark.asyncio
async def test_agent_loop_signal_active_during_tool_execution() -> None:
    """Verifies that an unaborted AbortSignal passes through step and tool checks."""
    from magitech_harness.agent.cancellation import AbortSignal
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(ToolService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    ctx.tools.register(
        define_tool(name="echo", description="echo", execute=lambda args: "echoed")
    )

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    tc = ToolCallData(call_id="call_echo", name="echo", arguments={})
    mock_llm.queue_response(text="step1", tool_calls=[tc])
    mock_llm.queue_response(text="done")

    session = ctx.sessions.create("sess_signal_ok")
    signal = AbortSignal()  # not aborted
    res = await ctx.agent_loop.run_turn(session, "echo please", signal=signal)
    assert res == "done"

    # Also test stream_turn with active unaborted signal
    mock_llm.queue_response(text="step1", tool_calls=[tc])
    mock_llm.queue_response(text="stream_done")
    session2 = ctx.sessions.create("sess_signal_stream_ok")
    stream_events = []
    async for ev in ctx.agent_loop.stream_turn(session2, "stream echo", signal=signal):
        stream_events.append(ev)
    assert any(ev["type"] == "tool_start" for ev in stream_events)


# ---------------------------------------------------------------------------
# P2-6 request reconstruction: loop tests
# ---------------------------------------------------------------------------


class _RecordingProvider:
    """Provider recording the dispatched model; declares a default_model
    (or a legacy ``model`` attribute) and no other model source."""

    def __init__(
        self,
        default_model: str | None = None,
        *,
        model: str | None = None,
    ) -> None:
        if default_model is not None:
            self.default_model = default_model
        if model is not None:
            self.model = model
        self.seen_models: list = []

    async def generate(self, messages, tools=None, system_prompt=None, **kwargs):  # type: ignore[no-untyped-def]
        self.seen_models.append(kwargs.get("model"))
        return ModelResponse(
            text="ok",
            tool_calls=[],
            finish_reason="stop",
            usage=UsageData(),
        )

    async def stream(  # type: ignore[no-untyped-def]
        self,
        messages,
        tools=None,
        system_prompt=None,
        model=None,
        signal=None,
        **kwargs,
    ):
        self.seen_models.append(model)
        yield ModelChunk(
            text="ok",
            tool_calls=[],
            finish_reason="stop",
            usage=UsageData(),
        )


class _FailingStreamProvider:
    """Provider whose stream yields chunks then raises."""

    def __init__(self, chunks: list, error: BaseException) -> None:
        self._chunks = list(chunks)
        self._error = error

    async def generate(self, messages, tools=None, system_prompt=None, **kwargs):  # type: ignore[no-untyped-def]
        raise self._error

    async def stream(  # type: ignore[no-untyped-def]
        self,
        messages,
        tools=None,
        system_prompt=None,
        model=None,
        signal=None,
        **kwargs,
    ):
        for chunk in self._chunks:
            yield chunk
        raise self._error


class _FakeTitleProvider:
    """Scriptable first-prompt title provider."""

    def __init__(self) -> None:
        self.id = "fake-title-provider"
        self.automatic = "first-prompt"
        self.requests: list = []

    async def generate(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return {
            "title": "Provider Title",
            "messageSeqs": [message["seq"] for message in request.messages],
        }


async def _ctx_with_mock() -> tuple[Context, MockLLMProvider]:
    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()
    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    return ctx, mock_llm


@pytest.mark.asyncio
async def test_agent_loop_second_turn_resume_header() -> None:
    """Verifies the second turn numbers from the log and resumes its header."""
    ctx, mock_llm = await _ctx_with_mock()
    # Distinct from the registration name, so the route below proves
    # the header carries the provider default.
    mock_llm.default_model = "mock-model"
    mock_llm.queue_response(text="one")
    mock_llm.queue_response(text="two")
    session = ctx.sessions.create("sess_resume")
    await ctx.agent_loop.run_turn(session, "first")
    await ctx.agent_loop.run_turn(session, "second")

    turns = [
        e.payload["turn"] for e in session.events if e.type == EventType.TURN_START
    ]
    assert turns == [1, 2]
    headers = [e for e in session.events if e.type == EventType.REQUEST_HEADER]
    assert [h.payload["reason"] for h in headers] == ["initial", "resume"]
    # The route never changed, so exactly one context event exists.
    contexts = [e for e in session.events if e.type == EventType.REQUEST_CONTEXT]
    assert len(contexts) == 1
    assert contexts[0].payload == {"provider": "mock", "model": "mock-model"}
    derived = session.derive_messages()
    assert [message["role"] for message in derived] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["content"][0]["text"] for message in derived] == [
        "first",
        "one",
        "second",
        "two",
    ]


@pytest.mark.asyncio
async def test_agent_loop_tool_call_canonical_log() -> None:
    """Verifies tool calls log in wire shape and dispatch reads the log."""
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(ToolService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    seen: list[dict] = []
    ctx.tools.register(
        define_tool(
            name="greet",
            description="Greets a person",
            execute=lambda args: seen.append(dict(args)) or "Hello Alice",
        )
    )
    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(
        text="",
        tool_calls=[
            ToolCallData(
                call_id="call_greet_1", name="greet", arguments={"name": "Alice"}
            )
        ],
    )
    mock_llm.queue_response(text="Greeted Alice successfully.")

    session = ctx.sessions.create("sess_tool_log")
    reply = await ctx.agent_loop.run_turn(session, "Greet Alice")
    assert reply == "Greeted Alice successfully."
    assert seen == [{"name": "Alice"}]

    tool_call = next(e for e in session.events if e.type == EventType.TOOL_CALL)
    assert tool_call.payload == {
        "turn": 1,
        "step": 1,
        "callId": "call_greet_1",
        "name": "greet",
        "arguments": '{"name":"Alice"}',
    }
    tool_result = next(e for e in session.events if e.type == EventType.TOOL_RESULT)
    assert tool_result.payload["turn"] == 1 and tool_result.payload["step"] == 1
    message = tool_result.payload["message"]
    assert message["role"] == "tool"
    assert message["toolCallId"] == "call_greet_1"
    assert message["content"] == [{"type": "text", "text": "Hello Alice"}]
    assert len(message["id"]) == 32  # factory-minted hex, not tool-result-<callId>
    assert not message["id"].startswith("tool-result-")
    assistant = next(e for e in session.events if e.type == EventType.ASSISTANT_MESSAGE)
    assert assistant.payload["message"]["content"] == [
        {
            "type": "tool-call",
            "id": "call_greet_1",
            "name": "greet",
            "arguments": '{"name":"Alice"}',
        }
    ]
    derived = session.derive_messages()
    assert [message["role"] for message in derived] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert derived[2]["toolCallId"] == "call_greet_1"


@pytest.mark.asyncio
async def test_agent_loop_stream_failed_attempt_preserved() -> None:
    """Verifies a failed streaming attempt logs assistant/attempt only."""
    ctx, _ = await _ctx_with_mock()
    failing = _FailingStreamProvider([ModelChunk(text="partial")], RuntimeError("boom"))
    ctx.llm.register_provider("mock", failing)
    session = ctx.sessions.create("sess_attempt")

    with pytest.raises(RuntimeError, match="boom"):
        # Explicit route: the failing provider declares no default.
        async for _ in ctx.agent_loop.stream_turn(
            session, "Prompt", model="mock/failing"
        ):
            pass

    attempt = next(e for e in session.events if e.type == EventType.ASSISTANT_ATTEMPT)
    assert attempt.payload["turn"] == 1 and attempt.payload["step"] == 1
    assert len(attempt.payload["stream"]) == 1
    assert attempt.payload["stream"][0]["chunk"]["text"] == "partial"
    assert "surface_op" not in attempt.payload
    assert not any(e.type == EventType.ASSISTANT_MESSAGE for e in session.events)
    # The attempt is preserved verbatim in the log and invisible in history.
    assert [message["role"] for message in session.derive_messages()] == ["user"]
    turn_end = next(e for e in session.events if e.type == EventType.TURN_END)
    assert turn_end.payload["reason"] == {
        "kind": "error",
        "error": {"message": "boom", "code": "RuntimeError"},
    }


@pytest.mark.asyncio
async def test_agent_loop_stream_cancelled_prefix_finalized() -> None:
    """Verifies a cancelled stream with text finalizes an interrupted message."""
    from magitech_harness.agent.cancellation import AbortError

    ctx, _ = await _ctx_with_mock()
    failing = _FailingStreamProvider(
        [ModelChunk(text="prefix text")], AbortError("cancelled by test")
    )
    ctx.llm.register_provider("mock", failing)
    session = ctx.sessions.create("sess_cancel_prefix")

    with pytest.raises(AbortError, match="cancelled by test"):
        # Explicit route: the failing provider declares no default.
        async for _ in ctx.agent_loop.stream_turn(
            session, "Prompt", model="mock/failing"
        ):
            pass

    assistant = next(e for e in session.events if e.type == EventType.ASSISTANT_MESSAGE)
    assert assistant.payload["interrupted"] is True
    assert assistant.payload["message"]["content"] == [
        {"type": "text", "text": "prefix text"}
    ]
    assert not any(e.type == EventType.ASSISTANT_ATTEMPT for e in session.events)
    derived = session.derive_messages()
    assert [message["role"] for message in derived] == ["user", "assistant"]
    assert derived[1]["content"] == [{"type": "text", "text": "prefix text"}]
    turn_end = next(e for e in session.events if e.type == EventType.TURN_END)
    assert turn_end.payload["reason"]["kind"] == "aborted"


@pytest.mark.asyncio
async def test_agent_loop_stream_cancelled_empty_prefix_logs_attempt() -> None:
    """Verifies a cancelled stream with no delivered text logs an attempt."""
    from magitech_harness.agent.cancellation import AbortError

    ctx, _ = await _ctx_with_mock()
    failing = _FailingStreamProvider(
        [
            ModelChunk(
                text="",
                tool_calls=[ToolCallData(call_id="c1", name="ping", arguments={})],
            )
        ],
        AbortError("cancelled by test"),
    )
    ctx.llm.register_provider("mock", failing)
    session = ctx.sessions.create("sess_cancel_empty")

    with pytest.raises(AbortError, match="cancelled by test"):
        # Explicit route: the failing provider declares no default.
        async for _ in ctx.agent_loop.stream_turn(
            session, "Prompt", model="mock/failing"
        ):
            pass

    assert any(e.type == EventType.ASSISTANT_ATTEMPT for e in session.events)
    assert not any(e.type == EventType.ASSISTANT_MESSAGE for e in session.events)
    assert [message["role"] for message in session.derive_messages()] == ["user"]


@pytest.mark.asyncio
async def test_agent_loop_stream_empty_failure_logs_nothing() -> None:
    """Verifies a stream failing before any chunk logs no settlement."""
    ctx, _ = await _ctx_with_mock()
    failing = _FailingStreamProvider([], RuntimeError("no chunks"))
    ctx.llm.register_provider("mock", failing)
    session = ctx.sessions.create("sess_empty_fail")

    with pytest.raises(RuntimeError, match="no chunks"):
        # Explicit route: the failing provider declares no default.
        async for _ in ctx.agent_loop.stream_turn(
            session, "Prompt", model="mock/failing"
        ):
            pass

    assert not any(
        e.type in (EventType.ASSISTANT_MESSAGE, EventType.ASSISTANT_ATTEMPT)
        for e in session.events
    )
    assert [message["role"] for message in session.derive_messages()] == ["user"]


@pytest.mark.asyncio
async def test_agent_loop_empty_content_with_usage_logs_message() -> None:
    """Verifies an empty-content step with usage logs a message (max-tokens)."""
    from magitech_harness.llm.types import ModelResponse

    ctx, mock_llm = await _ctx_with_mock()
    mock_llm.responses.append(
        ModelResponse(
            text="",
            usage=UsageData(prompt_tokens=5, completion_tokens=0, total_tokens=5),
            finish_reason="max_tokens",
        )
    )
    session = ctx.sessions.create("sess_max_tokens")
    reply = await ctx.agent_loop.run_turn(session, "Prompt")
    assert reply == ""

    assistant = next(e for e in session.events if e.type == EventType.ASSISTANT_MESSAGE)
    assert assistant.payload["message"]["content"] == []
    assert assistant.payload["usage"]["total_tokens"] == 5
    # An empty-content message hosts only usage: it projects to nothing.
    assert [message["role"] for message in session.derive_messages()] == ["user"]
    turn_end = next(e for e in session.events if e.type == EventType.TURN_END)
    assert turn_end.payload["reason"] == {"kind": "max-tokens"}


@pytest.mark.asyncio
async def test_agent_loop_empty_content_without_usage_logs_nothing() -> None:
    """Verifies an empty-content step with no usage logs no assistant event."""
    ctx, mock_llm = await _ctx_with_mock()
    mock_llm.queue_response(text="")
    session = ctx.sessions.create("sess_empty")
    reply = await ctx.agent_loop.run_turn(session, "Prompt")
    assert reply == ""

    assert not any(e.type == EventType.ASSISTANT_MESSAGE for e in session.events)
    assert [e.type for e in session.events].count(EventType.STEP_END) == 1
    turn_end = next(e for e in session.events if e.type == EventType.TURN_END)
    assert turn_end.payload["reason"] == {"kind": "completed"}


@pytest.mark.asyncio
async def test_agent_loop_developer_tool_addition_and_removal() -> None:
    """Verifies tool-set changes emit developer messages with headerSeq rules."""
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(ToolService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    dispose_ping = ctx.tools.register(
        define_tool(name="ping", description="ping", execute=lambda args: "pong")
    )
    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="one")
    mock_llm.queue_response(text="two")
    mock_llm.queue_response(text="three")

    session = ctx.sessions.create("sess_dev_tools")
    await ctx.agent_loop.run_turn(session, "first")
    assert not any(e.type == EventType.DEVELOPER_MESSAGE for e in session.events)

    ctx.tools.register(
        define_tool(name="greet", description="greet", execute=lambda args: "hi")
    )
    await ctx.agent_loop.run_turn(session, "second")
    developers = [e for e in session.events if e.type == EventType.DEVELOPER_MESSAGE]
    assert len(developers) == 1
    addition = developers[0]
    assert addition.payload["message"]["content"] == [
        {"type": "tool-addition", "toolName": "greet"}
    ]
    assert addition.payload["message"]["source"] == {"kind": "tool-registry"}
    header_seq = addition.payload["headerSeq"]
    header = session.events[header_seq]
    assert header.type == EventType.REQUEST_HEADER
    assert header.payload["reason"] == "resume"
    # A tools-only change does not start a series for the header: the
    # tools term feeds the system projection input only.
    assert "startsSeries" not in header.payload

    dispose_ping()
    await ctx.agent_loop.run_turn(session, "third")
    developers = [e for e in session.events if e.type == EventType.DEVELOPER_MESSAGE]
    assert len(developers) == 2
    removal = developers[1]
    assert removal.payload["message"]["content"] == [
        {"type": "tool-removal", "toolName": "ping"}
    ]
    # Removals only: headerSeq is absent.
    assert "headerSeq" not in removal.payload
    history = session.tool_history()
    # No re-baseline: the removal leaves the baseline series intact and
    # is carried as an empty-additions update.
    assert [tool["function"]["name"] for tool in history["tools"]] == ["ping"]
    assert [
        [tool["function"]["name"] for tool in update["additions"]]
        for update in history["updates"]
    ] == [["greet"], []]


@pytest.mark.asyncio
async def test_agent_loop_tools_only_change_keeps_tool_baseline() -> None:
    """Verifies a tools-only change marks no header series and keeps the baseline."""
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(ToolService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    ctx.tools.register(
        define_tool(name="a_tool", description="a", execute=lambda args: "a")
    )
    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="one")
    mock_llm.queue_response(text="two")

    session = ctx.sessions.create("sess_tools_series")
    await ctx.agent_loop.run_turn(session, "first")
    baseline_schemas = ctx.tools.get_schemas()
    ctx.tools.register(
        define_tool(name="b_tool", description="b", execute=lambda args: "b")
    )
    added_schemas = ctx.tools.get_schemas()
    await ctx.agent_loop.run_turn(session, "second")

    headers = [e for e in session.events if e.type == EventType.REQUEST_HEADER]
    assert [h.payload["reason"] for h in headers] == ["initial", "resume"]
    # Tools alone do not start a header series: the tools term feeds
    # the system projection input only.
    assert "startsSeries" not in headers[1].payload
    # The baseline is not re-based: the addition is recorded as an
    # update against the original baseline.
    history = session.tool_history()
    assert history["tools"] == baseline_schemas
    assert len(history["updates"]) == 1
    assert history["updates"][0]["additions"] == [added_schemas[1]]
    assert "messageId" in history["updates"][0]


@pytest.mark.asyncio
async def test_agent_loop_system_prompt_normalization_across_turns() -> None:
    """Verifies a changed rendering rewrites the system head by replace."""
    from magitech_harness.system_prompt.service import SystemPromptService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(SystemPromptService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    ctx.system_prompt.register_section("main", "Prompt A")
    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="one")
    mock_llm.queue_response(text="two")

    session = ctx.sessions.create("sess_sys_norm")
    await ctx.agent_loop.run_turn(session, "first")
    system_nodes = [e for e in session.events if e.type == EventType.SYSTEM_MESSAGE]
    assert len(system_nodes) == 1
    assert system_nodes[0].payload["message"]["content"] == [
        {"type": "text", "text": "Prompt A"}
    ]
    head_seq = system_nodes[0].seq

    ctx.system_prompt.register_section("main", "Prompt B")
    await ctx.agent_loop.run_turn(session, "second")
    system_nodes = [e for e in session.events if e.type == EventType.SYSTEM_MESSAGE]
    assert len(system_nodes) == 2
    rewrite = system_nodes[1]
    assert rewrite.surface_op == {
        "op": "replace",
        "startSeq": head_seq,
        "endSeq": head_seq,
    }
    assert rewrite.source_event_seqs == [head_seq]
    assert rewrite.payload["message"]["content"] == [
        {"type": "text", "text": "Prompt B"}
    ]
    derived = session.derive_messages()
    assert derived[0]["role"] == "system"
    assert derived[0]["content"] == [{"type": "text", "text": "Prompt B"}]
    assert sum(1 for m in derived if m["role"] == "system") == 1


@pytest.mark.asyncio
async def test_agent_loop_resumed_turn_system_change_marks_series() -> None:
    """Verifies a resumed turn whose system prompt changed marks a series."""
    from magitech_harness.system_prompt.service import SystemPromptService

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(SystemPromptService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    ctx.system_prompt.register_section("main", "Prompt A.")
    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="one")
    mock_llm.queue_response(text="two")

    session = ctx.sessions.create("sess_series_system")
    await ctx.agent_loop.run_turn(session, "first")
    ctx.system_prompt.register_section("main", "Prompt B.")
    await ctx.agent_loop.run_turn(session, "second")

    headers = [e for e in session.events if e.type == EventType.REQUEST_HEADER]
    assert [h.payload["reason"] for h in headers] == ["initial", "resume"]
    # The second turn's first header carries startsSeries: the system
    # normalization replace in that step bumped the surface generation
    # after the projection input was computed, and the header decision
    # re-reads it.
    assert headers[1].payload["startsSeries"] is True
    system_nodes = [e for e in session.events if e.type == EventType.SYSTEM_MESSAGE]
    assert len(system_nodes) == 2
    assert system_nodes[1].payload["message"]["content"] == [
        {"type": "text", "text": "Prompt B."}
    ]


@pytest.mark.asyncio
async def test_agent_loop_title_fires_through_run_turn() -> None:
    """Verifies a first-prompt title provider commits through run_turn."""
    ctx, mock_llm = await _ctx_with_mock()
    mock_llm.queue_response(text="Direct answer")
    provider = _FakeTitleProvider()
    ctx.sessions.title_service.register(provider)

    session = ctx.sessions.create("sess_title_turn")
    await ctx.agent_loop.run_turn(session, "Explain quantum computing simply")
    await ctx.sessions.title_service.drain()

    assert len(provider.requests) == 1
    title = ctx.sessions.title_service.get(session)
    assert title is not None
    assert title["title"] == "Provider Title"
    assert title["source"]["kind"] == "provider"
    titles = [e for e in session.events if e.type == EventType.SESSION_TITLE]
    assert titles[-1].payload["title"] == "Provider Title"


def test_loop_tool_call_message_helpers() -> None:
    """Verifies the loop's message-parsing helpers, branch by branch."""
    from magitech_harness.agent.loop import (
        _system_node_text,
        _tool_calls_from_message,
        _tool_name_or_none,
    )

    message = {
        "content": [
            {
                "type": "tool-call",
                "id": "c1",
                "name": "add",
                "arguments": '{"a":1}',
            }
        ]
    }
    calls = _tool_calls_from_message(message)
    assert [(c.call_id, c.name, c.arguments) for c in calls] == [
        ("c1", "add", {"a": 1})
    ]
    assert _tool_calls_from_message({"content": [{"type": "text", "text": "x"}]}) == []
    with pytest.raises(ValueError, match="unparseable arguments"):
        _tool_calls_from_message(
            {
                "content": [
                    {
                        "type": "tool-call",
                        "id": "c2",
                        "name": "add",
                        "arguments": "{bad",
                    }
                ]
            }
        )
    with pytest.raises(TypeError, match="arguments must be an object"):
        _tool_calls_from_message(
            {
                "content": [
                    {
                        "type": "tool-call",
                        "id": "c3",
                        "name": "add",
                        "arguments": "[1,2]",
                    }
                ]
            }
        )

    def _sys_event(content):  # type: ignore[no-untyped-def]
        return SessionEvent(
            id="e",
            session_id="s",
            type="system/message",
            timestamp=0.0,
            payload={
                "message": {
                    "id": "m",
                    "role": "system",
                    "content": content,
                    "source": {"kind": "system-prompt"},
                }
            },
            seq=0,
        )

    assert _system_node_text(_sys_event([])) == ""
    assert _system_node_text(_sys_event([{"type": "text", "text": "hi"}])) == "hi"
    assert _system_node_text(_sys_event([{"type": "text", "text": 42}])) is None
    assert (
        _system_node_text(
            _sys_event([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        )
        is None
    )
    assert _system_node_text(_sys_event("not-a-list")) is None

    assert _tool_name_or_none({"type": "function", "function": {"name": "a"}}) == "a"
    assert _tool_name_or_none({"type": "function"}) is None
    assert _tool_name_or_none({"type": "function", "function": {}}) is None


@pytest.mark.asyncio
async def test_loop_next_turn_reads_max_from_log() -> None:
    """Verifies turn numbering takes the max logged turn, not the last."""
    from magitech_harness.session.store import Session

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()
    session = Session(session_id="turns")
    session.append("turn/start", {"turn": 2})
    session.append("turn/start", {"turn": 1})
    assert ctx.agent_loop._next_turn(session) == 3


@pytest.mark.asyncio
async def test_build_request_change_and_series_reasons() -> None:
    """Verifies _build_request reason taxonomy across successive requests."""
    ctx, mock_llm = await _ctx_with_mock()
    # Distinct from the registration name, so the route below proves
    # the header carries the provider default.
    mock_llm.default_model = "mock-model"
    session = ctx.sessions.create("sess_build_request")
    loop = ctx.agent_loop
    state = _RequestState(surface_generation=session.surface_content_generation)
    ping = {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "ping",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    request = loop._build_request(
        session, turn=1, step=1, tools=[], model=None, request_state=state
    )
    assert request["provider"] == "mock" and request["model"] == "mock-model"
    assert "tools" not in request
    headers = [e for e in session.events if e.type == EventType.REQUEST_HEADER]
    assert len(headers) == 1
    assert headers[0].payload["reason"] == "initial"

    # Tools added: changed header with startsSeries, plus a developer
    # message naming the addition against the new header.
    loop._build_request(
        session,
        turn=1,
        step=2,
        tools=[ping],
        model=None,
        starts_request_series=True,
        request_state=state,
    )
    headers = [e for e in session.events if e.type == EventType.REQUEST_HEADER]
    assert headers[1].payload["reason"] == "change"
    assert headers[1].payload["startsSeries"] is True
    developer = next(e for e in session.events if e.type == EventType.DEVELOPER_MESSAGE)
    assert developer.payload["headerSeq"] == headers[1].seq
    assert developer.payload["message"]["content"] == [
        {"type": "tool-addition", "toolName": "ping"}
    ]

    # Unchanged header with a new series: reason "series", no marker.
    loop._build_request(
        session,
        turn=1,
        step=3,
        tools=[ping],
        model=None,
        starts_request_series=True,
        request_state=state,
    )
    headers = [e for e in session.events if e.type == EventType.REQUEST_HEADER]
    assert headers[2].payload["reason"] == "series"
    assert "startsSeries" not in headers[2].payload

    # Config change, tools and surface unchanged: "change", no marker.
    loop._build_request(
        session,
        turn=1,
        step=4,
        tools=[ping],
        model="mock/other",
        request_state=state,
    )
    headers = [e for e in session.events if e.type == EventType.REQUEST_HEADER]
    assert headers[3].payload["reason"] == "change"
    assert "startsSeries" not in headers[3].payload

    # Unchanged header, no series: nothing appended.
    before = len(session.events)
    request = loop._build_request(
        session,
        turn=1,
        step=5,
        tools=[ping],
        model="mock/other",
        request_state=state,
    )
    assert len(session.events) == before
    assert request["provider"] == "mock" and request["model"] == "other"


@pytest.mark.asyncio
async def test_agent_loop_model_resolution() -> None:
    """Verifies model resolution: provider default, verbatim slugs, legacy model, fail-loud."""
    # No slug: the header and the dispatch use the provider's declared
    # default, never the registration name.
    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()
    recording = _RecordingProvider("real-model-x")
    ctx.llm.register_provider("prov", recording)
    session = ctx.sessions.create("sess_default_model")
    await ctx.agent_loop.run_turn(session, "hi")
    header = next(e for e in session.events if e.type == EventType.REQUEST_HEADER)
    assert header.payload["header"]["config"] == {
        "provider": "prov",
        "model": "real-model-x",
    }
    assert recording.seen_models == ["prov/real-model-x"]

    # Explicit slug for a registered provider resolves exactly.
    ctx2 = Context()
    ctx2.use(SessionService)
    ctx2.use(LLMService)
    ctx2.use(AgentLoopService)
    await ctx2.flush()
    recording2 = _RecordingProvider("other-default")
    ctx2.llm.register_provider("prov2", recording2)
    session2 = ctx2.sessions.create("sess_explicit_slug")
    await ctx2.agent_loop.run_turn(session2, "hi", model="prov2/explicit-model")
    header2 = next(e for e in session2.events if e.type == EventType.REQUEST_HEADER)
    assert header2.payload["header"]["config"] == {
        "provider": "prov2",
        "model": "explicit-model",
    }
    assert recording2.seen_models == ["prov2/explicit-model"]

    # Unregistered vendor segment: the whole slug is the model, and the
    # dispatch passes it verbatim to the active provider.
    ctx3 = Context()
    ctx3.use(SessionService)
    ctx3.use(LLMService)
    ctx3.use(AgentLoopService)
    await ctx3.flush()
    recording3 = _RecordingProvider("real-model-x")
    ctx3.llm.register_provider("prov", recording3)
    session3 = ctx3.sessions.create("sess_unregistered_vendor")
    await ctx3.agent_loop.run_turn(session3, "hi", model="openrouter/gpt-4o-mini")
    header3 = next(e for e in session3.events if e.type == EventType.REQUEST_HEADER)
    assert header3.payload["header"]["config"] == {
        "provider": "prov",
        "model": "openrouter/gpt-4o-mini",
    }
    assert recording3.seen_models == ["openrouter/gpt-4o-mini"]

    # A legacy ``model`` attribute is accepted where no default exists.
    ctx4 = Context()
    ctx4.use(SessionService)
    ctx4.use(LLMService)
    ctx4.use(AgentLoopService)
    await ctx4.flush()
    legacy = _RecordingProvider(model="legacy-model")
    ctx4.llm.register_provider("prov", legacy)
    session4 = ctx4.sessions.create("sess_legacy_model")
    await ctx4.agent_loop.run_turn(session4, "hi")
    header4 = next(e for e in session4.events if e.type == EventType.REQUEST_HEADER)
    assert header4.payload["header"]["config"] == {
        "provider": "prov",
        "model": "legacy-model",
    }
    assert legacy.seen_models == ["prov/legacy-model"]

    # No slug and no determinable model: fail loud before any header.
    ctx5 = Context()
    ctx5.use(SessionService)
    ctx5.use(LLMService)
    ctx5.use(AgentLoopService)
    await ctx5.flush()
    ctx5.llm.register_provider("prov", _RecordingProvider())
    session5 = ctx5.sessions.create("sess_no_model")
    with pytest.raises(NoProviderError):
        await ctx5.agent_loop.run_turn(session5, "hi")
    assert not any(e.type == EventType.REQUEST_HEADER for e in session5.events)


@pytest.mark.asyncio
async def test_agent_loop_no_provider_logs_no_header() -> None:
    """Verifies a turn with no provider fails before any request exists."""
    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()

    session = ctx.sessions.create("sess_no_provider")
    with pytest.raises(NoProviderError):
        await ctx.agent_loop.run_turn(session, "hi")

    assert not any(e.type == EventType.REQUEST_HEADER for e in session.events)
    turn_end = next(e for e in session.events if e.type == EventType.TURN_END)
    assert turn_end.payload["reason"]["kind"] == "error"
    assert turn_end.payload["reason"]["error"]["code"] == "NoProviderError"


@pytest.mark.asyncio
async def test_agent_loop_abort_during_call_ends_aborted() -> None:
    """Verifies an abort mid-call ends the turn aborted, after the request."""
    from magitech_harness.agent.cancellation import AbortError

    class _AbortingProvider:
        async def generate(self, messages, tools=None, system_prompt=None, **kwargs):  # type: ignore[no-untyped-def]
            raise AbortError("aborted in generate")

        async def stream(
            self,
            messages,
            tools=None,
            system_prompt=None,
            model=None,
            signal=None,
            **kwargs,
        ):  # type: ignore[no-untyped-def]
            raise AbortError("aborted in stream")
            yield  # makes this an async generator; unreachable by design

    ctx = Context()
    ctx.use(SessionService)
    ctx.use(LLMService)
    ctx.use(AgentLoopService)
    await ctx.flush()
    ctx.llm.register_provider("mock", _AbortingProvider())

    session = ctx.sessions.create("sess_abort_call")
    with pytest.raises(AbortError, match="aborted in generate"):
        # Explicit route: the aborting provider declares no default.
        await ctx.agent_loop.run_turn(session, "hi", model="mock/aborting")

    # The request was built before dispatch, so its header stands.
    assert any(e.type == EventType.REQUEST_HEADER for e in session.events)
    turn_end = next(e for e in session.events if e.type == EventType.TURN_END)
    assert turn_end.payload["reason"]["kind"] == "aborted"
    assert turn_end.payload["reason"]["reason"] == "aborted in generate"


@pytest.mark.asyncio
async def test_agent_loop_stream_max_tokens_turn_reason() -> None:
    """Verifies a streamed max-tokens step ends the turn max-tokens."""
    ctx, mock_llm = await _ctx_with_mock()
    mock_llm.responses.append(
        ModelResponse(
            text="done",
            usage=UsageData(total_tokens=5),
            finish_reason="max_tokens",
        )
    )
    session = ctx.sessions.create("sess_stream_max_tokens")
    yielded = []
    async for ev in ctx.agent_loop.stream_turn(session, "hi"):
        yielded.append(ev)

    assert {"type": "chunk", "text": "done"} in yielded
    assistant = next(e for e in session.events if e.type == EventType.ASSISTANT_MESSAGE)
    assert assistant.payload["usage"]["total_tokens"] == 5
    turn_end = next(e for e in session.events if e.type == EventType.TURN_END)
    assert turn_end.payload["reason"] == {"kind": "max-tokens"}


@pytest.mark.asyncio
async def test_agent_loop_stream_empty_content_without_usage_logs_nothing() -> None:
    """Verifies a streamed empty step with no usage logs no assistant event."""
    ctx, mock_llm = await _ctx_with_mock()
    mock_llm.queue_response(text="")
    session = ctx.sessions.create("sess_stream_empty")
    yielded = []
    async for ev in ctx.agent_loop.stream_turn(session, "hi"):
        yielded.append(ev)

    assert yielded == [{"type": "turn_end", "text": ""}]
    assert not any(e.type == EventType.ASSISTANT_MESSAGE for e in session.events)
    assert [message["role"] for message in session.derive_messages()] == ["user"]
