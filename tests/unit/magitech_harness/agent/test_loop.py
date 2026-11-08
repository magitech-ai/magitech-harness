"""Unit tests for AgentLoopService."""

import pytest
from anima import Context, EventMode

from magitech_harness.agent.loop import AgentLoopService
from magitech_harness.llm.service import LLMService, MockLLMProvider
from magitech_harness.llm.types import UsageData
from magitech_harness.session.events import EventType, ToolCallData
from magitech_harness.session.service import SessionService


@pytest.mark.asyncio
async def test_agent_loop_single_step_no_tools() -> None:
    """Verifies a 1-step turn when the LLM returns plain text with no tool calls."""
    ctx = Context()
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(AgentLoopService)
    await ctx.flush()

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="Direct answer")

    session = ctx.sessions.create("sess_1")
    reply = await ctx.agent_loop.run_turn(session, "Hello agent")

    assert reply == "Direct answer"
    types = [e.type for e in session.events]
    assert types == [
        EventType.TURN_START,
        EventType.USER_MESSAGE,
        EventType.STEP_START,
        EventType.ASSISTANT_MESSAGE,
        EventType.STEP_END,
        EventType.TURN_END,
    ]


@pytest.mark.asyncio
async def test_agent_loop_pre_step_reject_short_circuit() -> None:
    """Verifies that pre-step waterfall veto immediately stops the turn."""
    ctx = Context()
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(AgentLoopService)
    await ctx.flush()

    async def guard(decision: dict, next_fn):
        return {"action": "reject"}

    ctx.on("agent/pre-step", guard, mode=EventMode.WATERFALL)

    session = ctx.sessions.create("sess_veto")
    reply = await ctx.agent_loop.run_turn(session, "Prompt")

    assert reply == ""
    types = [e.type for e in session.events]
    assert types == [
        EventType.TURN_START,
        EventType.USER_MESSAGE,
        EventType.STEP_START,
        EventType.TURN_END,
    ]


@pytest.mark.asyncio
async def test_agent_loop_max_steps_termination() -> None:
    """Verifies that reaching max_steps terminates the loop even if tool calls continue."""
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(ToolService)
    ctx.plugin(AgentLoopService)
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
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(AgentLoopService)
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
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(AgentLoopService)
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
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(ToolService)
    ctx.plugin(AgentLoopService)
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
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(AgentLoopService)
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
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(AgentLoopService)
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
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(ToolService)
    ctx.plugin(AgentLoopService)
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
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(ToolService)
    ctx.plugin(AgentLoopService)
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
