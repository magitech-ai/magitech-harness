import json
from pathlib import Path

import pytest
from anima import EventMode

from magitech_harness import (
    EventType,
    MockLLMProvider,
    ToolCallData,
    create_harness,
    define_tool,
)
from magitech_harness.llm.types import ModelChunk
from magitech_harness.session.messages import create_user_message
from magitech_harness.session.store import Session


@pytest.mark.asyncio
async def test_full_turn_with_tool_call(tmp_path: Path):
    # 1. Boot harness
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()

    # Verify all 6 capability seams are active
    assert ctx.has("sessions")
    assert ctx.has("tools")
    assert ctx.has("system_prompt")
    assert ctx.has("llm")
    assert ctx.has("agent_loop")
    assert ctx.has("shell")

    # 2. Register custom tool
    executed_tools = []

    def multiply(args: dict) -> int:
        a = args.get("a", 0)
        b = args.get("b", 0)
        res = a * b
        executed_tools.append((a, b, res))
        return res

    ctx.tools.register(
        define_tool(
            name="multiply",
            description="Multiplies two numbers",
            parameters={
                "a": {"type": "integer", "required": True},
                "b": {"type": "integer", "required": True},
            },
            execute=multiply,
        )
    )

    # 3. Configure mock LLM responses for multi-step turn:
    # Step 1: LLM returns tool_call "multiply" (a=6, b=7)
    # Step 2: LLM returns final text answer "The answer is 42."
    mock_llm: MockLLMProvider = ctx.llm.active_provider
    mock_llm.queue_response(
        text="Let me calculate that for you.",
        tool_calls=[
            ToolCallData(call_id="call_1", name="multiply", arguments={"a": 6, "b": 7})
        ],
    )
    mock_llm.queue_response(
        text="The answer is 42.",
        tool_calls=[],
    )

    # 4. Run the turn
    session = ctx.sessions.create("test_session_1")
    final_output = await ctx.agent_loop.run_turn(session, "What is 6 * 7?")

    # Verify output & execution
    assert final_output == "The answer is 42."
    assert executed_tools == [(6, 7, 42)]

    # 5. Verify durable session events in canonical shapes. The
    # deferred fallback title may land mid-turn (tool execution yields
    # to the loop), so it is excluded from the turn's exact sequence
    # and asserted separately below.
    event_types = [
        ev.type for ev in session.events if ev.type != EventType.SESSION_TITLE
    ]
    assert event_types == [
        EventType.TURN_START,
        EventType.STEP_START,
        EventType.SYSTEM_MESSAGE,
        EventType.USER_MESSAGE,
        EventType.REQUEST_HEADER,
        EventType.REQUEST_CONTEXT,
        EventType.ASSISTANT_MESSAGE,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.STEP_END,
        EventType.STEP_START,
        EventType.ASSISTANT_MESSAGE,
        EventType.STEP_END,
        EventType.TURN_END,
    ]
    by_type = {ev.type: ev for ev in session.events}
    assert by_type[EventType.TURN_START].payload == {"turn": 1}
    # No prompt sections registered: the empty prompt reserves node 0.
    assert by_type[EventType.SYSTEM_MESSAGE].payload["message"]["content"] == []
    user = by_type[EventType.USER_MESSAGE].payload
    assert user["content"] == [{"type": "text", "text": "What is 6 * 7?"}]
    assert user["source"] == {"kind": "user"}
    header = by_type[EventType.REQUEST_HEADER].payload
    assert header["reason"] == "initial"
    assert header["header"]["config"] == {"provider": "mock", "model": "mock"}
    assert [tool["function"]["name"] for tool in header["header"]["tools"]] == [
        "multiply"
    ]
    tool_call = by_type[EventType.TOOL_CALL].payload
    assert tool_call == {
        "turn": 1,
        "step": 1,
        "callId": "call_1",
        "name": "multiply",
        "arguments": '{"a":6,"b":7}',
    }

    # 6. Verify derive_messages deterministic projection (verbatim blocks).
    projected = session.derive_messages()
    assert len(projected) == 4
    assert projected[0]["role"] == "user"
    assert projected[0]["content"] == [{"type": "text", "text": "What is 6 * 7?"}]
    assert projected[1]["role"] == "assistant"
    assert projected[1]["content"] == [
        {"type": "text", "text": "Let me calculate that for you."},
        {
            "type": "tool-call",
            "id": "call_1",
            "name": "multiply",
            "arguments": '{"a":6,"b":7}',
        },
    ]
    assert projected[2]["role"] == "tool"
    assert projected[2]["content"] == [{"type": "text", "text": "42"}]
    assert projected[3]["role"] == "assistant"
    assert projected[3]["content"] == [{"type": "text", "text": "The answer is 42."}]

    # 7. Verify JSONL file persistence on disk: one v4 header line plus
    # one physical row per event. The deferred fallback title commits on
    # drain, so the file holds the 14 turn events plus the title event.
    await ctx.sessions.title_service.drain()
    await ctx.sessions.flush()
    titles = [ev for ev in session.events if ev.type == EventType.SESSION_TITLE]
    assert len(titles) == 1
    assert titles[0].payload["title"] == "What is 6 * 7?"
    assert titles[0].payload["source"] == {"kind": "fallback"}
    location = ctx.sessions.location(session.session_id)
    assert location is not None
    assert location.exists()
    lines = location.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 16
    header = json.loads(lines[0])
    assert header["type"] == "session"
    assert header["version"] == 4
    assert header["id"] == session.session_id

    # 8. Replay leg: loading the written log re-derives the same messages.
    reloaded = Session.load(location)
    assert reloaded is not None
    assert reloaded.derive_messages() == projected


@pytest.mark.asyncio
async def test_waterfall_guard_pre_step_veto(tmp_path: Path):
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()

    # Install a security component via waterfall hook on agent/pre-step
    async def security_guard(decision: dict, next_fn):
        if decision.get("step_index") == 1:
            return {"action": "reject", "reason": "blocked by policy"}
        return await next_fn(decision)

    ctx.on("agent/pre-step", security_guard, mode=EventMode.WATERFALL)

    session = ctx.sessions.create("blocked_session")
    res = await ctx.agent_loop.run_turn(session, "Malicious instruction")

    # Since step 1 was rejected, assistant text is empty and 0 steps ran.
    # The user message is logged inside the step attempt, after the
    # pre-step waterfall, so a veto logs no user message; the entered
    # step still ends exactly once.
    assert res == ""
    event_types = [ev.type for ev in session.events]
    assert event_types == [
        EventType.TURN_START,
        EventType.STEP_START,
        EventType.STEP_END,
        EventType.TURN_END,
    ]


@pytest.mark.asyncio
async def test_shell_service_execution(tmp_path: Path):
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    code, stdout, _stderr = await ctx.shell.run("echo 'Magitech Rocks'")
    assert code == 0
    assert "Magitech Rocks" in stdout


class _ExplodingStreamProvider:
    """Provider whose stream yields one chunk then raises."""

    async def generate(self, messages, tools=None, system_prompt=None, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("stream exploded")

    async def stream(  # type: ignore[no-untyped-def]
        self,
        messages,
        tools=None,
        system_prompt=None,
        model=None,
        signal=None,
        **kwargs,
    ):
        yield ModelChunk(text="partial output")
        raise RuntimeError("stream exploded")


@pytest.mark.asyncio
async def test_replay_reproduces_identical_message_sequence(tmp_path: Path):
    """Headline acceptance: replay of a recorded session reproduces the
    identical message sequence — same messages, same order, same system
    prompt, no merged nodes, no invented user messages."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()

    ctx.system_prompt.register_section("main", "You are a calculator.")
    ctx.tools.register(
        define_tool(
            name="multiply",
            description="Multiplies two numbers",
            parameters={
                "a": {"type": "integer", "required": True},
                "b": {"type": "integer", "required": True},
            },
            execute=lambda args: args.get("a", 0) * args.get("b", 0),
        )
    )
    mock_llm: MockLLMProvider = ctx.llm.active_provider
    mock_llm.queue_response(
        text="Let me calculate that for you.",
        tool_calls=[
            ToolCallData(call_id="call_1", name="multiply", arguments={"a": 6, "b": 7})
        ],
    )
    mock_llm.queue_response(text="The answer is 42.", tool_calls=[])
    mock_llm.queue_response(text="Noted.")

    session = ctx.sessions.create("replay_session")
    await ctx.agent_loop.run_turn(session, "What is 6 * 7?")

    # Two consecutive same-role user messages, appended directly.
    first = ctx.sessions.log_event(
        session,
        EventType.USER_MESSAGE,
        create_user_message(
            content=[{"type": "text", "text": "Follow-up one."}],
            source={"kind": "user"},
        ),
    )
    second = ctx.sessions.log_event(
        session,
        EventType.USER_MESSAGE,
        create_user_message(
            content=[{"type": "text", "text": "Follow-up two."}],
            source={"kind": "user"},
        ),
    )

    # A changed rendering normalizes the system head by replace.
    ctx.system_prompt.register_section("main", "You are a calculator. Be brief.")
    await ctx.agent_loop.run_turn(session, "Thanks.")

    # One failed streaming attempt, driven through the loop.
    ctx.llm.register_provider("failing", _ExplodingStreamProvider())
    with pytest.raises(RuntimeError, match="stream exploded"):
        async for _ in ctx.agent_loop.stream_turn(
            session, "Boom prompt", model="failing/failing"
        ):
            pass

    # One compaction range shadowing the two consecutive user messages.
    start_event = ctx.sessions.log_event(
        session, "compaction/start", {"compactionId": "c1", "turn": 3}
    )
    summary_blocks = [{"type": "text", "text": "Follow-ups about the product."}]
    summary_event = ctx.sessions.log_event(
        session,
        "compaction/summary",
        {
            "compactionId": "c1",
            "summary": summary_blocks,
            "shadowedRange": {"start": first.seq, "end": second.seq},
            "shadowedSeqs": [first.seq, second.seq],
            "shadowedTokenCount": 10,
            "provider": "mock",
            "model": "mock",
        },
    )
    replacement = create_user_message(
        content=[
            {"type": "text", "text": "Summary of earlier conversation:\n"},
            *summary_blocks,
            {"type": "text", "text": "\nEnd of summary."},
        ],
        source={"kind": "compact-checkpoint", "compactionId": "c1"},
    )
    ctx.sessions.log_event(
        session,
        EventType.USER_MESSAGE,
        replacement,
        surface_op={"op": "replace", "startSeq": first.seq, "endSeq": second.seq},
        source_event_seqs=[start_event.seq, summary_event.seq, first.seq, second.seq],
    )

    live = session.derive_messages()
    assert [message["role"] for message in live] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "user",
        "assistant",
        "user",
    ]
    # The system head holds the normalized rendering, once.
    assert live[0]["content"] == [
        {"type": "text", "text": "You are a calculator. Be brief."}
    ]
    # The replacement stands where the shadowed range was; the two
    # consecutive user messages after it stay separate nodes, unmerged.
    assert live[5] == replacement
    assert live[5]["content"][0]["text"].startswith("Summary of earlier")
    assert live[6]["content"] == [{"type": "text", "text": "Thanks."}]
    # The failed attempt is invisible; nothing was synthesized for it.
    assert not any("partial output" in str(m) for m in live)
    assert not any("[Conversation summary" in str(m) for m in live)

    await ctx.sessions.title_service.drain()
    await ctx.sessions.flush()
    location = ctx.sessions.location(session.session_id)
    assert location is not None
    reloaded = Session.load(location)
    assert reloaded is not None
    assert reloaded.derive_messages() == live


@pytest.mark.asyncio
async def test_run_turn_and_stream_turn_log_identical_messages(tmp_path: Path):
    """Verifies a run_turn and a stream_turn log the identical messages."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()

    ctx.tools.register(
        define_tool(
            name="multiply",
            description="Multiplies two numbers",
            parameters={
                "a": {"type": "integer", "required": True},
                "b": {"type": "integer", "required": True},
            },
            execute=lambda args: args.get("a", 0) * args.get("b", 0),
        )
    )

    mock_llm: MockLLMProvider = ctx.llm.active_provider
    for _ in range(2):
        mock_llm.queue_response(
            text="Let me calculate that for you.",
            tool_calls=[
                ToolCallData(
                    call_id="call_1", name="multiply", arguments={"a": 6, "b": 7}
                )
            ],
        )
        mock_llm.queue_response(text="The answer is 42.", tool_calls=[])

    run_session = ctx.sessions.create("run_session")
    run_result = await ctx.agent_loop.run_turn(run_session, "What is 6 * 7?")
    assert run_result == "The answer is 42."

    stream_session = ctx.sessions.create("stream_session")
    streamed = [
        chunk
        async for chunk in ctx.agent_loop.stream_turn(stream_session, "What is 6 * 7?")
    ]
    assert streamed[-1] == {"type": "turn_end", "text": "The answer is 42."}

    # The deferred fallback title may land in either turn, so it is
    # excluded from the exact sequence and asserted separately below.
    # Message ids are generated per message and the stream turn records
    # its streamed chunks (with wall-clock times) in the assistant
    # message payload, so both are stripped before comparing payloads;
    # everything else must match exactly.
    def normalise(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: normalise(item)
                for key, item in value.items()
                if key not in ("id", "stream")
            }
        if isinstance(value, list):
            return [normalise(item) for item in value]
        return value

    def logged(session: Session) -> list[tuple[str, object]]:
        return [
            (ev.type, normalise(ev.payload))
            for ev in session.events
            if ev.type != EventType.SESSION_TITLE
        ]

    assert logged(run_session) == logged(stream_session)
    assert normalise(run_session.derive_messages()) == normalise(
        stream_session.derive_messages()
    )

    await ctx.sessions.title_service.drain()
    await ctx.sessions.flush()
    for session in (run_session, stream_session):
        titles = [ev for ev in session.events if ev.type == EventType.SESSION_TITLE]
        assert len(titles) == 1
        assert titles[0].payload["title"] == "What is 6 * 7?"
        assert titles[0].payload["source"] == {"kind": "fallback"}
