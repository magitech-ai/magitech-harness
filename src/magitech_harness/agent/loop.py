"""AgentLoopService (ctx.agent_loop): Turn and Step execution engine.
Implements the Magitech Harness turn execution state machine.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from anima import Context, Service

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.session.events import EventType, ToolCallData
from magitech_harness.session.store import Session

logger = logging.getLogger("harness.agent")


class AgentLoopService(Service):
    """Service managing multi-step agent turns, tool loops, and session logs."""

    # Explicitly inject dependencies
    inject: ClassVar[list[str]] = ["sessions", "llm"]

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "agent_loop")

    async def run_turn(
        self,
        session: Session,
        user_input: str,
        *,
        model: str | None = None,
        max_steps: int = 10,
        signal: AbortSignal | None = None,
    ) -> str:
        """Run an agent turn to completion across multiple steps."""
        if signal:
            signal.throw_if_aborted()

        turn_id = str(uuid.uuid4())

        # 1. turn/start
        self.ctx.sessions.log_event(
            session,
            EventType.TURN_START,
            {"turn_id": turn_id},
        )
        self.ctx.emit("turn/start", session, turn_id)

        # 2. Append user message to durable session
        self.ctx.sessions.log_event(
            session,
            EventType.USER_MESSAGE,
            {"text": user_input, "turn_id": turn_id},
        )

        step_index = 0
        final_assistant_text = ""

        try:
            while step_index < max_steps:
                if signal:
                    signal.throw_if_aborted()

                step_index += 1

                # 3. step/start
                self.ctx.sessions.log_event(
                    session,
                    EventType.STEP_START,
                    {"step_index": step_index, "turn_id": turn_id},
                )
                self.ctx.emit("step/start", session, step_index)

                # 4. agent/pre-step waterfall (opportunity to guard, rewrite, or reject)
                pre_step_decision = await self.ctx.waterfall(
                    "agent/pre-step",
                    {"action": "continue", "step_index": step_index},
                    default_handler=lambda dec: dec,
                )

                if pre_step_decision.get("action") == "reject":
                    logger.warning(
                        "Step %d rejected by pre-step waterfall hook", step_index
                    )
                    break

                # 5. Assemble prompt & tools
                system_prompt = (
                    await self.ctx.system_prompt.render()
                    if self.ctx.has("system_prompt")
                    else ""
                )
                tool_schemas = (
                    self.ctx.tools.get_schemas() if self.ctx.has("tools") else []
                )

                # 6. Reconstruct messages deterministically from session log
                messages = session.derive_messages()

                # 7. Model request
                response = await self.ctx.llm.call(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    system_prompt=system_prompt if system_prompt else None,
                    model=model,
                )

                # 8. Log assistant response
                self.ctx.sessions.log_event(
                    session,
                    EventType.ASSISTANT_MESSAGE,
                    {
                        "text": response.text,
                        "tool_calls": [tc.model_dump() for tc in response.tool_calls],
                        "turn_id": turn_id,
                    },
                )
                final_assistant_text = response.text

                # 9. Handle tool calls if any
                if response.tool_calls and self.ctx.has("tools"):
                    for tc in response.tool_calls:
                        if signal:
                            signal.throw_if_aborted()

                        # Log tool call event
                        self.ctx.sessions.log_event(
                            session,
                            EventType.TOOL_CALL,
                            tc.model_dump(),
                        )

                        # Execute via tool pipeline
                        tool_result = await self.ctx.tools.execute(tc)

                        # Log tool result event
                        self.ctx.sessions.log_event(
                            session,
                            EventType.TOOL_RESULT,
                            tool_result.model_dump(),
                        )

                    # Step end after tool execution; loop continues to next step
                    self.ctx.sessions.log_event(
                        session,
                        EventType.STEP_END,
                        {"step_index": step_index, "owed_step": True},
                    )
                    self.ctx.emit("step/end", session, step_index)
                else:
                    # No tool calls; turn is finished
                    self.ctx.sessions.log_event(
                        session,
                        EventType.STEP_END,
                        {"step_index": step_index, "owed_step": False},
                    )
                    self.ctx.emit("step/end", session, step_index)
                    break

        finally:
            # 10. turn/end
            self.ctx.emit("agent/turn-stopping", session, turn_id)
            self.ctx.sessions.log_event(
                session,
                EventType.TURN_END,
                {"turn_id": turn_id, "total_steps": step_index},
            )
            self.ctx.emit("turn/end", session, turn_id)

        return final_assistant_text

    async def stream_turn(
        self,
        session: Session,
        user_input: str,
        *,
        model: str | None = None,
        max_steps: int = 10,
        signal: AbortSignal | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream an agent turn step-by-step, yielding token chunks and tool lifecycle events."""
        if signal:
            signal.throw_if_aborted()

        turn_id = str(uuid.uuid4())

        self.ctx.sessions.log_event(
            session,
            EventType.TURN_START,
            {"turn_id": turn_id},
        )
        self.ctx.emit("turn/start", session, turn_id)

        self.ctx.sessions.log_event(
            session,
            EventType.USER_MESSAGE,
            {"text": user_input, "turn_id": turn_id},
        )

        step_index = 0
        final_assistant_text = ""

        try:
            while step_index < max_steps:
                if signal:
                    signal.throw_if_aborted()

                step_index += 1

                self.ctx.sessions.log_event(
                    session,
                    EventType.STEP_START,
                    {"step_index": step_index, "turn_id": turn_id},
                )
                self.ctx.emit("step/start", session, step_index)

                pre_step_decision = await self.ctx.waterfall(
                    "agent/pre-step",
                    {"action": "continue", "step_index": step_index},
                    default_handler=lambda dec: dec,
                )

                if pre_step_decision.get("action") == "reject":
                    logger.warning(
                        "Step %d rejected by pre-step waterfall hook", step_index
                    )
                    break

                system_prompt = (
                    await self.ctx.system_prompt.render()
                    if self.ctx.has("system_prompt")
                    else ""
                )
                tool_schemas = (
                    self.ctx.tools.get_schemas() if self.ctx.has("tools") else []
                )
                messages = session.derive_messages()

                accumulated_text = ""
                collected_tool_calls: list[ToolCallData] = []

                async for chunk in self.ctx.llm.stream(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    system_prompt=system_prompt if system_prompt else None,
                    model=model,
                    signal=signal,
                ):
                    if chunk.text:
                        accumulated_text += chunk.text
                        yield {"type": "chunk", "text": chunk.text}
                    if chunk.tool_calls:
                        collected_tool_calls.extend(chunk.tool_calls)

                final_assistant_text = accumulated_text
                self.ctx.sessions.log_event(
                    session,
                    EventType.ASSISTANT_MESSAGE,
                    {
                        "text": accumulated_text,
                        "tool_calls": [tc.model_dump() for tc in collected_tool_calls],
                        "turn_id": turn_id,
                    },
                )

                if collected_tool_calls and self.ctx.has("tools"):
                    for tc in collected_tool_calls:
                        if signal:
                            signal.throw_if_aborted()

                        yield {
                            "type": "tool_start",
                            "name": tc.name,
                            "call_id": tc.call_id,
                            "arguments": tc.arguments,
                        }

                        self.ctx.sessions.log_event(
                            session,
                            EventType.TOOL_CALL,
                            tc.model_dump(),
                        )

                        tool_result = await self.ctx.tools.execute(tc)

                        self.ctx.sessions.log_event(
                            session,
                            EventType.TOOL_RESULT,
                            tool_result.model_dump(),
                        )

                        yield {
                            "type": "tool_end",
                            "name": tc.name,
                            "call_id": tc.call_id,
                            "result": tool_result.content,
                            "is_error": tool_result.is_error,
                        }

                    self.ctx.sessions.log_event(
                        session,
                        EventType.STEP_END,
                        {"step_index": step_index, "owed_step": True},
                    )
                    self.ctx.emit("step/end", session, step_index)
                else:
                    self.ctx.sessions.log_event(
                        session,
                        EventType.STEP_END,
                        {"step_index": step_index, "owed_step": False},
                    )
                    self.ctx.emit("step/end", session, step_index)
                    break
        finally:
            self.ctx.emit("agent/turn-stopping", session, turn_id)
            self.ctx.sessions.log_event(
                session,
                EventType.TURN_END,
                {"turn_id": turn_id, "total_steps": step_index},
            )
            self.ctx.emit("turn/end", session, turn_id)

        yield {"type": "turn_end", "text": final_assistant_text}
