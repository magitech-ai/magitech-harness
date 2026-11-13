"""AgentLoopService (ctx.agent_loop): Turn and Step execution engine.
Implements the Magitech Harness turn execution state machine.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar

from anima import Context, Service

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.agent.system_prompt import SystemPromptProjection
from magitech_harness.llm.assistant_stream import AssistantStreamAccumulator
from magitech_harness.llm.service import NoProviderError
from magitech_harness.llm.slug import parse_model_slug
from magitech_harness.llm.types import UsageData
from magitech_harness.session.events import (
    EventType,
    SessionEvent,
    ToolCallData,
    ToolResultData,
)
from magitech_harness.session.messages import (
    create_assistant_message,
    create_developer_message,
    create_tool_result_message,
    create_user_message,
)
from magitech_harness.session.request_header import (
    canonical_header,
    header_equals,
)
from magitech_harness.session.store import Session
from magitech_harness.session.tool_history import tool_name

logger = logging.getLogger("harness.agent")


def _tool_result_payload(
    tc: ToolCallData, result: ToolResultData, *, turn: int, step: int
) -> dict[str, Any]:
    """Build the admission-shaped tool/result payload for one tool call.

    The result is carried as a tool-role message — the shape the v4
    admission check requires — with a factory-minted message id.
    """
    content = result.content
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return {
        "turn": turn,
        "step": step,
        "message": create_tool_result_message(
            call_id=tc.call_id,
            content=[{"type": "text", "text": text}],
            is_error=result.is_error,
        ),
    }


def _has_usage(usage: UsageData) -> bool:
    """Whether the adapter reported any token accounting."""
    return (
        usage.prompt_tokens != 0
        or usage.completion_tokens != 0
        or usage.total_tokens != 0
        or usage.reasoning_tokens != 0
    )


def _tool_call_blocks(tool_calls: list[ToolCallData]) -> list[dict[str, Any]]:
    """One ``tool-call`` content block per call, arguments re-serialized."""
    return [
        {
            "type": "tool-call",
            "id": tc.call_id,
            "name": tc.name,
            "arguments": json.dumps(
                tc.arguments, separators=(",", ":"), ensure_ascii=False
            ),
        }
        for tc in tool_calls
    ]


def _tool_calls_from_message(message: dict[str, Any]) -> list[ToolCallData]:
    """Parse the ``tool-call`` blocks of a logged message back into calls.

    A block whose arguments fail to parse is a corrupt log: fail loud
    rather than executing a guessed call.
    """
    calls: list[ToolCallData] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict) or block.get("type") != "tool-call":
            continue
        try:
            arguments = json.loads(block["arguments"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"tool-call block {block.get('id')!r} has unparseable arguments"
            ) from exc
        if not isinstance(arguments, dict):
            raise TypeError(
                f"tool-call block {block.get('id')!r} arguments must be an object"
            )
        calls.append(
            ToolCallData(call_id=block["id"], name=block["name"], arguments=arguments)
        )
    return calls


def _system_node_text(event: SessionEvent) -> str | None:
    """Effective text of a surface ``system/message`` node.

    ``""`` for empty content; ``None`` when the content is not a single
    text block.
    """
    message = event.payload.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return None
    if content == []:
        return ""
    if (
        len(content) == 1
        and isinstance(content[0], dict)
        and content[0].get("type") == "text"
    ):
        text = content[0].get("text")
        if isinstance(text, str):
            return text
    return None


def _tool_name_or_none(schema: dict[str, Any]) -> str | None:
    """The envelope tool name, or ``None`` when the schema has none."""
    try:
        return tool_name(schema)
    except (ValueError, TypeError):
        return None


@dataclass
class _RequestState:
    """Per-turn request state: fresh turn driver, fresh state."""

    header_logged: bool = False
    # The surface content generation at the last request; set from the
    # session at turn start.
    surface_generation: int = 0


class AgentLoopService(Service):
    """Service managing multi-step agent turns, tool loops, and session logs."""

    # Explicitly inject dependencies
    inject: ClassVar[list[str]] = ["sessions", "llm"]

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, "agent_loop")

    # -- request construction -------------------------------------------------

    def _resolve_route(self, model: str | None) -> tuple[str, str]:
        """Resolve the ``(provider, model)`` pair the dispatch will use.

        The slug's provider wins when registered; otherwise the active
        provider's registered name. The model is the slug's model part,
        else the provider object's ``model`` attribute when it is a
        non-empty string, else the provider registration name. Raises
        ``NoProviderError`` when no provider resolves, before any header
        is logged.
        """
        provider_name, model_name = parse_model_slug(model or "")
        providers = self.ctx.llm.providers
        if provider_name and provider_name in providers:
            provider = providers[provider_name]
        else:
            provider = self.ctx.llm.active_provider
            if provider is None:
                raise NoProviderError(
                    "No active LLM provider component registered. Please "
                    "install or configure a provider component or specify "
                    "one in your profile.yaml."
                )
            provider_name = next(
                name for name, candidate in providers.items() if candidate is provider
            )
        if not model_name:
            candidate = getattr(provider, "model", None)
            model_name = (
                candidate if isinstance(candidate, str) and candidate else provider_name
            )
        return provider_name, model_name

    def _starts_series(
        self,
        session: Session,
        request_state: _RequestState,
        tools: list[dict[str, Any]],
    ) -> bool:
        """Whether this attempt starts a new request series.

        The pre-step declaration is always ``False`` in this chunk (no
        pre-step series signal exists); a surface change since the last
        request or a changed tool list starts a series.
        """
        baseline = session.request_header()
        tools_changed = False
        if baseline is not None:
            tools_changed = not header_equals(
                baseline, canonical_header({**baseline, "tools": list(tools)})
            )
        return (
            request_state.surface_generation != session.surface_content_generation
            or tools_changed
        )

    def _system_nodes(self, session: Session) -> list[dict[str, Any]]:
        """The surviving ``system/message`` nodes in surface order."""
        nodes: list[dict[str, Any]] = []
        for seq in session.surface_nodes():
            event = session.events[seq]
            if event.type != "system/message":
                continue
            nodes.append({"seq": seq, "text": _system_node_text(event)})
        return nodes

    def _build_request(
        self,
        session: Session,
        *,
        turn: int,
        step: int,
        tools: list[dict[str, Any]],
        model: str | None,
        starts_request_series: bool = False,
        request_state: _RequestState,
    ) -> dict[str, Any]:
        """Log the resolved header/context/developer events for one step's
        request; return the request dict the model call consumes."""
        provider, model_name = self._resolve_route(model)
        header = canonical_header(
            {
                "config": {"provider": provider, "model": model_name},
                **({"tools": tools} if tools else {}),
            }
        )
        baseline = session.request_header()
        header_seq: int | None = None
        if not request_state.header_logged:
            reason = "initial" if baseline is None else "resume"
            payload: dict[str, Any] = {
                "header": header,
                "reason": reason,
            }
            if starts_request_series:
                payload["startsSeries"] = True
            event = self.ctx.sessions.log_event(
                session, EventType.REQUEST_HEADER, payload
            )
            header_seq = event.seq
            request_state.header_logged = True
        elif baseline is None or not header_equals(baseline, header):
            payload = {"header": header, "reason": "change"}
            if starts_request_series:
                payload["startsSeries"] = True
            event = self.ctx.sessions.log_event(
                session, EventType.REQUEST_HEADER, payload
            )
            header_seq = event.seq
        elif starts_request_series:
            event = self.ctx.sessions.log_event(
                session,
                EventType.REQUEST_HEADER,
                {"header": header, "reason": "series"},
            )
            header_seq = event.seq
        # The comparison above is against the generation at the last
        # request, not at turn start.
        request_state.surface_generation = session.surface_content_generation
        if baseline is not None and header_seq is not None:
            baseline_tools = baseline.get("tools") or []
            baseline_names = {
                name
                for tool in baseline_tools
                if (name := _tool_name_or_none(tool)) is not None
            }
            current_names = {
                name for tool in tools if (name := _tool_name_or_none(tool)) is not None
            }
            additions = [
                tool
                for tool in tools
                if _tool_name_or_none(tool) is not None
                and _tool_name_or_none(tool) not in baseline_names
            ]
            removals = [
                tool
                for tool in baseline_tools
                if _tool_name_or_none(tool) is not None
                and _tool_name_or_none(tool) not in current_names
            ]
            if additions or removals:
                blocks: list[dict[str, Any]] = [
                    {"type": "tool-addition", "toolName": _tool_name_or_none(tool)}
                    for tool in additions
                ]
                blocks.extend(
                    {"type": "tool-removal", "toolName": _tool_name_or_none(tool)}
                    for tool in removals
                )
                message = create_developer_message(
                    content=blocks, source={"kind": "tool-registry"}
                )
                developer_payload: dict[str, Any] = {
                    "turn": turn,
                    "step": step,
                    "message": message,
                }
                if additions:
                    developer_payload["headerSeq"] = header_seq
                self.ctx.sessions.log_event(
                    session, EventType.DEVELOPER_MESSAGE, developer_payload
                )
        request_context = {"provider": provider, "model": model_name}
        previous = session.request_context()
        if (
            previous is None
            or previous.get("provider") != provider
            or previous.get("model") != model_name
            or previous.get("contextWindow") != request_context.get("contextWindow")
            or previous.get("systemPromptUpdate")
            != request_context.get("systemPromptUpdate")
        ):
            self.ctx.sessions.log_event(
                session, EventType.REQUEST_CONTEXT, request_context
            )
        request: dict[str, Any] = {
            **header["config"],
            "messages": session.derive_messages(),
            "tool_history": session.tool_history(),
        }
        if "tools" in header:
            request["tools"] = header["tools"]
        return request

    # -- shared step scaffolding ----------------------------------------------

    def _next_turn(self, session: Session) -> int:
        """The next turn number, derived from the log's ``turn/start`` turns."""
        highest = 0
        for event in session.events:
            if event.type != "turn/start":
                continue
            value = event.payload.get("turn")
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > highest
            ):
                highest = value
        return highest + 1

    async def _open_step(
        self,
        session: Session,
        *,
        turn: int,
        step: int,
        user_input: str,
        model: str | None,
        request_state: _RequestState,
    ) -> dict[str, Any] | None:
        """Run the pre-step waterfall, system commits, user logging, and
        request build for one step. Returns the request dict, or ``None``
        when the pre-step waterfall rejected the step."""
        pre_step_decision = await self.ctx.waterfall(
            "agent/pre-step",
            {
                "action": "continue",
                "step_index": step,
                "session": session,
            },
            default_handler=lambda dec: dec,
        )
        if pre_step_decision.get("action") == "reject":
            logger.warning("Step %d rejected by pre-step waterfall hook", step)
            return None
        system_prompt = (
            await self.ctx.system_prompt.render()
            if self.ctx.has("system_prompt")
            else ""
        )
        tool_schemas = self.ctx.tools.get_schemas() if self.ctx.has("tools") else []
        starts_series = self._starts_series(session, request_state, tool_schemas)
        projection = SystemPromptProjection(lambda: self._system_nodes(session))
        for commit in projection.project(
            system_prompt, in_history=False, starts_series=starts_series
        ):
            self.ctx.sessions.log_event(
                session,
                EventType.SYSTEM_MESSAGE,
                {"turn": turn, "step": step, "message": commit.message},
                surface_op=commit.intent["surface_op"],
                source_event_seqs=commit.intent.get("source_event_seqs"),
            )
        if step == 1:
            self.ctx.sessions.log_event(
                session,
                EventType.USER_MESSAGE,
                create_user_message(
                    content=[{"type": "text", "text": user_input}],
                    source={"kind": "user"},
                ),
            )
        return self._build_request(
            session,
            turn=turn,
            step=step,
            tools=tool_schemas,
            model=model,
            starts_request_series=starts_series,
            request_state=request_state,
        )

    def _log_assistant_message(
        self,
        session: Session,
        *,
        turn: int,
        step: int,
        message: dict[str, Any],
        stream: list[dict[str, Any]],
        usage: UsageData | None = None,
        interrupted: bool = False,
    ) -> SessionEvent:
        """Log one ``assistant/message``; ``usage`` only when reported."""
        payload: dict[str, Any] = {
            "turn": turn,
            "step": step,
            "message": message,
            "stream": stream,
        }
        if usage is not None and _has_usage(usage):
            payload["usage"] = usage.model_dump()
        if interrupted:
            payload["interrupted"] = True
        event: SessionEvent = self.ctx.sessions.log_event(
            session, EventType.ASSISTANT_MESSAGE, payload
        )
        return event

    async def _run_tool_call(
        self,
        session: Session,
        *,
        turn: int,
        step: int,
        tc: ToolCallData,
        signal: AbortSignal | None,
    ) -> ToolResultData:
        """Log ``tool/call``, execute, log ``tool/result``; return the result."""
        if signal:
            signal.throw_if_aborted()
        self.ctx.sessions.log_event(
            session,
            EventType.TOOL_CALL,
            {
                "turn": turn,
                "step": step,
                "callId": tc.call_id,
                "name": tc.name,
                "arguments": json.dumps(
                    tc.arguments, separators=(",", ":"), ensure_ascii=False
                ),
            },
        )
        tool_result: ToolResultData = await self.ctx.tools.execute(
            tc, session=session, signal=signal
        )
        self.ctx.sessions.log_event(
            session,
            EventType.TOOL_RESULT,
            _tool_result_payload(tc, tool_result, turn=turn, step=step),
        )
        return tool_result

    # -- turns ------------------------------------------------------------------

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
        turn = self._next_turn(session)
        request_state = _RequestState(
            surface_generation=session.surface_content_generation
        )

        # 1. turn/start
        self.ctx.sessions.log_event(
            session,
            EventType.TURN_START,
            {"turn": turn},
        )
        self.ctx.emit("turn/start", session, turn_id)

        step_index = 0
        final_assistant_text = ""
        turn_end_reason: dict[str, Any] = {"kind": "completed"}
        max_tokens_hit = False

        try:
            while step_index < max_steps:
                if signal:
                    signal.throw_if_aborted()

                step_index += 1
                step = step_index

                # 3. step/start
                self.ctx.sessions.log_event(
                    session,
                    EventType.STEP_START,
                    {"turn": turn, "step": step},
                )
                self.ctx.emit("step/start", session, step_index)

                try:
                    request = await self._open_step(
                        session,
                        turn=turn,
                        step=step,
                        user_input=user_input,
                        model=model,
                        request_state=request_state,
                    )
                    if request is None:
                        break

                    # 7. Model request (no separate system_prompt: it is
                    # surface node 0 of the derived history).
                    response = await self.ctx.llm.call(
                        messages=request["messages"],
                        tools=request.get("tools"),
                        model=f"{request['provider']}/{request['model']}",
                    )
                    if response.finish_reason == "max_tokens":
                        max_tokens_hit = True

                    # 8. Log assistant response
                    content_blocks: list[dict[str, Any]] = []
                    if response.text:
                        content_blocks.append({"type": "text", "text": response.text})
                    content_blocks.extend(_tool_call_blocks(response.tool_calls))
                    assistant_event: SessionEvent | None = None
                    if content_blocks or _has_usage(response.usage):
                        message = create_assistant_message(
                            content=content_blocks,
                            provider=request["provider"],
                            model=request["model"],
                        )
                        stream = [
                            {
                                "type": "chunk",
                                "time": int(time.time() * 1000),
                                "chunk": {
                                    "text": response.text,
                                    "tool_calls": [
                                        tc.model_dump() for tc in response.tool_calls
                                    ],
                                    "finish_reason": response.finish_reason,
                                    "usage": response.usage.model_dump(),
                                },
                            }
                        ]
                        assistant_event = self._log_assistant_message(
                            session,
                            turn=turn,
                            step=step,
                            message=message,
                            stream=stream,
                            usage=response.usage,
                        )
                    final_assistant_text = response.text

                    # 9. Handle tool calls if any (read back from the log).
                    tool_calls = (
                        _tool_calls_from_message(assistant_event.payload["message"])
                        if assistant_event is not None
                        else []
                    )
                    if tool_calls and self.ctx.has("tools"):
                        for tc in tool_calls:
                            await self._run_tool_call(
                                session,
                                turn=turn,
                                step=step,
                                tc=tc,
                                signal=signal,
                            )
                    else:
                        # No tool calls; turn is finished
                        break
                finally:
                    self.ctx.sessions.log_event(
                        session,
                        EventType.STEP_END,
                        {"turn": turn, "step": step},
                    )
                    self.ctx.emit("step/end", session, step_index)
        except AbortError as exc:
            turn_end_reason = {"kind": "aborted", "reason": str(exc)}
            raise
        except Exception as exc:
            turn_end_reason = {
                "kind": "error",
                "error": {"message": str(exc), "code": type(exc).__name__},
            }
            raise
        finally:
            # 10. turn/end
            if turn_end_reason["kind"] == "completed" and max_tokens_hit:
                turn_end_reason = {"kind": "max-tokens"}
            self.ctx.emit("agent/turn-stopping", session, turn_id)
            self.ctx.sessions.log_event(
                session,
                EventType.TURN_END,
                {"turn": turn, "reason": turn_end_reason},
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
        turn = self._next_turn(session)
        request_state = _RequestState(
            surface_generation=session.surface_content_generation
        )

        self.ctx.sessions.log_event(
            session,
            EventType.TURN_START,
            {"turn": turn},
        )
        self.ctx.emit("turn/start", session, turn_id)

        step_index = 0
        final_assistant_text = ""
        turn_end_reason: dict[str, Any] = {"kind": "completed"}
        max_tokens_hit = False

        try:
            while step_index < max_steps:
                if signal:
                    signal.throw_if_aborted()

                step_index += 1
                step = step_index

                self.ctx.sessions.log_event(
                    session,
                    EventType.STEP_START,
                    {"turn": turn, "step": step},
                )
                self.ctx.emit("step/start", session, step_index)

                try:
                    request = await self._open_step(
                        session,
                        turn=turn,
                        step=step,
                        user_input=user_input,
                        model=model,
                        request_state=request_state,
                    )
                    if request is None:
                        break

                    accumulator = AssistantStreamAccumulator()
                    accumulated_text = ""
                    collected_tool_calls: list[ToolCallData] = []
                    accumulated_usage = UsageData()
                    finish_reason: str | None = None
                    try:
                        async for chunk in self.ctx.llm.stream(
                            messages=request["messages"],
                            tools=request.get("tools"),
                            model=f"{request['provider']}/{request['model']}",
                            signal=signal,
                            session_id=session.session_id,
                        ):
                            accumulator.push(chunk)
                            if chunk.text:
                                accumulated_text += chunk.text
                                yield {"type": "chunk", "text": chunk.text}
                            if chunk.tool_calls:
                                collected_tool_calls.extend(chunk.tool_calls)
                            if chunk.finish_reason is not None:
                                finish_reason = chunk.finish_reason
                            if _has_usage(chunk.usage):
                                accumulated_usage = chunk.usage
                                yield {
                                    "type": "usage",
                                    "usage": chunk.usage.model_dump(),
                                }
                    except Exception as exc:
                        # A failed or aborted attempt that reached
                        # settlement is preserved verbatim, then the
                        # original error propagates unchanged.
                        if isinstance(exc, AbortError) and accumulated_text:
                            message = create_assistant_message(
                                content=[{"type": "text", "text": accumulated_text}],
                                provider=request["provider"],
                                model=request["model"],
                            )
                            self._log_assistant_message(
                                session,
                                turn=turn,
                                step=step,
                                message=message,
                                stream=accumulator.snapshot(),
                                usage=accumulated_usage,
                                interrupted=True,
                            )
                            final_assistant_text = accumulated_text
                        elif accumulator:
                            self.ctx.sessions.log_event(
                                session,
                                EventType.ASSISTANT_ATTEMPT,
                                {
                                    "turn": turn,
                                    "step": step,
                                    "stream": accumulator.snapshot(),
                                },
                            )
                        raise
                    if finish_reason == "max_tokens":
                        max_tokens_hit = True

                    content_blocks: list[dict[str, Any]] = []
                    if accumulated_text:
                        content_blocks.append(
                            {"type": "text", "text": accumulated_text}
                        )
                    content_blocks.extend(_tool_call_blocks(collected_tool_calls))
                    assistant_event: SessionEvent | None = None
                    if content_blocks or _has_usage(accumulated_usage):
                        message = create_assistant_message(
                            content=content_blocks,
                            provider=request["provider"],
                            model=request["model"],
                        )
                        assistant_event = self._log_assistant_message(
                            session,
                            turn=turn,
                            step=step,
                            message=message,
                            stream=accumulator.snapshot(),
                            usage=accumulated_usage,
                        )
                    final_assistant_text = accumulated_text

                    tool_calls = (
                        _tool_calls_from_message(assistant_event.payload["message"])
                        if assistant_event is not None
                        else []
                    )
                    if tool_calls and self.ctx.has("tools"):
                        for tc in tool_calls:
                            if signal:
                                signal.throw_if_aborted()

                            yield {
                                "type": "tool_start",
                                "name": tc.name,
                                "call_id": tc.call_id,
                                "arguments": tc.arguments,
                            }

                            tool_result = await self._run_tool_call(
                                session,
                                turn=turn,
                                step=step,
                                tc=tc,
                                signal=signal,
                            )

                            yield {
                                "type": "tool_end",
                                "name": tc.name,
                                "call_id": tc.call_id,
                                "result": tool_result.content,
                                "is_error": tool_result.is_error,
                            }
                    else:
                        break
                finally:
                    self.ctx.sessions.log_event(
                        session,
                        EventType.STEP_END,
                        {"turn": turn, "step": step},
                    )
                    self.ctx.emit("step/end", session, step_index)
        except AbortError as exc:
            turn_end_reason = {"kind": "aborted", "reason": str(exc)}
            raise
        except Exception as exc:
            turn_end_reason = {
                "kind": "error",
                "error": {"message": str(exc), "code": type(exc).__name__},
            }
            raise
        finally:
            if turn_end_reason["kind"] == "completed" and max_tokens_hit:
                turn_end_reason = {"kind": "max-tokens"}
            self.ctx.emit("agent/turn-stopping", session, turn_id)
            self.ctx.sessions.log_event(
                session,
                EventType.TURN_END,
                {"turn": turn, "reason": turn_end_reason},
            )
            self.ctx.emit("turn/end", session, turn_id)

        yield {"type": "turn_end", "text": final_assistant_text}
