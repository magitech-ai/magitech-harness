"""Default Agent driver over queued turns and step-boundary input.

Every request is derived directly from the session log.
"""

from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from anima import Context

from magitech_harness.agent.assistant_stream import AssistantStreamAttempt
from magitech_harness.agent.cancellation import AbortController, AbortSignal
from magitech_harness.agent.constants import DEFAULT_MAX_PARALLEL_TOOL_CALLS
from magitech_harness.agent.contracts import (
    AgentOptions,
    AgentStatus,
    CancelOptions,
    InboxTarget,
    ModelSelection,
)
from magitech_harness.agent.dispatch import (
    AgentEventDispatch,
    agent_events,
    assemble_context_for,
)
from magitech_harness.agent.inbox import AgentInbox, InboxState
from magitech_harness.agent.model_selection import (
    AgentModel,
    ModelSelectionProjectionState,
)
from magitech_harness.agent.runtime_context import RuntimeContextProjection
from magitech_harness.agent.system_prompt import SystemPromptProjection
from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.call_config import mark_agent_loop_request
from magitech_harness.llm.error import error_chain
from magitech_harness.llm.message import (
    create_assistant_message,
    create_developer_message,
)
from magitech_harness.llm.service import PreparedLlmCall
from magitech_harness.llm.types import GenerateOptions, LlmFailure, ToolSchema
from magitech_harness.llm.values import deep_freeze, is_frozen
from magitech_harness.scope import Scope, create_scope
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.repair import ToolCallRecovery
from magitech_harness.session.request_header import (
    canonical_header,
    header_equals,
    request_proposal,
)
from magitech_harness.session.store import Session
from magitech_harness.session.tool_history import tool_name
from magitech_harness.system_prompt.service import (
    PromptAssembly,
    join_context_sections,
    render_context_sections,
    render_prompt,
)
from magitech_harness.tools.contracts import (
    ResolvedToolSet,
    ToolActivationError,
    ToolBatchExecutionError,
    ToolBatchInput,
)

StepEndKind = Literal["completed", "max-tokens"]


@dataclass
class _IdlePhase:
    last_turn: int
    kind: Literal["idle"] = "idle"


@dataclass
class _MaintenancePhase:
    abort: AbortController
    last_turn: int
    wake_requested: bool = False
    kind: Literal["maintenance"] = "maintenance"


@dataclass
class _RunningPhase:
    abort: AbortController
    turn: int
    step: int
    wake_requested: bool = False
    kind: Literal["running"] = "running"


_Phase = _IdlePhase | _MaintenancePhase | _RunningPhase


def _aborted_cancel_cause(signal: AbortSignal) -> dict[str, Any] | None:
    """Copy durable ``turn/end`` abort cause fields without host stack metadata."""
    if not signal.aborted:
        return None
    cause = signal.reason
    if isinstance(cause, Mapping):
        kind = cause.get("kind")
        if kind in ("user", "parent", "disposed"):
            return {"kind": kind}
        if kind == "hook":
            return {"kind": "hook", "reason": cause.get("reason")}
    return {"kind": "user"}


def _system_nodes_of(session: Session) -> list[dict[str, Any]]:
    """Collect surviving ``system/message`` surface nodes with comparable text."""
    nodes: list[dict[str, Any]] = []
    for seq in session.surface_nodes():
        event = session.event_at(seq)
        if event is None or event.type != "system/message":
            continue
        message = event.payload.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        text: str | None
        if isinstance(content, list):
            if len(content) == 0:
                text = ""
            elif (
                len(content) == 1
                and isinstance(content[0], Mapping)
                and content[0].get("type") == "text"
                and isinstance(content[0].get("text"), str)
            ):
                text = content[0]["text"]
            else:
                text = None
        else:
            text = None
        nodes.append({"seq": seq, "text": text})
    return nodes


def _llm_error_from_failure(failure: Mapping[str, Any]) -> LlmError:
    """Construct an LlmError carrying every structured field from *failure*."""
    message = str(failure.get("message") or "Model request failed")
    code = str(failure.get("code") or "UNKNOWN")
    kwargs: dict[str, Any] = {}
    if "status" in failure and failure["status"] is not None:
        kwargs["status"] = failure["status"]
    if (
        "providerRetryAfterMs" in failure
        and failure["providerRetryAfterMs"] is not None
    ):
        kwargs["providerRetryAfterMs"] = failure["providerRetryAfterMs"]
    if "requestId" in failure and failure["requestId"] is not None:
        kwargs["requestId"] = failure["requestId"]
    if "offloadImages" in failure and failure["offloadImages"] is not None:
        kwargs["offloadImages"] = failure["offloadImages"]
    return LlmError(message, code, **kwargs)


def _step_failure_reason(error: BaseException) -> LlmFailure | dict[str, Any]:
    if isinstance(error, LlmError):
        return dict(error.failure)
    if isinstance(error, ToolActivationError):
        return {"message": error_chain(error), "code": error.code}
    return {"message": error_chain(error), "code": "UNKNOWN"}


class ReactLoopAgent:
    """Drives one session through turn and step boundaries."""

    def __init__(
        self,
        loop_ctx: Context,
        agent_id: str,
        options: AgentOptions | Mapping[str, Any],
        session: Session,
        projections: Mapping[str, Callable[[], Any]],
        last_turn: int,
    ) -> None:
        self._loop_ctx = loop_ctx
        self.id = agent_id
        self.options: AgentOptions = cast(AgentOptions, dict(options))
        self.session = session
        self._phase: _Phase = _IdlePhase(last_turn=last_turn)
        self._activity_done: asyncio.Future[None] | None = None

        self._dispatch: AgentEventDispatch = agent_events(loop_ctx, self)
        self.scope: Scope = create_scope(loop_ctx, self)
        self.ctx: Context = self.scope.ctx

        model_reader = cast(
            Callable[[], ModelSelectionProjectionState],
            projections["modelSelection"],
        )
        inbox_reader = cast(Callable[[], InboxState], projections["inbox"])
        self.model = AgentModel(model_reader, session, self.options)
        self.inbox = AgentInbox(
            inbox_reader,
            session,
            cast(Any, self._dispatch),
        )

        self._request_header_logged = False
        self._request_surface_generation = session.surface_content_generation
        self._runtime_context = RuntimeContextProjection(self.ctx, session)
        self._system_prompt = SystemPromptProjection(
            lambda: _system_nodes_of(self.session)
        )
        self._assistant_stream_revision = 0
        self._assistant_attempt_counter = 0
        self._driver_tasks: set[asyncio.Task[None]] = set()

    @property
    def status(self) -> AgentStatus:
        """Return ``'idle'`` when idle or running maintenance, else ``'running'``."""
        return "idle" if self._phase.kind in ("idle", "maintenance") else "running"

    def _set_phase(self, next_phase: _Phase) -> None:
        prev_status = self.status
        self._phase = next_phase
        next_status = self.status
        if next_status != prev_status:
            self._dispatch.emit("agent/status", {"status": next_status})

    def send(self, message: dict[str, Any], target: InboxTarget, wakeup: bool) -> None:
        """Insert *message* into *target*, optionally waking the driver."""
        waking_after_abort = (
            wakeup and self._phase.kind != "idle" and self._phase.abort.signal.aborted
        )
        resolved_target: InboxTarget = "next-turn" if waking_after_abort else target
        self.inbox.splice(resolved_target, math.inf, 0, [message])
        if wakeup:
            self._wake_driver(waking_after_abort)

    def followup(self, input_message: dict[str, Any]) -> None:
        """Append one prompt to ``next-turn`` and wake the loop if idle."""
        self.send(input_message, "next-turn", True)

    def steer(self, input_message: dict[str, Any]) -> None:
        """Append one steering message to ``next-step`` and wake if idle."""
        self.send(input_message, "next-step", True)

    def inject(self, input_message: dict[str, Any]) -> None:
        """Append one message to ``next-step`` without waking an idle loop."""
        self.send(input_message, "next-step", False)

    def cancel(
        self, cause: Mapping[str, Any], options: CancelOptions | None = None
    ) -> None:
        """Abort the active phase and optionally clear pending inbox work."""
        keep_inbox = bool(options.get("keepInbox")) if options is not None else False
        if not keep_inbox:
            self.inbox.clear()
            if self._phase.kind != "idle":
                self._phase.wake_requested = False
        if self._phase.kind != "idle":
            self._phase.abort.abort(dict(cause))

    async def run_maintenance[T](
        self, task: Callable[[AbortSignal], Awaitable[T]]
    ) -> T:
        """Run non-model maintenance work exclusive of active turn execution."""
        if self._phase.kind != "idle":
            raise RuntimeError(f'agent "{self.id}" already has active work')
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        maintenance = _MaintenancePhase(
            abort=AbortController(),
            last_turn=self._phase.last_turn,
        )
        self._set_phase(maintenance)
        self._activity_done = done
        try:
            return await task(maintenance.abort.signal)
        finally:
            self._set_phase(_IdlePhase(last_turn=maintenance.last_turn))
            cause = _aborted_cancel_cause(maintenance.abort.signal)
            if (
                (cause is None or cause["kind"] != "disposed")
                and maintenance.wake_requested
                and self.inbox.has_pending
            ):
                self._wake_driver()
            if not done.done():
                done.set_result(None)

    runMaintenance = run_maintenance

    def _wake_driver(self, wake_after_abort: bool = False) -> None:
        if self._phase.kind != "idle":
            reason = _aborted_cancel_cause(self._phase.abort.signal)
            if (reason is None or reason["kind"] != "disposed") and (
                self._phase.kind == "maintenance" or wake_after_abort
            ):
                self._phase.wake_requested = True
            return

        loop = asyncio.get_running_loop()
        driver_done: asyncio.Future[None] = loop.create_future()
        self._activity_done = driver_done
        self._set_phase(
            _RunningPhase(
                abort=AbortController(),
                turn=self._phase.last_turn,
                step=0,
            )
        )

        async def _run() -> None:
            try:
                await self._kick()
            except BaseException as exc:  # noqa: BLE001 - surface driver failure on future
                driver_done.set_exception(exc)
                return
            driver_done.set_result(None)

        coro = self._loop_ctx.agents.with_initiator(self, _run)
        task = asyncio.ensure_future(coro)
        self._driver_tasks.add(task)
        task.add_done_callback(self._driver_tasks.discard)

    async def when_idle(self) -> None:
        """Resolve once any running turn or maintenance work settles."""
        while self._activity_done is not None:
            current = self._activity_done
            await current
            if current is self._activity_done:
                break

    whenIdle = when_idle

    def _throw_error(self, error: BaseException) -> None:
        turn = (
            self._phase.turn
            if isinstance(self._phase, _RunningPhase)
            else self._phase.last_turn
        )
        step = self._phase.step if isinstance(self._phase, _RunningPhase) else 0
        self._dispatch.emit("agent/error", {"turn": turn, "step": step, "error": error})
        raise error

    async def _kick(self) -> None:
        try:
            while await self._turn():
                pass
        except Exception:  # noqa: BLE001,S110 - driver faults already emitted via _throw_error
            pass
        finally:
            assert isinstance(self._phase, _RunningPhase)
            turn = self._phase.turn
            wake_requested = self._phase.wake_requested
            self._set_phase(_IdlePhase(last_turn=turn))
            if wake_requested and self.inbox.has_pending:
                self._wake_driver()

    async def _pre_step(
        self, target: InboxTarget, position: dict[str, int]
    ) -> dict[str, Any]:
        assert isinstance(self._phase, _RunningPhase)
        signal = self._phase.abort.signal
        claimed = self.inbox.claim(target, position["turn"])
        selection = self.model.current()
        if selection is None:
            raise RuntimeError(
                f'agent "{self.id}" has no provider/model: set '
                "AgentOptions.provider and AgentOptions.model or record a "
                "model selection"
            )
        tool_set = await self._loop_ctx.tools.resolve(
            {"agent": self, "modelSelection": selection, "signal": signal}
        )
        try:
            assembly = await self._loop_ctx.system_prompt.assemble(
                {
                    **assemble_context_for(self, signal),
                    "modelSelection": selection,
                    "toolSet": tool_set,
                }
            )
            signal.throw_if_aborted()

            def _default_pre_step(_payload: Mapping[str, Any]) -> dict[str, Any]:
                sections = render_context_sections(assembly)
                context_msg = self._runtime_context.project(
                    join_context_sections(sections), sections
                )
                return {
                    "kind": "enter",
                    "messages": (
                        claimed if context_msg is None else [*claimed, context_msg]
                    ),
                }

            decision = await self._dispatch.waterfall(
                "agent/pre-step",
                {
                    "messages": claimed,
                    "turn": position["turn"],
                    "step": position["step"],
                    "signal": signal,
                    "modelSelection": selection,
                    "agent": self,
                    "session": self.session,
                },
                default_handler=_default_pre_step,
            )
            signal.throw_if_aborted()
            if decision.get("kind") == "reject":
                tool_set.dispose()
                return dict(decision)
            return {
                **decision,
                "assembly": assembly,
                "selection": selection,
                "toolSet": tool_set,
            }
        except Exception:
            tool_set.dispose()
            raise

    def _tools_changed(self, tools: Sequence[ToolSchema]) -> bool:
        baseline = self.session.request_header()
        if baseline is None:
            return False
        candidate = canonical_header({**baseline, "tools": list(tools)})
        return not header_equals(baseline, candidate)

    def _record_turn_end(self, turn: int, reason: Mapping[str, Any]) -> None:
        try:
            self.session.append("turn/end", {"turn": turn, "reason": dict(reason)})
        except Exception as end_err:  # noqa: BLE001 - emit agent/error before re-raising
            self._throw_error(end_err)

    async def _turn(self) -> bool:
        if not isinstance(self._phase, _RunningPhase):
            self._throw_error(
                RuntimeError(f'agent "{self.id}": turn without driver reservation')
            )
        assert isinstance(self._phase, _RunningPhase)
        phase = self._phase
        signal = phase.abort.signal
        signal.throw_if_aborted()
        turn = phase.turn + 1
        try:
            self.session.append("turn/start", {"turn": turn})
        except Exception as err:  # noqa: BLE001 - emit agent/error before re-raising
            self._throw_error(err)
        phase.turn = turn
        turn_ends: dict[str, Any] | None = None
        stop_driver = False
        target: InboxTarget = "next-turn"
        try:
            while True:
                signal.throw_if_aborted()
                step = phase.step + 1
                decision = await self._pre_step(target, {"turn": turn, "step": step})
                if decision.get("kind") == "reject":
                    turn_ends = {"kind": "blocked"}
                    stop_driver = True
                    break
                tool_set = cast(ResolvedToolSet, decision["toolSet"])
                try:
                    messages = decision.get("messages") or []
                    if turn_ends is not None and len(messages) == 0:
                        break
                    if phase.step == 0 and len(messages) == 0:
                        turn_ends = {"kind": "completed"}
                        stop_driver = True
                        break
                    signal.throw_if_aborted()
                    self.session.append("step/start", {"turn": turn, "step": step})
                    phase.step = step
                    recovery = ToolCallRecovery()

                    def _observe_recovery(
                        subj: Any,
                        event: SessionEvent,
                        rec: ToolCallRecovery = recovery,
                    ) -> None:
                        if subj is self.session:
                            rec.observe(event)

                    stop_recovery = self.ctx.on("session/event", _observe_recovery)
                    try:
                        step_end = await self._step(decision)
                        if turn_ends is None or turn_ends.get("kind") != "max-tokens":
                            turn_ends = step_end
                    except Exception as step_err:
                        try:
                            for rec_ev in recovery.results():
                                self.session.append(
                                    "tool/result",
                                    copy.deepcopy(rec_ev.payload),
                                    surface_op="append",
                                    source_event_seqs=rec_ev.source_event_seqs,
                                )
                        except Exception as rec_err:  # noqa: BLE001 - bundle step + recovery errors
                            raise ExceptionGroup(
                                "Step failed and its pending tool results could not "
                                "be recorded",
                                [step_err, rec_err],
                            ) from step_err
                        raise
                    finally:
                        stop_recovery()
                        self.session.append("step/end", {"turn": turn, "step": step})
                finally:
                    tool_set.dispose()
                signal.throw_if_aborted()
                if turn_ends is not None and len(self.inbox.next_step) == 0:
                    await self._dispatch.serial(
                        "agent/turn-stopping", {"turn": turn, "signal": signal}
                    )
                    signal.throw_if_aborted()
                if turn_ends is not None and len(self.inbox.next_step) == 0:
                    break
                target = "next-step"
        except Exception as err:  # noqa: BLE001 - record turn/end and re-emit fault
            cause = _aborted_cancel_cause(signal)
            if signal.aborted and cause is not None:
                try:
                    raise
                finally:
                    self._record_turn_end(turn, {"kind": "aborted", "reason": cause})
            try:
                self._throw_error(err)
            finally:
                self._record_turn_end(
                    turn, {"kind": "error", "error": _step_failure_reason(err)}
                )
        assert turn_ends is not None
        self._record_turn_end(turn, turn_ends)

        if stop_driver or not self.inbox.has_pending:
            return False
        phase.abort = AbortController()
        phase.wake_requested = False
        phase.step = 0
        return True

    def _max_parallel_tool_calls(self) -> int:
        agent_loop = self._loop_ctx.get("agentLoop", None, strict=False)
        if agent_loop is not None and hasattr(agent_loop, "max_parallel_tool_calls"):
            val = agent_loop.max_parallel_tool_calls
            return int(val() if callable(val) else val)
        return DEFAULT_MAX_PARALLEL_TOOL_CALLS

    async def _step(self, decision: Mapping[str, Any]) -> dict[str, Any] | None:
        assert isinstance(self._phase, _RunningPhase)
        turn = self._phase.turn
        step = self._phase.step
        signal = self._phase.abort.signal
        signal.throw_if_aborted()

        assembly = cast(PromptAssembly, decision["assembly"])
        selection = cast(ModelSelection, decision["selection"])
        tool_set = cast(ResolvedToolSet, decision["toolSet"])
        rendered_prompt = render_prompt(assembly)
        first_attempt = True

        while True:
            config, prepared_call = await self._prepare_request(
                turn, step, selection, signal
            )
            current_context = self._loop_ctx.system_prompt.refresh_context(
                assembly,
                {**assemble_context_for(self, signal), "modelSelection": selection},
            )
            sections = render_context_sections(current_context)
            context_msg = self._runtime_context.project(
                join_context_sections(sections), sections
            )
            signal.throw_if_aborted()

            starts_request_series = first_attempt and (
                decision.get("startsRequestSeries") is True
            )
            tool_update = (
                prepared_call.tool_update if prepared_call is not None else None
            )
            sp_update = (
                prepared_call.system_prompt_update
                if prepared_call is not None
                else None
            )
            step_tools = cast(Sequence[ToolSchema], assembly["tools"])
            commits = self._system_prompt.project(
                rendered_prompt,
                in_history=sp_update == "in-history",
                starts_series=(
                    starts_request_series
                    or self._request_surface_generation
                    != self.session.surface_content_generation
                    or (tool_update is None and self._tools_changed(step_tools))
                ),
            )
            for commit in commits:
                self.session.append(
                    "system/message",
                    {"turn": turn, "step": step, "message": commit.message},
                    surface_op=commit.intent["surface_op"],
                    source_event_seqs=commit.intent.get("source_event_seqs"),
                )

            context_admitted = False
            if first_attempt:
                for msg in decision.get("messages") or []:
                    src = msg.get("source") if isinstance(msg, Mapping) else None
                    if (
                        isinstance(src, Mapping)
                        and src.get("kind") == "runtime-context"
                    ):
                        if context_msg is not None and not context_admitted:
                            self.session.append(
                                "user/message",
                                dict(context_msg),
                                surface_op="append",
                            )
                            context_admitted = True
                    else:
                        self.session.append(
                            "user/message", dict(msg), surface_op="append"
                        )
            if context_msg is not None and not context_admitted:
                self.session.append(
                    "user/message", dict(context_msg), surface_op="append"
                )
            first_attempt = False

            request = self._build_request(
                config,
                prepared_call,
                step_tools,
                {"turn": turn, "step": step},
                starts_request_series,
                signal,
            )
            self._assistant_attempt_counter += 1
            attempt_num = self._assistant_attempt_counter

            def _next_rev() -> int:
                self._assistant_stream_revision += 1
                return self._assistant_stream_revision

            live = AssistantStreamAttempt(
                self.session.id,
                attempt_num,
                _next_rev,
                turn,
                step,
                lambda frame: self._dispatch.emit(
                    "agent/assistant-stream", {"frame": frame}
                ),
            )

            def _append_attempt_seq(
                cur_live: AssistantStreamAttempt = live,
            ) -> int:
                return self.session.append(
                    "assistant/attempt",
                    {
                        "turn": turn,
                        "step": step,
                        "stream": cur_live.stream,
                    },
                ).seq

            started = False
            try:
                stream = (
                    prepared_call.stream(request)
                    if prepared_call is not None
                    else self._loop_ctx.llm.stream(request)
                )
                signal.throw_if_aborted()
                live.start()
                started = True
                async for chunk in stream:
                    signal.throw_if_aborted()
                    live.push(chunk)
                signal.throw_if_aborted()
            except Exception as stream_err:
                if not started:
                    raise

                try:
                    if signal.aborted:
                        content = live.interrupted_blocks()
                        if len(content) > 0:
                            source: dict[str, Any] = {
                                "provider": request["provider"],
                                "model": request["model"],
                            }
                            if live.replay_state is not None:
                                source["replayState"] = live.replay_state
                            payload: dict[str, Any] = {
                                "turn": turn,
                                "step": step,
                                "message": create_assistant_message(
                                    content=content,
                                    source=source,
                                ),
                                "interrupted": True,
                                "stream": live.stream,
                            }
                            if live.usage is not None:
                                payload["usage"] = live.usage

                            def _append_interrupted_msg(
                                p: dict[str, Any] = payload,
                            ) -> int:
                                return self.session.append(
                                    "assistant/message",
                                    p,
                                    surface_op="append",
                                ).seq

                            live.settle(
                                "assistant/message",
                                _append_interrupted_msg,
                            )
                        else:
                            live.settle("assistant/attempt", _append_attempt_seq)
                    else:
                        live.settle("assistant/attempt", _append_attempt_seq)
                except Exception as settle_err:  # noqa: BLE001 - bundle stream + settlement errors
                    raise ExceptionGroup(
                        "Assistant stream failed and its durable settlement was "
                        "rejected",
                        [stream_err, settle_err],
                    ) from stream_err
                raise

            try:
                finish = live.finish
                if finish["kind"] in ("error", "aborted"):
                    live.settle("assistant/attempt", _append_attempt_seq)
                    failure = cast(dict[str, Any], finish["failure"])
                    retry_policy = (
                        prepared_call.retry_policy
                        if prepared_call is not None
                        else None
                    )
                    action = await self._dispatch.waterfall(
                        "agent/request-error",
                        {
                            "turn": turn,
                            "step": step,
                            "provider": request["provider"],
                            "failure": failure,
                            "retryPolicy": retry_policy,
                            "signal": signal,
                        },
                        default_handler=lambda _p: None,
                    )
                    signal.throw_if_aborted()
                    if not (
                        isinstance(action, Mapping) and action.get("kind") == "retry"
                    ):
                        raise _llm_error_from_failure(failure)
                    continue

                msg_source: dict[str, Any] = {
                    "provider": request["provider"],
                    "model": request["model"],
                }
                if live.replay_state is not None:
                    msg_source["replayState"] = live.replay_state
                message = create_assistant_message(
                    content=live.blocks(),
                    source=msg_source,
                )
                msg_payload: dict[str, Any] = {
                    "turn": turn,
                    "step": step,
                    "message": message,
                    "stream": live.stream,
                }
                if live.usage is not None:
                    msg_payload["usage"] = live.usage

                def _append_msg(mp: dict[str, Any] = msg_payload) -> int:
                    return self.session.append(
                        "assistant/message",
                        mp,
                        surface_op="append",
                    ).seq

                live.settle("assistant/message", _append_msg)
                if finish["kind"] == "max-tokens":
                    return {"kind": "max-tokens"}

                tool_calls = [
                    block
                    for block in message["content"]
                    if isinstance(block, Mapping) and block.get("type") == "tool-call"
                ]
                if not tool_calls:
                    return {"kind": "completed"}

                batch_input: ToolBatchInput = {
                    "calls": cast(Any, tool_calls),
                    "signal": signal,
                    "agent": self,
                    "log": {"turn": turn, "step": step},
                    "maxParallelToolCalls": self._max_parallel_tool_calls(),
                    "toolSet": tool_set,
                }
                try:
                    batch_result = await self._loop_ctx.tools.execute(batch_input)
                except ToolBatchExecutionError as batch_err:
                    for extra in batch_err.partial["additionalContexts"]:
                        self.inbox.splice(
                            "next-step", len(self.inbox.next_step), 0, [dict(extra)]
                        )
                    cause = batch_err.cause
                    if isinstance(cause, BaseException):
                        raise cause from batch_err
                    raise RuntimeError(str(cause)) from batch_err

                for extra in batch_result["additionalContexts"]:
                    self.inbox.splice(
                        "next-step", len(self.inbox.next_step), 0, [dict(extra)]
                    )
                return {"kind": "completed"} if batch_result["concluded"] else None
            except Exception:
                if not live.ended:
                    live.abandon()
                raise

    async def _prepare_request(
        self,
        turn: int,
        step: int,
        selection: ModelSelection,
        signal: AbortSignal,
    ) -> tuple[dict[str, Any], PreparedLlmCall | None]:
        max_tokens = self.options.get("maxTokens")
        if self._request_header_logged:
            header = self.session.request_header()
            assert header is not None
            persisted: dict[str, Any] = request_proposal(header)
        elif max_tokens is not None:
            persisted = {"maxTokens": max_tokens}
        else:
            persisted = {}

        controls = {
            k: v
            for k, v in persisted.items()
            if k not in ("provider", "model", "reasoningEffort")
        }
        seed_config = deep_freeze(copy.deepcopy({**controls, **selection}))
        proposed_config = await self._dispatch.waterfall(
            "agent/request",
            {"turn": turn, "step": step, "signal": signal},
            default_handler=lambda _p: seed_config,
        )
        signal.throw_if_aborted()

        if (
            proposed_config.get("provider") != selection["provider"]
            or proposed_config.get("model") != selection["model"]
        ):
            raise ValueError(
                f'agent "{self.id}": agent/request proposed '
                f'"{proposed_config.get("provider")}/{proposed_config.get("model")}" '
                "after step preparation fixed "
                f'"{selection["provider"]}/{selection["model"]}"; use '
                "agent.model.select() before step preparation"
            )
        if proposed_config.get("reasoningEffort") != selection.get("reasoningEffort"):
            proposed_effort = (
                proposed_config.get("reasoningEffort") or "adapter default"
            )
            fixed_effort = selection.get("reasoningEffort") or "adapter default"
            raise ValueError(
                f'agent "{self.id}": agent/request proposed reasoning effort '
                f'"{proposed_effort}" after step preparation fixed '
                f'"{fixed_effort}"; use agent.model.select() before step '
                "preparation"
            )

        prepared_call: PreparedLlmCall | None = None
        try:
            prepared_call = await self._loop_ctx.llm.prepare_call(
                cast(Any, proposed_config), signal
            )
            config = dict(prepared_call.config)
        except LlmError as err:
            if err.code != "NO_ADAPTER":
                raise
            config = dict(proposed_config)
        signal.throw_if_aborted()
        return config, prepared_call

    def _build_request(
        self,
        config: Mapping[str, Any],
        prepared_call: PreparedLlmCall | None,
        tools: Sequence[ToolSchema],
        position: dict[str, int],
        starts_request_series: bool,
        signal: AbortSignal,
    ) -> GenerateOptions:
        surface_generation = self.session.surface_content_generation
        raw_header: dict[str, Any] = {"config": dict(config)}
        if prepared_call is not None:
            raw_header["adapterDefaults"] = prepared_call.adapter_defaults
        if len(tools) > 0:
            raw_header["tools"] = list(tools)
        header = canonical_header(raw_header)
        baseline = self.session.request_header()
        starts_series = (
            starts_request_series
            or self._request_surface_generation != surface_generation
        )

        header_seq: int | None = None
        if not self._request_header_logged:
            entry_payload: dict[str, Any] = {
                "header": header,
                "reason": "initial" if baseline is None else "resume",
            }
            if starts_series:
                entry_payload["startsSeries"] = True
            header_seq = self.session.append("request/header", entry_payload).seq
            self._request_header_logged = True
        elif baseline is None or not header_equals(baseline, header):
            change_payload: dict[str, Any] = {"header": header, "reason": "change"}
            if starts_series:
                change_payload["startsSeries"] = True
            header_seq = self.session.append("request/header", change_payload).seq
        elif starts_series:
            self.session.append(
                "request/header", {"header": header, "reason": "series"}
            )

        if baseline is not None and header_seq is not None:
            prev_tools = baseline.get("tools") or []
            prev_names = {tool_name(tool) for tool in prev_tools}
            curr_names = {tool["name"] for tool in tools}
            additions = [
                {"type": "tool-addition", "toolName": tool["name"]}
                for tool in tools
                if tool["name"] not in prev_names
            ]
            removals = [
                {"type": "tool-removal", "toolName": tool_name(tool)}
                for tool in prev_tools
                if tool_name(tool) not in curr_names
            ]
            if additions or removals:
                dev_payload: dict[str, Any] = {
                    **position,
                    "message": create_developer_message(
                        source={"kind": "tool-registry"},
                        content=[*additions, *removals],
                    ),
                }
                if additions:
                    dev_payload["headerSeq"] = header_seq
                self.session.append(
                    "developer/message", dev_payload, surface_op="append"
                )

        self._request_surface_generation = surface_generation

        context_window = (
            prepared_call.context.get("contextWindow")
            if prepared_call is not None and prepared_call.context is not None
            else None
        )
        sp_update = (
            prepared_call.system_prompt_update if prepared_call is not None else None
        )
        req_context: dict[str, Any] = {
            "provider": config["provider"],
            "model": config["model"],
        }
        if context_window is not None:
            req_context["contextWindow"] = context_window
        if sp_update is not None:
            req_context["systemPromptUpdate"] = sp_update

        prev_ctx = self.session.request_context()
        if (
            prev_ctx is None
            or prev_ctx.get("provider") != req_context["provider"]
            or prev_ctx.get("model") != req_context["model"]
            or prev_ctx.get("contextWindow") != req_context.get("contextWindow")
            or prev_ctx.get("systemPromptUpdate")
            != req_context.get("systemPromptUpdate")
        ):
            self.session.append("request/context", req_context)
        signal.throw_if_aborted()

        frozen_header = deep_freeze(header)
        raw_messages = self.session.derive_messages()
        frozen_messages = [
            msg if is_frozen(msg) else deep_freeze(msg) for msg in raw_messages
        ]

        request_dict: dict[str, Any] = {
            **dict(frozen_header["config"]),
            "messages": tuple(frozen_messages),
            "toolHistory": self.session.tool_history(),
            "sessionId": self.session.id,
            "signal": signal,
        }
        if "tools" in frozen_header:
            request_dict["tools"] = frozen_header["tools"]
        return cast(GenerateOptions, mark_agent_loop_request(deep_freeze(request_dict)))


__all__ = ["ReactLoopAgent"]
