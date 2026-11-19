"""Tool execution pipeline and bounded parallel/exclusive batch scheduler."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable
from weakref import WeakKeyDictionary

from anima import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.error import is_harness_error
from magitech_harness.llm.message import create_tool_result_message
from magitech_harness.llm.values import (
    FrozenDict,
    assert_never,
    deep_freeze,
    is_json_value,
    snapshot_json_value,
)
from magitech_harness.scope import emit_scoped, scope_target, waterfall_scoped
from magitech_harness.tools.contracts import (
    TOOL_ABORTED,
    TOOL_ABORTED_BEFORE_DISPATCH,
    ToolBatchExecutionError,
    ToolBatchInput,
    ToolBatchResult,
    ToolCallResult,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecution,
    ToolExecutionFailure,
    ToolExecutionInput,
    ToolExecutionMode,
    ToolExecutionResult,
    ToolExecutionSuccess,
    ToolFailure,
    ToolNotFoundError,
    ToolOutputError,
    ToolRunContext,
)
from magitech_harness.tools.json_schema import validate_json_schema_value

logger = logging.getLogger("harness.tools.pipeline")

TOOLS_PRE_EXECUTE = "tools/pre-execute"
TOOLS_EXECUTE = "tools/execute"
TOOLS_POST_EXECUTE = "tools/post-execute"
TOOLS_RESULT = "tools/result"
TOOLS_CHANGE = "tools/change"

_MISSING = object()


class _ExecutionToken:
    """Opaque weak-referenceable identity token minted per prepared execution."""


@runtime_checkable
class ToolExecutionSource(Protocol):
    """Optional executor-local tool lookup and direct-call rejection hook."""

    def get(
        self, name: str, scope: Any = None, tool_set: Any = None
    ) -> ToolDefinition | None:
        """Look up a tool visible in *scope* / *tool_set*."""
        ...

    def reject(self, name: str, nested: bool) -> Any:
        """Return a rejection reason/exception if dispatch must be refused, else None."""
        ...


@dataclass(frozen=True)
class _AskResolution:
    decision: dict[str, Any]
    approval_cancelled: bool


@dataclass
class _ExecutionState:
    caller_signal: AbortSignal
    deferred_contexts: list[Mapping[str, Any]] = field(default_factory=list)
    conclude_turn: bool = False
    body_invoked: bool = False
    content_finalizer: (
        Callable[[ToolExecution, ToolExecutionResult], Sequence[Any] | None] | None
    ) = None
    content_projector: (
        Callable[[ToolExecution, ToolExecutionResult], Sequence[Any] | None] | None
    ) = None
    canonical_results: dict[int, Any] = field(default_factory=dict)

    def mark_canonical(self, result: ToolExecutionResult) -> ToolExecutionResult:
        self.canonical_results[id(result)] = result
        return result

    def is_canonical(self, result: Any) -> bool:
        return self.canonical_results.get(id(result)) is result


def fuse_tool_signals(
    *signals: AbortSignal | None,
) -> tuple[AbortSignal, Callable[[], None]]:
    """Combine zero or more AbortSignals into one derived AbortSignal."""
    fused = AbortSignal()
    disposers: list[Callable[[], None]] = []

    def _dispose() -> None:
        for dispose in disposers:
            dispose()
        disposers.clear()

    for sig in signals:
        if sig is None:
            continue
        if sig.is_aborted:
            fused.abort(sig.reason)
            _dispose()
            return fused, lambda: None

        def _make_handler(source: AbortSignal) -> Callable[[], None]:
            def _on_abort() -> None:
                fused.abort(source.reason)
                _dispose()

            return _on_abort

        disposers.append(sig.on_abort(_make_handler(sig)))

    return fused, _dispose


def tool_error_message(error: Any) -> str:
    """Safely render any thrown error value without crashing on hostile objects."""
    try:
        if isinstance(error, BaseException):
            text = str(error)
            return text if text else type(error).__name__
        if isinstance(error, Mapping):
            msg = error.get("message")
            if isinstance(msg, str):
                return msg
        else:
            msg = getattr(error, "message", None)
            if isinstance(msg, str):
                return msg
        return str(error)
    except Exception:  # noqa: BLE001 - hostile thrown values must never escape
        return "<unprintable thrown value>"


def _error_info(error: Any) -> ToolErrorInfo | None:
    try:
        if is_harness_error(error):
            return {"name": type(error).__name__, "code": error.code}
    except Exception:  # noqa: BLE001 - hostile error checks must not escape
        return None
    return None


def _is_unclassified_cancellation(error: Any, signal: Any) -> bool:
    """True when a thrown error is an unlabeled cancellation.

    A tool body (or wrapper) may throw a plain ``AbortError``-named error or
    the signal's own reason after cancellation. Such a failure is a
    cancellation, not a generic tool error, and must be labeled with the
    canonical abort code rather than logged as ``UNKNOWN``.
    """
    if _error_info(error) is not None:
        return False  # already classified by a harness error
    if signal is not None and getattr(signal, "is_aborted", False):
        if error is getattr(signal, "reason", None):
            return True
        return type(error).__name__ == "AbortError"
    return False


def _tool_error_result(
    error: Any, *, info_override: ToolErrorInfo | None = None
) -> ToolExecutionFailure:
    info = info_override if info_override is not None else _error_info(error)
    message = tool_error_message(error)
    failure: ToolFailure = {"message": message}
    if info is not None:
        failure["info"] = info
    return {
        "isError": True,
        "error": failure,
        "content": [{"type": "text", "text": f"Error: {message}"}],
    }


def render_content_summary(content: Sequence[Mapping[str, Any]]) -> str:
    """Derive a concise diagnostic message from content blocks."""
    if not content:
        return "tool result blocked by post-execute policy"
    parts: list[str] = []
    for block in content:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        else:
            parts.append(f"[{block.get('type', 'unknown')} content]")
    text = "\n".join(parts)
    return text if text else "tool result blocked by post-execute policy"


def build_aborted_result(
    *, before_dispatch: bool = False, prior: Mapping[str, Any] | None = None
) -> ToolExecutionFailure:
    """Create the canonical aborted tool failure payload."""
    code = TOOL_ABORTED_BEFORE_DISPATCH if before_dispatch else TOOL_ABORTED
    msg = (
        "tool call aborted before dispatch" if before_dispatch else "tool call aborted"
    )
    out: dict[str, Any] = {
        "isError": True,
        "error": {"message": msg, "info": {"name": "AbortError", "code": code}},
        "content": [{"type": "text", "text": f"Error: {msg}"}],
    }
    additional = prior.get("additionalContexts") if prior is not None else None
    if additional:
        out["additionalContexts"] = list(additional)
    return cast(ToolExecutionFailure, out)


def _snapshot_json_lossless(value: Any, error_message: str) -> Any:
    if not is_json_value(value):
        raise TypeError(error_message)
    return None if value is None else snapshot_json_value(value)


def _snapshot_tool_value(tool_name: str, candidate: Any) -> Any:
    if not is_json_value(candidate):
        raise ToolOutputError(tool_name, ["value is not lossless JSON"])
    return None if candidate is None else snapshot_json_value(candidate)


def _snapshot_projection(tool_name: str, projector: str, candidate: Any) -> Any:
    if not is_json_value(candidate):
        raise ToolOutputError(
            tool_name, [f"output.{projector} returned non-lossless JSON"]
        )
    return None if candidate is None else snapshot_json_value(candidate)


def _materialize_presentation(candidate: Mapping[str, Any]) -> Any:
    detached = _snapshot_json_lossless(
        candidate, "tool result must be losslessly JSON-serializable"
    )
    return deep_freeze(detached)


def _materialize_final_result(result: Mapping[str, Any]) -> ToolExecutionResult:
    presentation: dict[str, Any] = {"content": result["content"]}
    if "meta" in result:
        presentation["meta"] = result["meta"]
    if "additionalContexts" in result and result["additionalContexts"] is not None:
        presentation["additionalContexts"] = result["additionalContexts"]
    if result.get("isError"):
        return cast(
            ToolExecutionResult,
            _materialize_presentation(
                {"isError": True, "error": result["error"], **presentation}
            ),
        )
    candidate: dict[str, Any] = {"isError": False, **presentation}
    if result.get("concludesTurn") is True:
        candidate["concludesTurn"] = True
    detached = _materialize_presentation(candidate)
    return cast(
        ToolExecutionResult,
        FrozenDict({**detached, "value": result["value"]}),
    )


def _validate_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


class ToolExecutionPipeline:
    """Configurable per-call pipeline driving validation, policy, guards, and completion."""

    def __init__(self, ctx: Context, source: Any = None) -> None:
        self._ctx = ctx
        self._source = source
        self._states: WeakKeyDictionary[_ExecutionToken, _ExecutionState] = (
            WeakKeyDictionary()
        )

    @property
    def ctx(self) -> Context:
        """Return the context hosting tool services and event pipelines."""
        return self._ctx

    def _tools_service(self) -> Any:
        return self._ctx.get("tools")

    def _scoped_target(self, agent: Any) -> Any:
        return scope_target(self._tools_service(), agent)

    def _get(
        self, name: str, scope: Any = None, tool_set: Any = None
    ) -> ToolDefinition | None:
        source_get = getattr(self._source, "get", None)
        if callable(source_get):
            return cast(ToolDefinition | None, source_get(name, scope, tool_set))
        tools = self._tools_service()
        return cast(ToolDefinition | None, tools.executable(name, scope, tool_set))

    def _reject(self, name: str, nested: bool) -> Any:
        reject_fn = getattr(self._source, "reject", None)
        if callable(reject_fn):
            return reject_fn(name, nested)
        return None

    def _resolve_execution(
        self, name: str, scope: Any, nested: bool, tool_set: Any
    ) -> ToolDefinition | None:
        tool = self._get(name, scope, tool_set)
        if tool is not None and self._reject(name, nested) is None:
            return tool
        return None

    def _require_state(self, exec_view: Mapping[str, Any]) -> _ExecutionState:
        token = exec_view.get("token")
        state = self._states.get(token) if isinstance(token, _ExecutionToken) else None
        if state is None:
            raise RuntimeError(
                "tool registry scheduler invariant violated: missing execution state"
            )
        return state

    def execution_mode(self, execution_input: Mapping[str, Any]) -> ToolExecutionMode:
        """Fail-closed per-call concurrency classification."""
        tool = self._resolve_execution(
            execution_input["name"],
            execution_input.get("agent"),
            execution_input.get("parent") is not None,
            execution_input.get("toolSet"),
        )
        if tool is None:
            return {"kind": "exclusive"}
        classifier = (
            tool.get("isConcurrencySafe")
            if isinstance(tool, Mapping)
            else getattr(tool, "isConcurrencySafe", None)
        )
        if not callable(classifier):
            return {"kind": "exclusive"}
        try:
            verdict = classifier(execution_input.get("arguments"))
        except Exception:  # noqa: BLE001 - fail closed on throwing classifier
            return {"kind": "exclusive"}
        return {"kind": "parallel"} if verdict is True else {"kind": "exclusive"}

    def _create_execution(self, execution_input: Mapping[str, Any]) -> dict[str, Any]:
        token = _ExecutionToken()
        call_id = str(execution_input["callId"])
        raw_root = execution_input.get("rootCallId")
        root_call_id = str(raw_root) if raw_root is not None else call_id
        name = str(execution_input["name"])
        agent = execution_input.get("agent")
        parent = execution_input.get("parent")
        tool_set = execution_input.get("toolSet")
        schema = execution_input.get("schema")
        signal: AbortSignal = execution_input["signal"]

        visible = self._get(name, agent, tool_set)
        rejection = None if visible is None else self._reject(name, parent is not None)
        rejected = rejection is not None

        state = _ExecutionState(caller_signal=signal)

        def _defer_context(context: Mapping[str, Any]) -> None:
            state.deferred_contexts.append(context)

        def _conclude_turn() -> None:
            state.conclude_turn = True

        captured_finalizer = (
            visible.get("finalizeContent") if isinstance(visible, Mapping) else None
        )
        captured_projector = (
            visible.get("projectContent") if isinstance(visible, Mapping) else None
        )

        def _finalizer_for() -> (
            Callable[[ToolExecution, ToolExecutionResult], Sequence[Any] | None] | None
        ):
            if rejected and not signal.is_aborted:
                return None
            return captured_finalizer

        base: dict[str, Any] = {
            "token": token,
            "callId": call_id,
            "rootCallId": root_call_id,
            "name": name,
            "signal": signal,
            "requiresApproval": (
                bool(visible.get("requiresApproval"))
                if isinstance(visible, Mapping)
                else False
            ),
            "deferContext": _defer_context,
            "concludeTurn": _conclude_turn,
        }
        if agent is not None:
            base["agent"] = agent
        if tool_set is not None:
            base["toolSet"] = tool_set
        if parent is not None:
            base["parent"] = parent
        if schema is not None:
            base["schema"] = schema

        try:
            raw_args = execution_input.get("arguments", _MISSING)
            if raw_args is _MISSING:
                raise TypeError(
                    "tool execution arguments must be losslessly JSON-serializable"
                )
            detached = _snapshot_json_lossless(
                raw_args,
                "tool execution arguments must be losslessly JSON-serializable",
            )
            exec_ctx: dict[str, Any] = {**base, "arguments": deep_freeze(detached)}
            state.content_finalizer = _finalizer_for()
            if not rejected:
                state.content_projector = captured_projector
            self._states[token] = state
            if rejected:
                if signal.is_aborted:
                    return {
                        "kind": "final-result",
                        "exec": exec_ctx,
                        "result": build_aborted_result(before_dispatch=True),
                    }
                return {
                    "kind": "final-result",
                    "exec": exec_ctx,
                    "result": _tool_error_result(rejection),
                }
            return {"kind": "ready", "exec": exec_ctx}
        except Exception as exc:  # noqa: BLE001 - snapshot failures become tool errors
            exec_ctx = {**base, "arguments": None}
            state.content_finalizer = _finalizer_for()
            self._states[token] = state
            return {
                "kind": "final-result",
                "exec": exec_ctx,
                "result": _tool_error_result(exc),
            }

    def _caller_cancelled(self, exec_view: Mapping[str, Any]) -> bool:
        return self._require_state(exec_view).caller_signal.is_aborted

    def _cancellation_result(
        self, exec_view: Mapping[str, Any], prior: ToolExecutionResult | None = None
    ) -> ToolExecutionFailure:
        state = self._require_state(exec_view)
        return build_aborted_result(before_dispatch=not state.body_invoked, prior=prior)

    async def prepare(self, execution_input: ToolExecutionInput) -> dict[str, Any]:
        """Snapshot input and run ordered pre-execute policy and guards."""
        created = self._create_execution(execution_input)
        if created["kind"] != "ready":
            return created
        exec_view: dict[str, Any] = created["exec"]
        if self._caller_cancelled(exec_view):
            return {
                "kind": "final-result",
                "exec": exec_view,
                "result": build_aborted_result(before_dispatch=True),
            }
        tools = self._tools_service()
        target = self._scoped_target(exec_view.get("agent"))
        try:
            gate = await waterfall_scoped(
                self._ctx,
                target,
                TOOLS_PRE_EXECUTE,
                exec_view,
                default_handler=lambda _e=None: {"kind": "allow"},
            )
            if gate is None:
                gate = {"kind": "allow"}
            ask_res = (
                await self._service_ask(exec_view, gate)
                if gate.get("kind") == "ask"
                else _AskResolution(decision=gate, approval_cancelled=False)
            )
            decision = ask_res.decision
            if self._caller_cancelled(exec_view) and ask_res.approval_cancelled:
                return {
                    "kind": "post-result",
                    "exec": exec_view,
                    "result": build_aborted_result(before_dispatch=True),
                }
            kind = decision.get("kind")
            if kind == "cancel":
                return {
                    "kind": "post-result",
                    "exec": exec_view,
                    "result": build_aborted_result(before_dispatch=True),
                }
            if kind == "allow":
                denial_reason = tools.check_guards(exec_view)
                denial_info = None
            elif kind == "deny":
                denial_reason = decision.get("reason")
                if not isinstance(denial_reason, str) or not denial_reason:
                    raise ValueError(
                        "tools/pre-execute deny decision must include a non-empty reason"
                    )
                denial_info = decision.get("info")
            else:
                raise ValueError(f"invalid tools/pre-execute decision: {decision!r}")

            if denial_reason is not None:
                err_obj: dict[str, Any] = {"message": denial_reason}
                if denial_info is not None:
                    err_obj["info"] = copy.deepcopy(denial_info)
                denied_result = _materialize_final_result(
                    {
                        "content": [
                            {"type": "text", "text": f"Error: {denial_reason}"}
                        ],
                        "isError": True,
                        "error": err_obj,
                    }
                )
                return {
                    "kind": "post-result",
                    "exec": exec_view,
                    "result": denied_result,
                }
            if self._caller_cancelled(exec_view):
                return {
                    "kind": "post-result",
                    "exec": exec_view,
                    "result": build_aborted_result(before_dispatch=True),
                }
            return {"kind": "dispatch", "exec": exec_view}
        except Exception as exc:  # noqa: BLE001 - policy throws become final error results
            return {
                "kind": "final-result",
                "exec": exec_view,
                "result": _tool_error_result(exc),
            }

    async def _service_ask(
        self, exec_view: Mapping[str, Any], ask: Mapping[str, Any]
    ) -> _AskResolution:
        approval = self._ctx.get("approval")
        tool_name = exec_view["name"]
        if approval is None:
            reason = ask.get("reason")
            msg = (
                reason
                if isinstance(reason, str) and reason
                else f'tool "{tool_name}" requires approval but no approval service is mounted'
            )
            return _AskResolution(
                decision={"kind": "deny", "reason": msg},
                approval_cancelled=False,
            )
        agent = exec_view.get("agent")
        if agent is None:
            msg = (
                f'tool "{tool_name}" requires approval, '
                "but the call has no agent to route it through"
            )
            return _AskResolution(
                decision={"kind": "deny", "reason": msg},
                approval_cancelled=False,
            )
        req: dict[str, Any] = {
            "agent": agent,
            "toolName": tool_name,
            "callId": exec_view["callId"],
            "signal": exec_view["signal"],
        }
        if "reason" in ask and ask["reason"] is not None:
            req["reason"] = ask["reason"]
        if "displayReason" in ask and ask["displayReason"] is not None:
            req["displayReason"] = ask["displayReason"]
        # The approval service owns the approval/asked + approval/decided
        # audit pair; it appends both to the requesting session.
        outcome = approval.request(req)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        if outcome == "allowed-once":
            return _AskResolution(decision={"kind": "allow"}, approval_cancelled=False)
        if outcome == "rejected":
            return _AskResolution(
                decision={
                    "kind": "deny",
                    "reason": f'the user rejected tool "{tool_name}"',
                },
                approval_cancelled=False,
            )
        if outcome == "cancelled":
            return _AskResolution(
                decision={
                    "kind": "deny",
                    "reason": f'approval for tool "{tool_name}" was cancelled',
                },
                approval_cancelled=True,
            )
        if outcome == "unavailable":
            return _AskResolution(
                decision={
                    "kind": "deny",
                    "reason": (
                        f'tool "{tool_name}" requires approval, '
                        "but no approval channel is available"
                    ),
                },
                approval_cancelled=False,
            )
        assert_never(outcome, "ApprovalOutcome")

    async def _dispatch_tool_body(
        self, exec_view: dict[str, Any]
    ) -> ToolExecutionResult:
        state = self._require_state(exec_view)
        wrapper_signal: AbortSignal = exec_view["signal"]
        if state.caller_signal is wrapper_signal:
            signal = state.caller_signal

            def _dispose() -> None:
                pass
        else:
            signal, _dispose = fuse_tool_signals(state.caller_signal, wrapper_signal)

        if signal.is_aborted:
            _dispose()
            return build_aborted_result(before_dispatch=True)

        exec_view["signal"] = signal
        try:
            tool = self._resolve_execution(
                exec_view["name"],
                exec_view.get("agent"),
                exec_view.get("parent") is not None,
                exec_view.get("toolSet"),
            )
            if tool is None:
                raise ToolNotFoundError(exec_view["name"])
            state.body_invoked = True
            # Timeout enforcement is a cooperative, scoped-deadline concern of
            # a mounted tools/execute guard (TOOL_TIMEOUT), not a hard cancel.
            # A sync body that declares timeoutMs runs in a worker thread so
            # the event-loop deadline can fire while it polls its signal.
            timeout_ms = tool.get("timeoutMs")
            if timeout_ms is not None:
                raw_returned = await asyncio.to_thread(
                    tool["execute"],
                    exec_view["arguments"],
                    cast(ToolRunContext, exec_view),
                )
            else:
                raw_returned = tool["execute"](
                    exec_view["arguments"], cast(ToolRunContext, exec_view)
                )
            if inspect.isawaitable(raw_returned):
                raw_returned = await raw_returned
            result = self._create_success_result(exec_view, tool, raw_returned)
            if signal.is_aborted:
                return build_aborted_result(before_dispatch=False, prior=result)
            return result
        except Exception as exc:  # noqa: BLE001 - contain tool execution failures
            if _is_unclassified_cancellation(exc, state.caller_signal):
                code = (
                    TOOL_ABORTED_BEFORE_DISPATCH
                    if not state.body_invoked
                    else TOOL_ABORTED
                )
                return _tool_error_result(
                    exc, info_override={"name": "AbortError", "code": code}
                )
            return _tool_error_result(exc)
        finally:
            _dispose()
            exec_view["signal"] = wrapper_signal

    async def dispatch(self, exec_view: dict[str, Any]) -> dict[str, Any]:
        """Run around-dispatch wrappers and the tool body."""
        try:
            target = self._scoped_target(exec_view.get("agent"))
            result = await waterfall_scoped(
                self._ctx,
                target,
                TOOLS_EXECUTE,
                exec_view,
                default_handler=lambda _e=None: self._dispatch_tool_body(exec_view),
            )
            normalized = self._normalize_dispatch_result(exec_view, result)
            state = self._require_state(exec_view)
            if state.deferred_contexts:
                combined_contexts = [
                    *state.deferred_contexts,
                    *(normalized.get("additionalContexts") or []),
                ]
                merged: dict[str, Any] = dict(normalized)
                merged["additionalContexts"] = combined_contexts
                result_with_deferred = state.mark_canonical(
                    cast(ToolExecutionResult, merged)
                )
            else:
                result_with_deferred = normalized

            if (
                self._caller_cancelled(exec_view)
                and not result_with_deferred["isError"]
            ):
                return {
                    "kind": "post-result",
                    "result": self._cancellation_result(
                        exec_view, result_with_deferred
                    ),
                }
            return {"kind": "post-result", "result": result_with_deferred}
        except Exception as exc:  # noqa: BLE001 - wrapper/normalization errors finalize
            # A wrapper throwing an unlabeled cancellation after the caller
            # aborted is a cancellation, not a generic tool error.
            try:
                state = self._require_state(exec_view)
                if _is_unclassified_cancellation(exc, state.caller_signal):
                    code = (
                        TOOL_ABORTED_BEFORE_DISPATCH
                        if not state.body_invoked
                        else TOOL_ABORTED
                    )
                    result = _tool_error_result(
                        exc, info_override={"name": "AbortError", "code": code}
                    )
                else:
                    result = _tool_error_result(exc)
            except RuntimeError:  # state already finalized
                result = _tool_error_result(exc)
            return {"kind": "final-result", "result": result}

    def _create_success_result(
        self,
        exec_view: Mapping[str, Any],
        tool: ToolDefinition,
        candidate: Any,
    ) -> ToolExecutionSuccess:
        tool_name = tool["name"]
        detached = _snapshot_tool_value(tool_name, candidate)
        output_def = tool["output"]
        violations = validate_json_schema_value(
            output_def["schema"], detached, name="value"
        )
        if violations:
            raise ToolOutputError(tool_name, violations)
        frozen_value = deep_freeze(detached)

        try:
            rendered = output_def["render"](exec_view["arguments"], frozen_value)
        except Exception as exc:
            raise ToolOutputError(
                tool_name, [f"output.render failed: {tool_error_message(exc)}"]
            ) from exc
        if isinstance(rendered, tuple):
            rendered = list(rendered)
        if not isinstance(rendered, list):
            raise ToolOutputError(
                tool_name, ["output.render returned non-lossless JSON"]
            )
        content = _snapshot_projection(tool_name, "render", rendered)

        payload: dict[str, Any] = {
            "isError": False,
            "value": frozen_value,
            "content": content,
        }
        meta_fn = output_def.get("presentationMeta")
        if callable(meta_fn):
            try:
                raw_meta = meta_fn(exec_view["arguments"], frozen_value)
            except Exception as exc:
                raise ToolOutputError(
                    tool_name,
                    [f"output.presentationMeta failed: {tool_error_message(exc)}"],
                ) from exc
            payload["meta"] = _snapshot_projection(
                tool_name, "presentationMeta", raw_meta
            )

        state = self._require_state(exec_view)
        if state.conclude_turn:
            payload["concludesTurn"] = True

        materialized = cast(ToolExecutionSuccess, _materialize_final_result(payload))
        return cast(ToolExecutionSuccess, state.mark_canonical(materialized))

    def _normalize_dispatch_result(
        self, exec_view: Mapping[str, Any], result: Mapping[str, Any]
    ) -> ToolExecutionResult:
        state = self._require_state(exec_view)
        if state.is_canonical(result):
            return cast(ToolExecutionResult, result)
        if result.get("isError"):
            authored: dict[str, Any] = {
                "isError": True,
                "error": result["error"],
                "content": result["content"],
            }
            if "meta" in result:
                authored["meta"] = result["meta"]
            if (
                "additionalContexts" in result
                and result["additionalContexts"] is not None
            ):
                authored["additionalContexts"] = result["additionalContexts"]
            return state.mark_canonical(cast(ToolExecutionResult, authored))
        if "value" not in result:
            raise ToolOutputError(exec_view["name"], ["value is not lossless JSON"])
        tool = self._resolve_execution(
            exec_view["name"],
            exec_view.get("agent"),
            exec_view.get("parent") is not None,
            exec_view.get("toolSet"),
        )
        if tool is None:
            raise ToolNotFoundError(exec_view["name"])
        normalized = self._create_success_result(exec_view, tool, result["value"])
        if "additionalContexts" in result and result["additionalContexts"] is not None:
            merged = dict(normalized)
            merged["additionalContexts"] = result["additionalContexts"]
            return state.mark_canonical(cast(ToolExecutionResult, merged))
        return normalized

    async def _post_execute(
        self, exec_view: Mapping[str, Any], result: ToolExecutionResult
    ) -> ToolExecutionResult:
        target = self._scoped_target(exec_view.get("agent"))
        frozen_input = _materialize_final_result(result)
        decision = await waterfall_scoped(
            self._ctx,
            target,
            TOOLS_POST_EXECUTE,
            exec_view,
            frozen_input,
            default_handler=lambda *_a: {"kind": "accept"},
        )
        if decision is None:
            decision = {"kind": "accept"}
        state = self._require_state(exec_view)
        raw_decision_contexts = decision.get("additionalContexts")
        if raw_decision_contexts is not None and (
            not isinstance(raw_decision_contexts, Sequence)
            or isinstance(raw_decision_contexts, (str, bytes))
        ):
            raise TypeError("additionalContexts must be a list of messages")
        decision_contexts = list(raw_decision_contexts or [])
        kind = decision.get("kind")
        if kind == "block":
            feedback = decision.get("feedback")
            if not isinstance(feedback, Sequence) or isinstance(feedback, (str, bytes)):
                raise TypeError("tools/post-execute block feedback must be a list")
            message = render_content_summary(feedback)
            blocked: dict[str, Any] = {
                "content": list(feedback),
                "isError": True,
                "error": {"message": message},
            }
            if decision_contexts:
                blocked["additionalContexts"] = decision_contexts
            return state.mark_canonical(cast(ToolExecutionResult, blocked))
        if kind != "accept":
            raise ValueError(f"invalid tools/post-execute decision: {decision!r}")
        if "content" in decision and "value" in decision:
            raise TypeError(
                "tools/post-execute accept decision cannot replace both value and content"
            )

        additional_contexts = [
            *(result.get("additionalContexts") or []),
            *decision_contexts,
        ]
        if "value" in decision:
            if result["isError"]:
                raise TypeError(
                    "tools/post-execute cannot replace the value of a failed result"
                )
            tool = self._resolve_execution(
                exec_view["name"],
                exec_view.get("agent"),
                exec_view.get("parent") is not None,
                exec_view.get("toolSet"),
            )
            if tool is None:
                raise ToolNotFoundError(exec_view["name"])
            replaced = self._create_success_result(exec_view, tool, decision["value"])
            if additional_contexts:
                out = dict(replaced)
                out["additionalContexts"] = additional_contexts
                return state.mark_canonical(cast(ToolExecutionResult, out))
            return replaced

        accepted: dict[str, Any] = dict(result)
        if "content" in decision and decision["content"] is not None:
            accepted["content"] = decision["content"]
        if additional_contexts:
            accepted["additionalContexts"] = additional_contexts
        elif "additionalContexts" in accepted:
            del accepted["additionalContexts"]
        return state.mark_canonical(cast(ToolExecutionResult, accepted))

    async def complete(
        self, exec_view: dict[str, Any], outcome: Mapping[str, Any]
    ) -> ToolExecutionResult:
        """Apply projectContent, tools/post-execute, finalizeContent, and tools/result."""
        kind = outcome.get("kind")
        if kind == "final-result":
            return self._finish(exec_view, outcome["result"])
        if kind != "post-result":
            assert_never(kind, "scheduled tool outcome")

        result: ToolExecutionResult = outcome["result"]
        try:
            state = self._require_state(exec_view)
            projector = state.content_projector
            state.content_projector = None
            projected = result
            if callable(projector):
                projected_content = projector(cast(ToolExecution, exec_view), result)
                if projected_content is not None:
                    projected = state.mark_canonical(
                        _materialize_final_result(
                            {**result, "content": projected_content}
                        )
                    )
            post_result = await self._post_execute(exec_view, projected)
            if self._caller_cancelled(exec_view) and not post_result["isError"]:
                return self._finish(
                    exec_view, self._cancellation_result(exec_view, post_result)
                )
            return self._finish(exec_view, post_result)
        except Exception as exc:  # noqa: BLE001 - contain post-execute & projectContent errors
            return self._finish(exec_view, _tool_error_result(exc))

    def _apply_final_content(
        self, exec_view: Mapping[str, Any], result: ToolExecutionResult
    ) -> ToolExecutionResult:
        state = self._require_state(exec_view)
        finalize_fn = state.content_finalizer
        if finalize_fn is None:
            return result
        final_content = finalize_fn(cast(ToolExecution, exec_view), result)
        if final_content is None:
            return result
        return cast(ToolExecutionResult, {**result, "content": final_content})

    def _notify_result(
        self, exec_view: Mapping[str, Any], result: ToolExecutionResult
    ) -> None:
        frozen_exec = FrozenDict(exec_view)
        try:
            emit_scoped(
                self._ctx,
                self._scoped_target(exec_view.get("agent")),
                TOOLS_RESULT,
                frozen_exec,
                result,
            )
        except Exception as exc:  # noqa: BLE001 - observer failures must not escape
            self._ctx.logger.warning(
                f'tool "{exec_view["name"]}" ({exec_view["callId"]}): '
                f"tools/result observer failed: {tool_error_message(exc)}"
            )

    def _finish(
        self, exec_view: dict[str, Any], result: ToolExecutionResult
    ) -> ToolExecutionResult:
        try:
            try:
                materialized = _materialize_final_result(result)
            except Exception as exc:  # noqa: BLE001 - materialize fault fallback
                materialized = _materialize_final_result(_tool_error_result(exc))
            try:
                final_result = _materialize_final_result(
                    self._apply_final_content(exec_view, materialized)
                )
            except Exception as exc:  # noqa: BLE001 - finalizer fault fallback
                final_result = _materialize_final_result(_tool_error_result(exc))
            self._notify_result(exec_view, final_result)
            return final_result
        finally:
            token = exec_view.get("token")
            if isinstance(token, _ExecutionToken):
                self._states.pop(token, None)

    async def run(self, execution_input: ToolExecutionInput) -> ToolExecutionResult:
        """Run one tool invocation end-to-end and emit its final tools/result."""
        prepared = await self.prepare(execution_input)
        if prepared["kind"] == "dispatch":
            dispatched = await self.dispatch(prepared["exec"])
            return await self.complete(prepared["exec"], dispatched)
        return await self.complete(prepared["exec"], prepared)


def parse_arguments(raw: Any) -> Any:
    """Parse model arguments, preserving malformed JSON as raw text and empty as {}."""
    if not isinstance(raw, str):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


class _BatchSettlement:
    """Collect model-ordered results and optionally append durable Session events."""

    def __init__(self, input_data: ToolBatchInput) -> None:
        self._input = input_data
        self._results: list[ToolCallResult] = []
        self._contexts: list[Mapping[str, Any]] = []

    def _session(self) -> Any:
        agent = self._input.get("agent")
        session = (
            agent.get("session")
            if isinstance(agent, Mapping)
            else getattr(agent, "session", None)
        )
        if session is None:
            raise ValueError(
                "ToolBatchInput requires agent.session when log is provided"
            )
        return session

    def start(self, block: Mapping[str, Any]) -> int | None:
        log = self._input.get("log")
        if log is None:
            return None
        session = self._session()
        event = session.append(
            "tool/call",
            {
                "turn": log["turn"],
                "step": log["step"],
                "callId": block["id"],
                "name": block["name"],
                "arguments": block["arguments"],
            },
        )
        return cast(int, event.seq)

    def commit(
        self,
        block: Mapping[str, Any],
        result: ToolExecutionResult,
        call_seq: int | None,
    ) -> None:
        log = self._input.get("log")
        if log is not None:
            session = self._session()
            payload: dict[str, Any] = {
                "turn": log["turn"],
                "step": log["step"],
                "message": create_tool_result_message(
                    call_id=block["id"],
                    content=result["content"],
                    is_error=result["isError"],
                ),
            }
            error_obj = result.get("error") if result.get("isError") else None
            if (
                isinstance(error_obj, Mapping)
                and "info" in error_obj
                and error_obj["info"] is not None
            ):
                payload["error"] = error_obj["info"]
            if "meta" in result:
                payload["meta"] = result["meta"]
            session.append(
                "tool/result",
                payload,
                surface_op="append",
                source_event_seqs=[call_seq],
            )
        self._results.append({"callId": block["id"], "result": result})
        self._contexts.extend(result.get("additionalContexts") or [])

    def skip(self, block: Mapping[str, Any]) -> None:
        self.commit(
            block, build_aborted_result(before_dispatch=True), self.start(block)
        )

    def result(self, concluded: bool | None = None) -> ToolBatchResult:
        is_concluded = (
            concluded
            if concluded is not None
            else any(
                entry["result"].get("concludesTurn") is True for entry in self._results
            )
        )
        return {
            "results": self._results,
            "additionalContexts": self._contexts,
            "concluded": is_concluded,
        }


@dataclass(frozen=True)
class _PlannedCall:
    block: Mapping[str, Any]
    exec_input: ToolExecutionInput


@dataclass(frozen=True)
class _GroupOutcome:
    consumed: int
    aborted: bool
    concluded: bool


async def _run_group(
    pipeline: ToolExecutionPipeline,
    settlement: _BatchSettlement,
    max_parallel_calls: int,
    group: list[_PlannedCall],
    mode: str,
    signal: AbortSignal,
) -> _GroupOutcome:
    slots: list[dict[str, Any] | None] = [None] * len(group)
    call_seqs: list[int | None] = [None] * len(group)
    next_to_start = 0
    committed = 0
    started = 0
    aborted = signal.is_aborted
    concluded = False
    scheduler_failure: list[BaseException] = []
    in_flight: dict[asyncio.Task[int], int] = {}

    def _raise_if_failed() -> None:
        if scheduler_failure:
            raise scheduler_failure[0]

    async def _commit_ready() -> None:
        nonlocal committed, concluded
        while committed < len(group):
            slot = slots[committed]
            if slot is None:
                break
            call = group[committed]
            result = await pipeline.complete(slot["exec"], slot)
            settlement.commit(call.block, result, call_seqs[committed])
            if result.get("concludesTurn") is True:
                concluded = True
            committed += 1

    async def _dispatch_slot(index: int, exec_ctx: dict[str, Any]) -> int:
        try:
            outcome = await pipeline.dispatch(exec_ctx)
            slots[index] = {**outcome, "exec": exec_ctx}
        except Exception as exc:  # noqa: BLE001 - defer raising until drain completes
            if not scheduler_failure:
                scheduler_failure.append(exc)
        return index

    async def _start_call(index: int) -> None:
        nonlocal started
        call = group[index]
        call_seqs[index] = settlement.start(call.block)
        started += 1
        prepared = await pipeline.prepare(call.exec_input)
        _raise_if_failed()
        prep_kind = prepared["kind"]
        if prep_kind == "dispatch":
            task = asyncio.create_task(_dispatch_slot(index, prepared["exec"]))
            in_flight[task] = index
        elif prep_kind in ("post-result", "final-result"):
            slots[index] = prepared
        else:
            assert_never(prep_kind, "tool-call scheduler prepare result")

    async def _fill_pool() -> None:
        nonlocal aborted, next_to_start
        while (
            not aborted
            and next_to_start < len(group)
            and len(in_flight) < max_parallel_calls
        ):
            next_call = group[next_to_start]
            if (
                next_to_start > 0
                and mode == "parallel"
                and pipeline.execution_mode(next_call.exec_input)["kind"] != "parallel"
            ):
                break
            await _start_call(next_to_start)
            next_to_start += 1
            _raise_if_failed()
            await _commit_ready()
            _raise_if_failed()
            if signal.is_aborted:
                aborted = True

    try:
        await _fill_pool()
        while in_flight:
            done, _pending = await asyncio.wait(
                in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for finished in done:
                in_flight.pop(finished, None)
            _raise_if_failed()
            await _commit_ready()
            _raise_if_failed()
            if signal.is_aborted:
                aborted = True
            await _fill_pool()
    except Exception as exc:
        if not scheduler_failure:
            scheduler_failure.append(exc)
        if in_flight:
            await asyncio.gather(*in_flight.keys(), return_exceptions=True)
            in_flight.clear()
        raise scheduler_failure[0] from exc

    if aborted:
        for call in group[started:]:
            settlement.skip(call.block)
        return _GroupOutcome(consumed=len(group), aborted=True, concluded=concluded)

    return _GroupOutcome(consumed=started, aborted=False, concluded=concluded)


async def execute_tool_batch(
    input_data: ToolBatchInput,
    pipeline: ToolExecutionPipeline,
    max_parallel: Callable[[], int] | int = 10,
) -> ToolBatchResult:
    """Execute a step's tool calls with exclusive barriers and bounded parallelism."""
    override = input_data.get("maxParallelToolCalls")
    if override is not None:
        _validate_positive_int(override, "maxParallelToolCalls")
    elif not callable(max_parallel):
        _validate_positive_int(max_parallel, "maxParallelToolCalls")

    settlement = _BatchSettlement(input_data)
    signal = input_data["signal"]
    agent = input_data.get("agent")
    tool_set = input_data.get("toolSet")

    try:
        planned: list[_PlannedCall] = []
        for block in input_data["calls"]:
            exec_input: dict[str, Any] = {
                "callId": block["id"],
                "name": block["name"],
                "arguments": parse_arguments(block.get("arguments", "")),
                "signal": signal,
            }
            if agent is not None:
                exec_input["agent"] = agent
            if tool_set is not None:
                exec_input["toolSet"] = tool_set
            planned.append(
                _PlannedCall(
                    block=block, exec_input=cast(ToolExecutionInput, exec_input)
                )
            )

        next_idx = 0
        concluded = False
        while next_idx < len(planned):
            first = planned[next_idx]
            mode = pipeline.execution_mode(first.exec_input)["kind"]
            group = planned[next_idx:] if mode == "parallel" else [first]
            limit = (
                override
                if override is not None
                else _validate_positive_int(
                    max_parallel() if callable(max_parallel) else max_parallel,
                    "maxParallelToolCalls",
                )
            )
            outcome = await _run_group(pipeline, settlement, limit, group, mode, signal)
            next_idx += outcome.consumed
            concluded = concluded or outcome.concluded
            if outcome.aborted:
                for call in planned[next_idx:]:
                    settlement.skip(call.block)
                return settlement.result(concluded)
        return settlement.result(concluded)
    except Exception as exc:
        raise ToolBatchExecutionError(exc, settlement.result()) from exc


__all__ = [
    "TOOLS_CHANGE",
    "TOOLS_EXECUTE",
    "TOOLS_POST_EXECUTE",
    "TOOLS_PRE_EXECUTE",
    "TOOLS_RESULT",
    "ToolExecutionPipeline",
    "ToolExecutionSource",
    "build_aborted_result",
    "execute_tool_batch",
    "fuse_tool_signals",
    "parse_arguments",
    "render_content_summary",
    "tool_error_message",
]
