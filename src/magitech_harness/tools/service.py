"""ToolsService (ctx.tools): Scoped tool registry, activation, restrictions, and execution."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from anima import Context, Disposer, Service
from anima.core import _access_context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.types import ToolSchema
from magitech_harness.scope import (
    AnonymousEntries,
    NamedEntries,
    ScopedLayers,
    scope_of,
)
from magitech_harness.session.events import ToolCallData, ToolResultData
from magitech_harness.tools.contracts import (
    RUN_CODE_NAME,
    ResolvedToolSet,
    ToolActivationContext,
    ToolActivationError,
    ToolBatchInput,
    ToolDefinition,
    ToolExecutor,
    ToolExecutorUnavailableError,
    ToolGuard,
    ToolRegistrationOptions,
    ToolRestriction,
    ToolRunContext,
)
from magitech_harness.tools.json_schema import assert_supported_json_schema
from magitech_harness.tools.pipeline import TOOLS_CHANGE
from magitech_harness.tools.schema import project_tool_schema, validate_timeout_ms

logger = logging.getLogger("harness.tools")

_EXECUTION_DEPTH: ContextVar[int] = ContextVar(
    "magitech_tools_execute_depth", default=0
)
_NESTED_CALL = object()


@dataclass
class _RegisteredToolEntry:
    definition: ToolDefinition
    when: Callable[[ToolActivationContext], bool | Awaitable[bool]] | None = None


@dataclass(frozen=True)
class _ToolView:
    visible: dict[str, ToolDefinition]
    registrations: dict[str, _RegisteredToolEntry]
    known_names: list[str]
    restrictable_names: set[str]


@dataclass
class _ExecutorSlot:
    current: ToolExecutor | None


@dataclass
class _RestrictionRule:
    allow: frozenset[str] | None
    deny: frozenset[str] | None


class _ToolLayer:
    def __init__(self, scope: Any) -> None:
        self.scope = scope
        self.tools: NamedEntries[_RegisteredToolEntry] = NamedEntries(
            lambda name: ValueError(f'duplicate tool "{name}" registered in scope')
        )
        self.restrictions: AnonymousEntries[_RestrictionRule] = AnonymousEntries()
        self.guards: AnonymousEntries[ToolGuard] = AnonymousEntries()
        self.executor: _ExecutorSlot | None = None

    def is_empty(self) -> bool:
        return (
            self.tools.is_empty()
            and self.restrictions.is_empty()
            and self.guards.is_empty()
            and self.executor is None
        )

    def admits(self, name: str) -> bool:
        """Return True when every restriction in this layer permits *name*."""
        for rule in self.restrictions.values():
            if rule.allow is not None and name not in rule.allow:
                return False
            if rule.deny is not None and name in rule.deny:
                return False
        return True


class _ResolvedToolSet:
    """Snapshot of admitted step registrations pinned to their resolving scope."""

    def __init__(
        self,
        agent: Any,
        admitted: Mapping[str, ToolDefinition],
        snapshot_registrations: Mapping[str, _RegisteredToolEntry],
        read_view: Callable[[Any], _ToolView],
    ) -> None:
        self._agent = agent
        self._admitted = dict(admitted)
        self._snapshot_registrations = dict(snapshot_registrations)
        self._read_view = read_view
        self._closed = False

    def dispose(self) -> None:
        self._closed = True
        self._admitted.clear()
        self._snapshot_registrations.clear()

    def view(self, scope: Any) -> dict[str, ToolDefinition]:
        if self._closed or scope is not self._agent:
            return {}
        current = self._read_view(self._agent)
        visible: dict[str, ToolDefinition] = {}
        for name, definition in self._admitted.items():
            if current.visible.get(name) is definition and current.registrations.get(
                name
            ) is self._snapshot_registrations.get(name):
                visible[name] = definition
        return visible


class ToolsService(Service):
    """Scoped tool registry, restriction layer, admission resolver, and executor router."""

    inject: ClassVar[list[str]] = ["systemPrompt"]

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        for removed in ("mode", "maxParallelSubCalls"):
            if removed in cfg:
                raise ValueError(
                    f'tools: invalid config "{removed}" is no longer supported'
                )

        super().__init__(ctx, "tools")
        self._layers: ScopedLayers[_ToolLayer] = ScopedLayers(
            lambda scope: _ToolLayer(scope),
            lambda: self.ctx.emit(TOOLS_CHANGE),
        )

        if ctx.has("systemPrompt"):
            system_prompt = ctx.get("systemPrompt", None, strict=False)
            if system_prompt is None:
                raise RuntimeError(
                    "tools service: systemPrompt is present but unresolvable"
                )
            system_prompt.tools(self._prompt_tools)

    def _caller_ctx(self) -> Context:
        return _access_context.get() or self.ctx

    def _prompt_tools(self, context: Mapping[str, Any]) -> dict[str, Any]:
        scope = context.get("scope") if "scope" in context else context.get("agent")
        tool_set = context.get("toolSet")
        executor = self.select_executor(scope)
        described = executor.describe(self.definitions(scope, tool_set))
        known_names = described.get("knownNames")
        return {
            "schemas": described["schemas"],
            "knownNames": (
                list(known_names)
                if known_names is not None
                else list(self._view(scope).known_names)
            ),
        }

    # ------------------------------------------------------------------
    # Registration, Restrictions & Guards
    # ------------------------------------------------------------------

    def register(
        self,
        tool_or_def: Mapping[str, Any],
        options: ToolRegistrationOptions | None = None,
    ) -> Disposer:
        """Register a parity ToolDefinition in the caller's scope."""
        name = tool_or_def.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool definition must provide a non-empty string name")
        if name == RUN_CODE_NAME:
            raise ValueError(f'tool name "{RUN_CODE_NAME}" is reserved')
        description = tool_or_def.get("description")
        if not isinstance(description, str):
            raise ValueError(  # noqa: TRY004
                f'tool "{name}" description must be a string'
            )

        output = tool_or_def.get("output")
        if (
            not isinstance(output, Mapping)
            or "schema" not in output
            or not callable(output.get("render"))
            or (
                output.get("presentationMeta") is not None
                and not callable(output.get("presentationMeta"))
            )
        ):
            raise ValueError(
                f'tool "{name}" must declare output {{ schema, render, presentationMeta? }}'
            )

        assert_supported_json_schema(
            output["schema"], path=f"tools.{name}.output.schema"
        )
        timeout_ms = validate_timeout_ms(tool_or_def.get("timeoutMs"))

        if not callable(tool_or_def.get("execute")):
            raise ValueError(  # noqa: TRY004
                f'tool "{name}" must declare a callable execute handler'
            )

        stored: dict[str, Any] = dict(tool_or_def)
        if timeout_ms is not None:
            stored["timeoutMs"] = timeout_ms

        when_fn = options.get("when") if options is not None else None
        if when_fn is not None and not callable(when_fn):
            raise ValueError("tool activation option 'when' must be callable")

        entry = _RegisteredToolEntry(
            definition=cast(ToolDefinition, stored),
            when=when_fn,
        )
        return self._layers.effect(
            self._caller_ctx(),
            lambda layer: layer.tools.insert(name, entry),
            {"label": f"tools.register({name})", "notify": True},
        )

    def restrict(self, restriction: ToolRestriction) -> Disposer:
        """Install an allow/deny restriction in the caller's scoped layer."""
        caller = self._caller_ctx()
        scope = scope_of(caller)
        if scope is None:
            raise ValueError("tools.restrict() requires a scoped context")

        has_allow = "allow" in restriction and restriction["allow"] is not None
        has_deny = "deny" in restriction and restriction["deny"] is not None
        if not has_allow and not has_deny:
            raise ValueError("tools.restrict() requires allow or deny (no-op rejected)")

        allow_list = list(restriction["allow"]) if has_allow else []
        deny_list = list(restriction["deny"]) if has_deny else []
        if RUN_CODE_NAME in allow_list or RUN_CODE_NAME in deny_list:
            raise ValueError(f'tool name "{RUN_CODE_NAME}" cannot be restricted')

        known = self._view(scope).restrictable_names
        unknown = [n for n in (*allow_list, *deny_list) if n not in known]
        if unknown:
            known_str = ", ".join(sorted(known)) if known else "(none)"
            quoted = ", ".join(f'"{u}"' for u in unknown)
            plural = "s" if len(unknown) > 1 else ""
            raise ValueError(
                f"unknown global tool{plural} {quoted}; known global tools: {known_str}"
            )

        rule = _RestrictionRule(
            allow=frozenset(allow_list) if has_allow else None,
            deny=frozenset(deny_list) if has_deny else None,
        )
        return self._layers.effect(
            caller,
            lambda layer: layer.restrictions.append(rule),
            {"label": "tools.restrict()", "notify": True},
        )

    def guard(self, fn: ToolGuard) -> Disposer:
        """Attach an owner-level monotonic guard in global or caller scope."""
        return self._layers.effect(
            self._caller_ctx(),
            lambda layer: layer.guards.append(fn),
            {"label": "tools.guard()", "notify": False},
        )

    def check_guards(self, execution: Any, *, scope: Any = None) -> str | None:
        """Live-iterate global and scoped guards; first string verdict denies the call."""
        target_scope = (
            scope
            if scope is not None
            else (
                execution.get("agent")
                if isinstance(execution, Mapping)
                else getattr(execution, "agent", None)
            )
        )
        for fn in self._layers.global_layer.guards.values():
            verdict = fn(execution)
            if isinstance(verdict, str):
                return verdict
        for layer in self._layers.chain_layers(target_scope):
            for fn in layer.guards.values():
                verdict = fn(execution)
                if isinstance(verdict, str):
                    return verdict
        return None

    # ------------------------------------------------------------------
    # Scoped Views & Activation Admission
    # ------------------------------------------------------------------

    def _view(self, scope: Any = None) -> _ToolView:
        target_scope = scope if scope is not None else scope_of(self._caller_ctx())
        layers = self._layers.chain_layers(target_scope)
        own = self._layers.peek(target_scope)
        inherited: dict[str, _RegisteredToolEntry] = dict(
            self._layers.global_layer.tools.entries()
        )
        for layer in layers:
            if layer is own:
                continue
            inherited.update(layer.tools.entries())

        visible: dict[str, ToolDefinition] = {}
        registrations: dict[str, _RegisteredToolEntry] = {}
        known_names: list[str] = []
        seen_known: set[str] = set()
        restrictable_names: set[str] = set()
        for name, entry in inherited.items():
            seen_known.add(name)
            known_names.append(name)
            restrictable_names.add(name)
            if all(layer.admits(name) for layer in layers):
                visible[name] = entry.definition
                registrations[name] = entry
        if own is not None:
            for name, entry in own.tools.entries():
                if name not in seen_known:
                    seen_known.add(name)
                    known_names.append(name)
                visible[name] = entry.definition
                registrations[name] = entry
        return _ToolView(
            visible=visible,
            registrations=registrations,
            known_names=known_names,
            restrictable_names=restrictable_names,
        )

    def get(
        self, name: str, scope: Any = None, tool_set: Any = None
    ) -> ToolDefinition | None:
        """Look up a tool definition in scope/toolSet."""
        target_scope = scope if scope is not None else scope_of(self._caller_ctx())
        if tool_set is not None:
            if isinstance(tool_set, ResolvedToolSet):
                return tool_set.view(target_scope).get(name)
            return None
        return self._view(target_scope).visible.get(name)

    def executable(
        self, name: str, agent: Any = None, tool_set: Any = None
    ) -> ToolDefinition | None:
        """Look up a parity tool executable for *agent* and optional admitted *tool_set*."""
        target_scope = agent if agent is not None else scope_of(self._caller_ctx())
        if tool_set is not None:
            if isinstance(tool_set, ResolvedToolSet):
                return tool_set.view(target_scope).get(name)
            return None
        view = self._view(target_scope)
        entry = view.registrations.get(name)
        if entry is None or entry.when is not None:
            return None
        return view.visible.get(name)

    async def resolve(self, context: Mapping[str, Any]) -> _ResolvedToolSet:
        """Snapshot conditional activation predicates for one agent step."""
        agent = context["agent"]
        model_selection = context["modelSelection"]
        signal: AbortSignal = context["signal"]
        signal.throw_if_aborted()

        view = self._view(agent)
        admitted: dict[str, ToolDefinition] = {}
        act_ctx: ToolActivationContext = {
            "agent": agent,
            "modelSelection": model_selection,
            "signal": signal,
        }

        for name, definition in view.visible.items():
            entry = view.registrations[name]
            if entry.when is None:
                admitted[name] = definition
                continue
            try:
                verdict = entry.when(act_ctx)
                if callable(getattr(verdict, "__await__", None)):
                    verdict = await cast(Awaitable[bool], verdict)
            except Exception as exc:
                signal.throw_if_aborted()
                raise ToolActivationError(name, cause=exc) from exc
            signal.throw_if_aborted()
            if verdict:
                admitted[name] = definition

        return _ResolvedToolSet(agent, admitted, view.registrations, self._view)

    def definitions(
        self, scope: Any = None, tool_set: Any = None
    ) -> list[ToolDefinition]:
        """Return ordered admitted ToolDefinitions visible in scope."""
        target_scope = scope if scope is not None else scope_of(self._caller_ctx())
        if tool_set is not None:
            if isinstance(tool_set, ResolvedToolSet):
                return list(tool_set.view(target_scope).values())
            return []
        return list(self._view(target_scope).visible.values())

    def schemas(
        self,
        scope: Any = None,
        tool_set: Any = None,
    ) -> list[ToolSchema]:
        """Project visible tools into model-safe ToolSchema dictionaries."""
        return [
            project_tool_schema(defn, detach_parameters=True)
            for defn in self.definitions(scope, tool_set)
        ]

    # ------------------------------------------------------------------
    # Executor Routing & Batch Dispatch
    # ------------------------------------------------------------------

    def set_executor(self, executor: ToolExecutor) -> Disposer:
        """Bind a ToolExecutor (and its optional sections) to the caller's scope."""
        caller = self._caller_ctx()
        scope = scope_of(caller)
        existing_layer = (
            self._layers.global_layer if scope is None else self._layers.peek(scope)
        )
        if (
            existing_layer is not None
            and existing_layer.executor is not None
            and existing_layer.executor.current is not None
        ):
            raise ValueError("Only one tool executor may be registered in each scope")

        slot = _ExecutorSlot(current=executor)
        sp = self.ctx.get("systemPrompt", None, strict=False)
        raw_sections = getattr(executor, "sections", None)
        sections_iter = list(
            raw_sections() if callable(raw_sections) else (raw_sections or ())
        )

        def _attach(layer: _ToolLayer) -> Callable[[], None]:
            section_disposers: list[Disposer] = []
            try:
                if sp is not None:
                    for sec in sections_iter:
                        raw_text = sec["text"]

                        def _section_text(
                            ctx_arg: Mapping[str, Any],
                            text_src: Any = raw_text,
                        ) -> str:
                            sec_scope = (
                                ctx_arg.get("scope")
                                if "scope" in ctx_arg
                                else ctx_arg.get("agent")
                            )
                            if self.select_executor(sec_scope) is not executor:
                                return ""
                            if callable(text_src):
                                defs = self.definitions(
                                    sec_scope, ctx_arg.get("toolSet")
                                )
                                return str(text_src(defs))
                            return str(text_src)

                        sec_spec: dict[str, Any] = {
                            "name": sec["name"],
                            "order": sec["order"],
                            "text": _section_text,
                        }
                        if "interpolate" in sec:
                            sec_spec["interpolate"] = sec["interpolate"]
                        section_disposers.append(sp.section(sec_spec))
            except Exception:
                for dispose_sec in reversed(section_disposers):
                    dispose_sec()
                raise

            layer.executor = slot

            def _detach() -> None:
                slot.current = None
                for dispose_sec in reversed(section_disposers):
                    dispose_sec()

            return _detach

        return self._layers.effect(
            caller,
            _attach,
            {"label": "tools.setExecutor()", "notify": False},
        )

    def select_executor(self, scope: Any = None) -> ToolExecutor:
        """Select the nearest scoped or global ToolExecutor, failing loud if unloaded."""
        target_scope = scope if scope is not None else scope_of(self._caller_ctx())
        selected: _ExecutorSlot | None = None
        for layer in reversed(self._layers.chain_layers(target_scope)):
            if layer.executor is not None:
                selected = layer.executor
                break
        if selected is None:
            selected = self._layers.global_layer.executor
        if selected is None or selected.current is None:
            raise ToolExecutorUnavailableError()
        return selected.current

    async def execute(
        self,
        batch_or_call: ToolBatchInput | Mapping[str, Any] | ToolCallData,
        *,
        session: Any = None,
        signal: AbortSignal | None = None,
        parent: Any = None,
    ) -> Any:
        """Execute a parity ToolBatchInput or one ToolCallData."""
        if isinstance(batch_or_call, ToolCallData):
            return await self._execute_single_call(
                batch_or_call, session=session, signal=signal, parent=parent
            )
        agent = batch_or_call.get("agent")
        executor = self.select_executor(agent)
        return await executor.execute(cast(ToolBatchInput, batch_or_call))

    # ------------------------------------------------------------------
    # Single-call execution (ToolCallData) through the scoped tool registry.
    # ------------------------------------------------------------------

    async def _execute_single_call(
        self,
        tool_call: ToolCallData,
        *,
        session: Any = None,
        signal: AbortSignal | None = None,
        parent: Any = None,
    ) -> ToolResultData:
        tool = self.get(tool_call.name)
        if tool is None:
            err_res = ToolResultData(
                call_id=tool_call.call_id,
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' not found.",
                is_error=True,
            )
            self.ctx.emit("tools/result", tool_call, err_res)
            return err_res

        depth = _EXECUTION_DEPTH.get()
        parent_marker = (
            parent if parent is not None else (_NESTED_CALL if depth > 0 else None)
        )
        token = _EXECUTION_DEPTH.set(depth + 1)
        try:

            async def _dispatch(execution: dict[str, Any]) -> Any:
                returned = tool["execute"](
                    execution["call"].arguments, cast(ToolRunContext, execution)
                )
                if inspect.isawaitable(returned):
                    return await returned
                return returned

            try:
                modified_call = await self.ctx.waterfall(
                    "tools/pre-execute",
                    tool_call,
                    default_handler=lambda tc: tc,
                )
                execution = {
                    "call": modified_call,
                    "session": session,
                    "signal": signal,
                    "parent": parent_marker,
                }
                raw_result = await self.ctx.waterfall(
                    "tools/execute", execution, default_handler=_dispatch
                )
                if isinstance(raw_result, ToolResultData):
                    result = raw_result
                else:
                    formatted_result = await self.ctx.waterfall(
                        "tools/post-execute",
                        raw_result,
                        modified_call,
                        default_handler=lambda res, *args, **kwargs: res,
                    )
                    result = ToolResultData(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        content=formatted_result,
                        is_error=False,
                    )
            except Exception as exc:
                logger.exception("Error executing tool %s", tool_call.name)
                result = ToolResultData(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    content=f"Execution error: {exc!s}",
                    is_error=True,
                )

            self.ctx.emit("tools/result", tool_call, result)
            return result
        finally:
            _EXECUTION_DEPTH.reset(token)


ToolService = ToolsService
ToolRuntime = ToolsService

__all__ = [
    "ToolCallData",
    "ToolResultData",
    "ToolRuntime",
    "ToolService",
    "ToolsService",
]
