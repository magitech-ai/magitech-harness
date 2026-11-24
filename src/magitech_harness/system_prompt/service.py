"""SystemPromptService: scoped system prompt, runtime context, tools, and variable registry."""

from __future__ import annotations

import copy
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, NotRequired, TypedDict, cast

from animas import Context, Disposer, Service
from animas.core import _access_context

from magitech_harness.scope import (
    AnonymousEntries,
    NamedEntries,
    ScopedLayers,
    ScopeKey,
    ScopeLayer,
    scope_target,
    waterfall_scoped,
)

DEFAULT_HARNESS_IDENTITY = "You are an AI agent powered by Magitech Harness."
PERSONA_PREFIX_SECTION = "deployment:persona-prefix"
PERSONA_SUFFIX_SECTION = "deployment:persona-suffix"
TOOL_ORDER_REST = "<unlisted-tools>"

_VARIABLE_NAME = re.compile(r"\A[a-z][a-z0-9_]*\Z")
_GROUP_AT = re.compile(r"\A\{\{([^{}]*)\}\}")
_SNAPSHOT_HEADER = (
    "Current runtime context. This snapshot supersedes earlier "
    "runtime-context snapshots."
)

SECTION_ORDERS: dict[str, int] = {
    "HARNESS_IDENTITY": -1000,
    "DEPLOYMENT_PERSONA_PREFIX": 0,
    "PLAN_POLICY": 500,
    "TEAM_POLICY": 600,
    "PTC_ONLY": 800,
    "FILE_REFERENCE": 900,
    "TOOL_BASH": 1000,
    "TOOL_PWSH": 1010,
    "TOOL_READ": 1100,
    "TOOL_WRITE": 1200,
    "TOOL_EDIT": 1300,
    "TOOL_APPLY_PATCH": 1350,
    "TOOL_GLOB": 1400,
    "TOOL_GREP": 1500,
    "TOOL_JOBS": 1600,
    "TOOL_PTY": 1700,
    "TOOL_WEB_SEARCH": 2000,
    "TOOL_WEB_FETCH": 2100,
    "TOOL_LSP": 2200,
    "TOOL_SESSION_QUERY": 2300,
    "TOOL_GOAL": 2400,
    "TOOL_SCHEDULE": 2500,
    "TOOL_WORKFLOW": 2600,
    "TOOL_RALPH": 2700,
    "TOOL_SUBAGENT": 2800,
    "TOOL_REPORT": 2900,
    "TOOL_COMPUTER_USE": 3000,
    "MCP_SERVERS": 3100,
    "TOOLS_SDK": 5000,
    "DELIVERABLE_FILE_REFERENCES": 9000,
    "STRUCTURED_OUTPUT": 9900,
    "HARNESS_SOURCE": 10000,
    "WEB_SURFACE": 10100,
    "DEPLOYMENT_PERSONA_SUFFIX": 10200,
}

CONTEXT_ORDERS: dict[str, int] = {
    "WORKING_DIRECTORY": 100,
    "SANDBOX_POLICY": 110,
    "APPROVAL_POLICY": 115,
    "SUBAGENT_DELEGATION": 120,
    "HARNESS_SOURCE": 200,
    "WEB_SURFACE": 210,
}


PromptSectionOrderName = Literal[
    "HARNESS_IDENTITY",
    "DEPLOYMENT_PERSONA_PREFIX",
    "PLAN_POLICY",
    "TEAM_POLICY",
    "PTC_ONLY",
    "FILE_REFERENCE",
    "TOOL_BASH",
    "TOOL_PWSH",
    "TOOL_READ",
    "TOOL_WRITE",
    "TOOL_EDIT",
    "TOOL_APPLY_PATCH",
    "TOOL_GLOB",
    "TOOL_GREP",
    "TOOL_JOBS",
    "TOOL_PTY",
    "TOOL_WEB_SEARCH",
    "TOOL_WEB_FETCH",
    "TOOL_LSP",
    "TOOL_SESSION_QUERY",
    "TOOL_GOAL",
    "TOOL_SCHEDULE",
    "TOOL_WORKFLOW",
    "TOOL_RALPH",
    "TOOL_SUBAGENT",
    "TOOL_REPORT",
    "TOOL_COMPUTER_USE",
    "MCP_SERVERS",
    "TOOLS_SDK",
    "DELIVERABLE_FILE_REFERENCES",
    "STRUCTURED_OUTPUT",
    "HARNESS_SOURCE",
    "WEB_SURFACE",
    "DEPLOYMENT_PERSONA_SUFFIX",
]

PromptContextOrderName = Literal[
    "WORKING_DIRECTORY",
    "SANDBOX_POLICY",
    "APPROVAL_POLICY",
    "SUBAGENT_DELEGATION",
    "HARNESS_SOURCE",
    "WEB_SURFACE",
]


class SystemPromptConfig(TypedDict, total=False):
    """Configuration options accepted by :class:`SystemPromptService`."""

    includeHarnessIdentity: bool
    include_harness_identity: bool
    personaPrefix: str
    persona_prefix: str
    personaSuffix: str
    persona_suffix: str
    includeRuntimeContext: bool
    include_runtime_context: bool
    toolOrder: list[str]
    tool_order: list[str]


Config = SystemPromptConfig


class AssembleContext(TypedDict, total=False):
    """Per-assembly context passed through system prompt assembly."""

    scope: ScopeKey
    signal: Any


class ToolSchema(TypedDict):
    """Tool schema declaration passed to model requests."""

    name: str
    description: str
    parameters: dict[str, Any]
    deferLoading: NotRequired[bool]


class PromptSection(TypedDict):
    """Registered section contribution."""

    name: str
    order: int | float
    text: str | Callable[[dict[str, Any]], str]
    interpolate: NotRequired[bool]
    complete: NotRequired[bool]


class PromptContext(TypedDict):
    """Registered dynamic runtime-context contribution."""

    name: str
    order: int | float
    text: str | Callable[[dict[str, Any]], str]
    interpolate: NotRequired[bool]
    required: NotRequired[bool]


class AssembledSection(TypedDict):
    """Resolved prompt section prior to variable interpolation."""

    name: str
    text: str
    interpolate: NotRequired[bool]


class AssembledContext(TypedDict):
    """Resolved dynamic context entry prior to variable interpolation."""

    name: str
    text: str
    interpolate: NotRequired[bool]


class ContextSnapshotSection(TypedDict):
    """Rendered named runtime-context section."""

    name: str
    text: str


class ToolProviderResult(TypedDict):
    """Tool schemas visible for an assembly plus pre-restriction known names."""

    schemas: list[ToolSchema]
    knownNames: NotRequired[list[str]]


class PromptPresentation(TypedDict):
    """Mutable prompt presentation transformed by the assemble waterfall."""

    sections: list[AssembledSection]
    contexts: list[AssembledContext]
    variables: dict[str, str | None]


class PromptAssembly(TypedDict):
    """Completed model-facing prompt assembly."""

    sections: list[AssembledSection]
    contexts: list[AssembledContext]
    variables: dict[str, str | None]
    tools: list[ToolSchema]


def _pick_config(
    config: Mapping[str, Any], camel: str, snake: str, default: Any
) -> Any:
    if camel in config:
        return config[camel]
    if snake in config:
        return config[snake]
    return default


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _utf16_key(name: str) -> bytes:
    """Sort key matching UTF-16 code-unit order deterministically across locales."""
    return name.encode("utf-16-be")


def _validate_bool_config(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_str_config(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _validate_tool_order(tool_order: Any) -> list[str] | None:
    if tool_order is None:
        return None
    if not isinstance(tool_order, (list, tuple)) or any(
        not isinstance(item, str) for item in tool_order
    ):
        raise TypeError("toolOrder must be a sequence of strings")
    seen: set[str] = set()
    for name in tool_order:
        if name in seen:
            raise ValueError(f'toolOrder lists "{name}" more than once')
        seen.add(name)
    if TOOL_ORDER_REST not in seen:
        raise ValueError(
            f'toolOrder must contain the "{TOOL_ORDER_REST}" rest entry '
            "(where unlisted tools are inserted)"
        )
    return list(tool_order)


def _order_tools(
    tools: list[ToolSchema],
    tool_order: list[str] | None,
    known_names: set[str],
) -> list[ToolSchema]:
    for tool in tools:
        if tool["name"] == TOOL_ORDER_REST:
            raise ValueError(
                f'tool provider returned reserved tool name "{TOOL_ORDER_REST}" '
                "(reserved for toolOrder's rest entry)"
            )
    if tool_order is None:
        return sorted(tools, key=lambda tool: _utf16_key(tool["name"]))
    unknown = [
        name
        for name in tool_order
        if name != TOOL_ORDER_REST and name not in known_names
    ]
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        bad_list = ", ".join(f'"{name}"' for name in unknown)
        known_list = ", ".join(sorted(known_names, key=_utf16_key)) or "(none)"
        raise ValueError(
            f"toolOrder lists unregistered tool{plural} {bad_list}; "
            f"known tools: {known_list}"
        )
    listed = set(tool_order)
    rest = sorted(
        (tool for tool in tools if tool["name"] not in listed),
        key=lambda tool: _utf16_key(tool["name"]),
    )
    ordered: list[ToolSchema] = []
    for name in tool_order:
        if name == TOOL_ORDER_REST:
            ordered.extend(rest)
        else:
            ordered.extend(tool for tool in tools if tool["name"] == name)
    return ordered


def _interpolate(
    entry: AssembledSection | AssembledContext,
    variables: Mapping[str, str | None],
    kind: Literal["section", "context"],
) -> str:
    text = entry["text"]
    entry_name = entry["name"]
    parts: list[str] = []
    last = 0
    open_idx = text.find("{{")
    while open_idx >= 0:
        group = _GROUP_AT.match(text[open_idx:])
        if group is None:
            if text.find("}}", open_idx + 2) >= 0:
                snippet = text[open_idx : open_idx + 16]
                raise ValueError(
                    f'malformed prompt variable reference at "{snippet}…" '
                    f'in {kind} "{entry_name}" '
                    "(references are complete simple {{name}} groups)"
                )
            parts.append(text[last : open_idx + 2])
            last = open_idx + 2
            open_idx = text.find("{{", last)
            continue

        name = group.group(1)
        if not _VARIABLE_NAME.match(name):
            raise ValueError(
                f'malformed prompt variable reference "{{{{{name}}}}}" '
                f'in {kind} "{entry_name}" '
                "(variable names match /^[a-z][a-z0-9_]*$/)"
            )
        if name not in variables:
            known = ", ".join(variables.keys()) or "(none)"
            raise ValueError(
                f'unknown prompt variable "{{{{{name}}}}}" '
                f'in {kind} "{entry_name}"; registered variables: {known}'
            )
        value = variables[name]
        if value is None:
            raise ValueError(
                f'prompt variable "{{{{{name}}}}}" has no value for this '
                f'assembly ({kind} "{entry_name}")'
            )
        parts.append(text[last:open_idx])
        parts.append(value)
        last = open_idx + len(group.group(0))
        open_idx = text.find("{{", last)
    parts.append(text[last:])
    return "".join(parts)


def _resolve_context_entry(
    entry: Mapping[str, Any], context: dict[str, Any]
) -> AssembledContext:
    raw_text = entry["text"]
    resolved_text = raw_text(context) if callable(raw_text) else str(raw_text)
    assembled: AssembledContext = {"name": entry["name"], "text": resolved_text}
    if entry.get("interpolate") is not None:
        assembled["interpolate"] = entry["interpolate"]
    return assembled


def render_prompt(assembly: PromptAssembly | Mapping[str, Any]) -> str:
    """Interpolate `{{variable}}` references, drop empty sections, and join."""
    variables = assembly["variables"]
    rendered = [
        section["text"]
        if section.get("interpolate") is False
        else _interpolate(section, variables, "section")
        for section in assembly["sections"]
    ]
    return "\n\n".join(text for text in rendered if text)


def render_context_sections(
    assembly: PromptAssembly | Mapping[str, Any],
) -> list[ContextSnapshotSection]:
    """Render dynamic context contributions into named non-empty sections."""
    variables = assembly["variables"]
    out: list[ContextSnapshotSection] = []
    for entry in assembly["contexts"]:
        text = (
            entry["text"]
            if entry.get("interpolate") is False
            else _interpolate(entry, variables, "context")
        )
        if text:
            out.append({"name": entry["name"], "text": text})
    return out


def join_context_sections(sections: list[ContextSnapshotSection]) -> str:
    """Format rendered context sections into the model-facing snapshot string."""
    body = "\n\n".join(section["text"] for section in sections)
    if not body:
        return ""
    return f"{_SNAPSHOT_HEADER}\n\n{body}"


def render_context_snapshot(assembly: PromptAssembly | Mapping[str, Any]) -> str:
    """Render the complete dynamic context snapshot for model consumption."""
    return join_context_sections(render_context_sections(assembly))


class _PromptLayer(ScopeLayer):
    def __init__(self, scope: ScopeKey | None) -> None:
        self.sections: NamedEntries[dict[str, Any]] = NamedEntries(
            lambda name: ValueError(
                f'prompt section "{name}" is already registered '
                "(for a per-agent override, register through that agent's "
                "`agent.ctx` instead)"
                if scope is None
                else f'prompt section "{name}" is already registered in this scope'
            )
        )
        self.contexts: NamedEntries[dict[str, Any]] = NamedEntries(
            lambda name: ValueError(
                f'prompt context "{name}" is already registered '
                "(for a per-agent override, register through that agent's "
                "`agent.ctx` instead)"
                if scope is None
                else f'prompt context "{name}" is already registered in this scope'
            )
        )
        self.runtime_context_suppressors: AnonymousEntries[bool] = AnonymousEntries()
        self.tool_providers: AnonymousEntries[Callable[[dict[str, Any]], Any]] = (
            AnonymousEntries()
        )
        self.variables: NamedEntries[Callable[[dict[str, Any]], str | None]] = (
            NamedEntries(
                lambda name: ValueError(
                    f'prompt variable "{name}" is already registered '
                    "(for a per-agent value, register through that agent's "
                    "`agent.ctx` instead)"
                    if scope is None
                    else f'prompt variable "{name}" is already registered in this scope'
                )
            )
        )

    def is_empty(self) -> bool:
        return (
            self.sections.is_empty()
            and self.contexts.is_empty()
            and self.runtime_context_suppressors.is_empty()
            and self.tool_providers.is_empty()
            and self.variables.is_empty()
        )


class SystemPromptService(Service):
    """Scoped registry service for system prompt assembly and rendering."""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        tool_order = _validate_tool_order(
            _pick_config(cfg, "toolOrder", "tool_order", None)
        )
        include_identity = _validate_bool_config(
            _pick_config(
                cfg, "includeHarnessIdentity", "include_harness_identity", True
            ),
            "includeHarnessIdentity",
        )
        persona_prefix = _validate_str_config(
            _pick_config(cfg, "personaPrefix", "persona_prefix", ""),
            "personaPrefix",
        )
        persona_suffix = _validate_str_config(
            _pick_config(cfg, "personaSuffix", "persona_suffix", ""),
            "personaSuffix",
        )
        include_runtime_context = _validate_bool_config(
            _pick_config(cfg, "includeRuntimeContext", "include_runtime_context", True),
            "includeRuntimeContext",
        )

        super().__init__(ctx, "system_prompt")
        # Expose camelCase `systemPrompt` slot alongside `system_prompt` and bind its
        # lifetime to the service's owning fiber.
        camel_disposer = ctx.reflect.provide(
            "systemPrompt", self, getattr(type(self), "check", None)
        )
        ctx.effect(lambda: camel_disposer, label='ctx.provide("systemPrompt")')
        self._tool_order = tool_order
        self._layers: ScopedLayers[_PromptLayer] = ScopedLayers(
            lambda scope: _PromptLayer(scope),
            lambda: self.ctx.emit("system-prompt/change"),
        )
        self._legacy_disposers: dict[str, Disposer] = {}
        self._async_legacy: dict[str, Callable[[], Awaitable[str]]] = {}

        if include_identity:
            self.section(
                {
                    "name": "harness:identity",
                    "order": self.get_section_order("HARNESS_IDENTITY"),
                    "text": DEFAULT_HARNESS_IDENTITY,
                }
            )
        self.section(
            {
                "name": PERSONA_PREFIX_SECTION,
                "order": self.get_section_order("DEPLOYMENT_PERSONA_PREFIX"),
                "text": persona_prefix,
            }
        )
        self.section(
            {
                "name": PERSONA_SUFFIX_SECTION,
                "order": self.get_section_order("DEPLOYMENT_PERSONA_SUFFIX"),
                "text": persona_suffix,
            }
        )
        if not include_runtime_context:
            self.suppress_runtime_context()

    def _caller_ctx(self) -> Context:
        return _access_context.get() or self.ctx

    def get_section_order(self, name: str) -> int:
        """Resolve the numeric placement of a canonical prompt section."""
        return SECTION_ORDERS[name]

    def get_context_order(self, name: str) -> int:
        """Resolve the numeric placement of a canonical runtime-context entry."""
        return CONTEXT_ORDERS[name]

    def section(self, section: Mapping[str, Any]) -> Disposer:
        """Register an ordered prompt section in the caller's scope."""
        order = section["order"]
        name = section["name"]
        if not _is_finite_number(order):
            raise TypeError(f'prompt section "{name}" order must be a finite number')
        entry = dict(section)
        return self._layers.effect(
            self._caller_ctx(),
            lambda layer: layer.sections.insert(name, entry),
            {"label": "systemPrompt.section()"},
        )

    def context(self, context: Mapping[str, Any]) -> Disposer:
        """Register an ordered dynamic runtime context in the caller's scope."""
        order = context["order"]
        name = context["name"]
        if not _is_finite_number(order):
            raise TypeError(f'prompt context "{name}" order must be a finite number')
        entry = dict(context)
        return self._layers.effect(
            self._caller_ctx(),
            lambda layer: layer.contexts.insert(name, entry),
            {"label": "systemPrompt.context()"},
        )

    def suppress_runtime_context(self) -> Disposer:
        """Suppress optional runtime contexts in the caller's scope."""
        return self._layers.effect(
            self._caller_ctx(),
            lambda layer: layer.runtime_context_suppressors.append(True),
            {"label": "systemPrompt.suppressRuntimeContext()"},
        )

    def tools(self, provider: Callable[[dict[str, Any]], Any]) -> Disposer:
        """Register a tool-schema provider in the caller's scope."""
        return self._layers.effect(
            self._caller_ctx(),
            lambda layer: layer.tool_providers.append(provider),
            {"label": "systemPrompt.tools()"},
        )

    def variable(
        self, name: str, provider: Callable[[dict[str, Any]], str | None]
    ) -> Disposer:
        """Register a prompt variable provider in the caller's scope."""
        if not _VARIABLE_NAME.match(name):
            raise ValueError(
                f'invalid prompt variable name "{name}" '
                "(must match /^[a-z][a-z0-9_]*$/)"
            )
        return self._layers.effect(
            self._caller_ctx(),
            lambda layer: layer.variables.insert(name, provider),
            {"label": "systemPrompt.variable()"},
        )

    def _is_suppressed(self, scope: ScopeKey | None) -> bool:
        return not self._layers.global_layer.runtime_context_suppressors.is_empty() or (
            any(
                not layer.runtime_context_suppressors.is_empty()
                for layer in self._layers.chain_layers(scope)
            )
        )

    def refresh_context(
        self,
        assembly: PromptAssembly | Mapping[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PromptAssembly:
        """Refresh accepted registered runtime contexts for request admission."""
        ctx_arg = {} if context is None else context
        scope = ctx_arg.get("scope")
        providers = self._layers.merge(scope, lambda layer: layer.contexts)
        suppressed = self._is_suppressed(scope)
        required = {
            name: entry
            for name, entry in providers.items()
            if entry.get("required") is True
        }
        contexts: list[AssembledContext] = []
        for accepted in assembly["contexts"]:
            name = accepted["name"]
            provider = providers.get(name)
            required.pop(name, None)
            if suppressed and (
                provider is None or provider.get("required") is not True
            ):
                continue
            contexts.append(
                accepted
                if provider is None
                else _resolve_context_entry(provider, ctx_arg)
            )
        for provider in required.values():
            insert_at = len(contexts)
            for idx, existing in enumerate(contexts):
                next_entry = providers.get(existing["name"])
                if next_entry is not None and next_entry["order"] > provider["order"]:
                    insert_at = idx
                    break
            contexts.insert(insert_at, _resolve_context_entry(provider, ctx_arg))
        return {
            "sections": assembly["sections"],
            "contexts": contexts,
            "variables": assembly["variables"],
            "tools": assembly["tools"],
        }

    async def assemble(self, context: dict[str, Any] | None = None) -> PromptAssembly:
        """Assemble sections, contexts, tools, and variables through the waterfall."""
        ctx_arg = {} if context is None else context
        scope = ctx_arg.get("scope")
        scope_layers = self._layers.chain_layers(scope)
        suppressed = (
            not self._layers.global_layer.runtime_context_suppressors.is_empty()
            or any(
                not layer.runtime_context_suppressors.is_empty()
                for layer in scope_layers
            )
        )

        variables: dict[str, str | None] = {}
        for name, var_provider in self._layers.global_layer.variables.entries():
            variables[name] = var_provider(ctx_arg)
        for layer in scope_layers:
            for name, var_provider in layer.variables.entries():
                variables[name] = var_provider(ctx_arg)

        section_by_name = self._layers.merge(scope, lambda layer: layer.sections)
        context_by_name = self._layers.merge(scope, lambda layer: layer.contexts)

        tool_providers = [
            *self._layers.global_layer.tool_providers.values(),
            *[prov for layer in scope_layers for prov in layer.tool_providers.values()],
        ]
        collected: list[ToolSchema] = []
        known_names: set[str] = set()
        for tool_provider in tool_providers:
            result = tool_provider(ctx_arg)
            schemas: list[ToolSchema] = []
            for raw_tool in result["schemas"]:
                cloned: ToolSchema = {
                    "name": raw_tool["name"],
                    "description": raw_tool["description"],
                    "parameters": copy.deepcopy(raw_tool["parameters"]),
                }
                if raw_tool.get("deferLoading") is True:
                    cloned["deferLoading"] = True
                schemas.append(cloned)
            raw_known = result.get("knownNames")
            accepted_known = (
                raw_known
                if raw_known is not None
                else [tool["name"] for tool in schemas]
            )
            collected.extend(schemas)
            known_names.update(accepted_known)

        section_defs = sorted(
            section_by_name.values(),
            key=lambda sec: (sec["order"], _utf16_key(sec["name"])),
        )
        complete_sections = [sec for sec in section_defs if sec.get("complete") is True]
        if len(complete_sections) > 1:
            names_str = ", ".join(json.dumps(s["name"]) for s in complete_sections)
            raise ValueError(
                f"multiple complete prompt sections are active: {names_str}"
            )

        complete_section: AssembledSection | None = None
        sections: list[AssembledSection] = []
        for sec in section_defs:
            raw_text = sec["text"]
            resolved = raw_text(ctx_arg) if callable(raw_text) else str(raw_text)
            assembled_sec: AssembledSection = {"name": sec["name"], "text": resolved}
            if sec.get("interpolate") is not None:
                assembled_sec["interpolate"] = sec["interpolate"]
            if sec.get("complete") is True:
                complete_section = cast(AssembledSection, dict(assembled_sec))
            sections.append(assembled_sec)

        tools = _order_tools(collected, self._tool_order, known_names)

        ordered_contexts = sorted(
            (
                entry
                for entry in context_by_name.values()
                if not suppressed or entry.get("required") is True
            ),
            key=lambda entry: entry["order"],
        )
        presentation: PromptPresentation = {
            "sections": sections,
            "contexts": [
                _resolve_context_entry(entry, ctx_arg) for entry in ordered_contexts
            ],
            "variables": variables,
        }

        transformed = cast(
            PromptPresentation,
            await waterfall_scoped(
                self.ctx,
                scope_target(self, scope),
                "system-prompt/assemble",
                presentation,
                ctx_arg,
            ),
        )
        out_sections = (
            transformed["sections"] if complete_section is None else [complete_section]
        )
        if suppressed:
            required_names = {
                entry["name"]
                for entry in context_by_name.values()
                if entry.get("required") is True
            }
            out_contexts = [
                entry
                for entry in transformed["contexts"]
                if entry["name"] in required_names
            ]
        else:
            out_contexts = transformed["contexts"]

        return {
            "sections": out_sections,
            "contexts": out_contexts,
            "variables": transformed["variables"],
            "tools": tools,
        }

    # --- Legacy convenience bridge (pre-P3-6 callers) ---

    def register_section(
        self,
        name: str,
        section: Callable[[], Any] | str,
        *,
        order: int = 0,
    ) -> Disposer:
        """Register or replace a prompt section; supports zero-arg async callables."""
        previous = self._legacy_disposers.pop(name, None)
        if previous is not None:
            previous()
        self._async_legacy.pop(name, None)

        if inspect.iscoroutinefunction(section):
            self._async_legacy[name] = section

            def _async_text(ctx_arg: dict[str, Any]) -> str:
                return str(ctx_arg.get("_async_legacy", {}).get(name, ""))

            text_provider: str | Callable[[dict[str, Any]], str] = _async_text
        elif callable(section):

            def _sync_text(_ctx_arg: dict[str, Any]) -> str:
                return str(section())

            text_provider = _sync_text
        else:
            text_provider = str(section)

        disposer = self.section({"name": name, "order": order, "text": text_provider})

        def _cleanup() -> None:
            self._async_legacy.pop(name, None)
            if self._legacy_disposers.get(name) is _cleanup:
                del self._legacy_disposers[name]
            disposer()

        self._legacy_disposers[name] = _cleanup
        return _cleanup

    async def render(self, context: dict[str, Any] | None = None) -> str:
        """Assemble and render the system prompt string."""
        ctx_arg = {} if context is None else dict(context)
        if self._async_legacy:
            resolved_async: dict[str, str] = {}
            for name, fn in self._async_legacy.items():
                resolved_async[name] = str(await fn())
            ctx_arg["_async_legacy"] = resolved_async
        assembly = await self.assemble(ctx_arg)
        return render_prompt(assembly)

    getSectionOrder = get_section_order
    getContextOrder = get_context_order
    suppressRuntimeContext = suppress_runtime_context
    refreshContext = refresh_context


SystemPrompt = SystemPromptService
renderPrompt = render_prompt
renderContextSnapshot = render_context_snapshot
renderContextSections = render_context_sections
joinContextSections = join_context_sections
