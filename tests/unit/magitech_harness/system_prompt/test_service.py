"""Unit tests for SystemPromptService, prompt rendering, scoping, and tool order."""

from __future__ import annotations

import asyncio
import itertools
import math
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from animas import Context, EventMode

from magitech_harness.scope import Scope, ScopeKey, create_scope, scope_of
from magitech_harness.system_prompt import (
    DEFAULT_HARNESS_IDENTITY,
    PERSONA_PREFIX_SECTION,
    PERSONA_SUFFIX_SECTION,
    TOOL_ORDER_REST,
    Config,
    PromptContextOrderName,
    PromptSectionOrderName,
    SystemPromptConfig,
    SystemPromptService,
    join_context_sections,
    render_context_sections,
    render_context_snapshot,
    render_prompt,
)

BUILT_IN = (
    "harness:identity",
    PERSONA_PREFIX_SECTION,
    PERSONA_SUFFIX_SECTION,
)
from tests._mount import mount

SECTION_ORDER_NAMES = (
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
)

CONTEXT_ORDER_NAMES = (
    "WORKING_DIRECTORY",
    "SANDBOX_POLICY",
    "APPROVAL_POLICY",
    "SUBAGENT_DELEGATION",
    "HARNESS_SOURCE",
    "WEB_SURFACE",
)


def _contributed(assembly: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in assembly["sections"] if s["name"] not in BUILT_IN]


async def _mount(config: dict[str, Any] | None = None) -> Context:
    ctx = Context()
    await mount(ctx, SystemPromptService, config or {})
    await ctx.flush()
    return ctx


async def _mint_scope(ctx: Context, name: str) -> Scope:
    minted: Scope | None = None

    class _Minter:
        inject: ClassVar[list[str]] = ["systemPrompt"]

        def __init__(self, inner: Context) -> None:
            nonlocal minted
            minted = create_scope(inner, {"name": name})

    await mount(ctx, _Minter)
    assert minted is not None
    return minted


def _schema(name: str, description: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": description if description is not None else f"tool {name}",
        "parameters": {"type": "object", "properties": {}},
    }


def _scope_key(scope: Scope) -> ScopeKey:
    key = scope_of(scope.ctx)
    assert key is not None
    return key


@pytest.mark.asyncio
async def test_retains_required_context_when_optional_context_suppressed() -> None:
    """Suppression keeps required contexts and skips optional providers/additions."""
    ctx = await _mount({"includeRuntimeContext": False})
    try:
        ctx.systemPrompt.context(
            {"name": "directory", "order": 0, "required": True, "text": "Directory: /p"}
        )

        def _boom(_context: dict[str, Any]) -> str:
            raise RuntimeError("suppressed provider ran")

        ctx.systemPrompt.context({"name": "optional", "order": 1, "text": _boom})

        async def _add_unregistered(
            assembly: dict[str, Any], _context: dict[str, Any], next_fn: Any
        ) -> Any:
            assembly["contexts"].append(
                {"name": "unregistered", "text": "should stay suppressed"}
            )
            return await next_fn()

        ctx.on("system-prompt/assemble", _add_unregistered, mode=EventMode.WATERFALL)
        assembly = await ctx.systemPrompt.assemble()
        assert "Directory: /p" in render_context_snapshot(assembly)
        assert assembly["contexts"] == [{"name": "directory", "text": "Directory: /p"}]
        assert "Directory: /p" not in render_prompt(assembly)
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_refresh_context_updates_contexts_without_reassembling() -> None:
    """refreshContext re-evaluates registered providers and retains expert additions."""
    ctx = await _mount()
    try:
        current = "before"
        assemblies = 0
        tools_calls = 0
        var_calls = 0

        ctx.systemPrompt.context(
            {
                "name": "live",
                "order": 0,
                "text": lambda _ctx: current,
                "interpolate": False,
            }
        )

        def _var(_ctx: dict[str, Any]) -> str:
            nonlocal var_calls
            var_calls += 1
            return current

        def _tools(_ctx: dict[str, Any]) -> dict[str, Any]:
            nonlocal tools_calls
            tools_calls += 1
            return {
                "schemas": [
                    {
                        "name": "accepted_tool",
                        "description": current,
                        "parameters": {},
                        "deferLoading": True,
                    }
                ]
            }

        ctx.systemPrompt.variable("fixed", _var)
        ctx.systemPrompt.tools(_tools)

        async def _expert(
            _assembly: dict[str, Any], _context: dict[str, Any], next_fn: Any
        ) -> Any:
            nonlocal assemblies
            assemblies += 1
            accepted = await next_fn()
            return {
                **accepted,
                "contexts": [
                    *accepted["contexts"],
                    {"name": "expert", "text": "accepted only"},
                ],
            }

        ctx.on("system-prompt/assemble", _expert, mode=EventMode.WATERFALL)
        accepted = await ctx.systemPrompt.assemble()
        current = "after"
        refreshed = ctx.systemPrompt.refresh_context(accepted)
        assert refreshed["contexts"] == [
            {"name": "live", "text": "after", "interpolate": False},
            {"name": "expert", "text": "accepted only"},
        ]
        assert accepted["contexts"][0]["text"] == "before"
        assert refreshed["sections"] is accepted["sections"]
        assert refreshed["tools"] is accepted["tools"]
        assert refreshed["tools"][0]["description"] == "before"
        assert refreshed["variables"] is accepted["variables"]
        assert (assemblies, tools_calls, var_calls) == (1, 1, 1)
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("later", [False, True])
async def test_refresh_context_restores_missing_required_contexts(later: bool) -> None:
    """Missing required contexts are restored in registry order."""
    ctx = await _mount()
    try:
        ctx.systemPrompt.context(
            {"name": "required", "order": 0, "text": "fresh", "required": True}
        )
        ctx.systemPrompt.context({"name": "later", "order": 10, "text": "last"})
        accepted = await ctx.systemPrompt.assemble()
        accepted["contexts"] = (
            [{"name": "expert", "text": "extra"}, accepted["contexts"][1]]
            if later
            else []
        )
        assert ctx.systemPrompt.refresh_context(accepted)["contexts"] == (
            [
                {"name": "expert", "text": "extra"},
                {"name": "required", "text": "fresh"},
                {"name": "later", "text": "last"},
            ]
            if later
            else [{"name": "required", "text": "fresh"}]
        )
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_refresh_context_applies_suppression() -> None:
    """refreshContext respects active suppression without evaluating optionals."""
    ctx = await _mount()
    try:
        suppressed = False
        ctx.systemPrompt.context(
            {
                "name": "required",
                "order": 0,
                "text": lambda _c: "current",
                "required": True,
            }
        )

        def _opt(_c: dict[str, Any]) -> str:
            if suppressed:
                raise RuntimeError("suppressed context was evaluated")
            return "optional"

        ctx.systemPrompt.context({"name": "optional", "order": 10, "text": _opt})
        accepted = await ctx.systemPrompt.assemble()
        accepted["contexts"].append({"name": "expert", "text": "accepted optional"})
        restore = ctx.systemPrompt.suppress_runtime_context()
        suppressed = True
        assert ctx.systemPrompt.refresh_context(accepted)["contexts"] == [
            {"name": "required", "text": "current"}
        ]
        restore()
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_preserves_literal_context_interpolation_false() -> None:
    """Contexts with interpolate=False keep literal braces in sections and snapshot."""
    ctx = await _mount()
    try:
        ctx.systemPrompt.variable("project", lambda _c: "expanded")
        literal = "/workspace/{{project}}/{{unregistered}}/{{malformed-name}}"
        ctx.systemPrompt.context(
            {"name": "literal", "order": 0, "text": literal, "interpolate": False}
        )
        ctx.systemPrompt.context(
            {
                "name": "template",
                "order": 1,
                "text": "{{project}}",
                "interpolate": True,
            }
        )
        ctx.systemPrompt.context({"name": "default", "order": 2, "text": "{{project}}"})
        assembly = await ctx.systemPrompt.assemble()
        assert assembly["contexts"] == [
            {"name": "literal", "text": literal, "interpolate": False},
            {"name": "template", "text": "{{project}}", "interpolate": True},
            {"name": "default", "text": "{{project}}"},
        ]
        sections = render_context_sections(assembly)
        assert sections == [
            {"name": "literal", "text": literal},
            {"name": "template", "text": "expanded"},
            {"name": "default", "text": "expanded"},
        ]
        assert render_context_snapshot(assembly) == (
            "Current runtime context. This snapshot supersedes earlier "
            f"runtime-context snapshots.\n\n{literal}\n\nexpanded\n\nexpanded"
        )
        assert join_context_sections([]) == ""
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_section_and_context_orders_are_unique_integral_and_spaced() -> None:
    """Allocated section orders are unique, integral, and >=10 apart."""
    ctx = await _mount()
    orders = [ctx.systemPrompt.get_section_order(name) for name in SECTION_ORDER_NAMES]
    assert all(isinstance(order, int) for order in orders)
    assert len(set(orders)) == len(orders)
    sorted_orders = sorted(orders)
    assert all(b - a >= 10 for a, b in itertools.pairwise(sorted_orders))

    ctx_orders = [
        ctx.systemPrompt.get_context_order(name) for name in CONTEXT_ORDER_NAMES
    ]
    assert all(isinstance(order, int) for order in ctx_orders)
    assert len(set(ctx_orders)) == len(ctx_orders)


@pytest.mark.asyncio
async def test_builtin_sections_identity_persona_and_direct_ctor() -> None:
    """Built-in sections support personas, omission, and direct construction."""
    ctx = await _mount({"personaPrefix": "You are Custom Agent."})
    assembly = await ctx.systemPrompt.assemble()
    assert [s["name"] for s in assembly["sections"]] == list(BUILT_IN)
    assert render_prompt(assembly) == (
        f"{DEFAULT_HARNESS_IDENTITY}\n\nYou are Custom Agent."
    )

    with pytest.raises(ValueError, match="is already registered"):
        ctx.systemPrompt.section(
            {"name": PERSONA_PREFIX_SECTION, "order": 0, "text": "imposter"}
        )

    # Empty defaults omit persona text.
    empty_ctx = await _mount()
    assert (
        render_prompt(await empty_ctx.systemPrompt.assemble())
        == DEFAULT_HARNESS_IDENTITY
    )

    # includeHarnessIdentity=False omits harness:identity.
    no_id_ctx = await _mount(
        {
            "includeHarnessIdentity": False,
            "personaPrefix": "Helpful engineer.",
            "personaSuffix": "Done.",
        }
    )
    no_id_assembly = await no_id_ctx.systemPrompt.assemble()
    assert [s["name"] for s in no_id_assembly["sections"]] == [
        PERSONA_PREFIX_SECTION,
        PERSONA_SUFFIX_SECTION,
    ]
    assert render_prompt(no_id_assembly) == "Helpful engineer.\n\nDone."

    # Direct construction tolerates omitted config.
    direct_ctx = Context()
    service = SystemPromptService(direct_ctx)
    assert render_prompt(await service.assemble()) == DEFAULT_HARNESS_IDENTITY


@pytest.mark.asyncio
async def test_assembles_sections_contexts_tools_and_variables_deterministically() -> (
    None
):
    """Sections, contexts, and tools sort canonically with UTF-16 code-unit ties."""
    ctx = await _mount({"personaPrefix": "Persona."})
    ctx.systemPrompt.section(
        {"name": "cwd", "order": 20, "text": lambda _c: "cwd: /tmp"}
    )
    ctx.systemPrompt.section({"name": "rules", "order": 10, "text": "Be precise."})
    ctx.systemPrompt.context(
        {"name": "later", "order": 20, "text": lambda _c: "context 2"}
    )
    ctx.systemPrompt.context({"name": "earlier", "order": 10, "text": "context 1"})
    ctx.systemPrompt.tools(
        lambda _c: {
            "schemas": [{"name": "echo", "description": "echo back", "parameters": {}}]
        }
    )

    assembly = await ctx.systemPrompt.assemble()
    assert [s["name"] for s in assembly["sections"]] == [
        "harness:identity",
        PERSONA_PREFIX_SECTION,
        "rules",
        "cwd",
        PERSONA_SUFFIX_SECTION,
    ]
    assert [s["text"] for s in assembly["sections"]] == [
        DEFAULT_HARNESS_IDENTITY,
        "Persona.",
        "Be precise.",
        "cwd: /tmp",
        "",
    ]
    assert assembly["contexts"] == [
        {"name": "earlier", "text": "context 1"},
        {"name": "later", "text": "context 2"},
    ]
    assert assembly["tools"] == [
        {"name": "echo", "description": "echo back", "parameters": {}}
    ]
    assert assembly["variables"] == {}
    assert render_prompt(assembly) == (
        f"{DEFAULT_HARNESS_IDENTITY}\n\nPersona.\n\nBe precise.\n\ncwd: /tmp"
    )
    assert render_context_snapshot(assembly) == (
        "Current runtime context. This snapshot supersedes earlier "
        "runtime-context snapshots.\n\ncontext 1\n\ncontext 2"
    )

    # Equal orders break deterministically by code-unit order.
    for names in (("äther", "zeta"), ("zeta", "äther")):
        tie_ctx = await _mount()
        for name in names:
            tie_ctx.systemPrompt.section({"name": name, "order": 10, "text": name})
        assert [
            s["name"] for s in _contributed(await tie_ctx.systemPrompt.assemble())
        ] == ["zeta", "äther"]


@pytest.mark.asyncio
async def test_resolves_dynamic_section_text_with_caller_context() -> None:
    """Dynamic sections receive AssembleContext on each assemble call."""
    ctx = await _mount()
    calls = 0

    def _dynamic(context: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return f"call {calls} for {context.get('who', 'nobody')}"

    ctx.systemPrompt.section({"name": "dynamic", "order": 0, "text": _dynamic})
    assert _contributed(await ctx.systemPrompt.assemble({"who": "alice"}))[0][
        "text"
    ] == ("call 1 for alice")
    assert _contributed(await ctx.systemPrompt.assemble())[0]["text"] == (
        "call 2 for nobody"
    )


@pytest.mark.asyncio
async def test_removes_contributions_on_fiber_dispose() -> None:
    """Disposing a contributing fiber removes sections, contexts, tools, and vars."""
    ctx = await _mount()

    class _Contributor:
        inject: ClassVar[list[str]] = ["systemPrompt"]

        def __init__(self, inner: Context) -> None:
            inner.systemPrompt.section(
                {"name": "scoped", "order": 0, "text": "scoped section"}
            )
            inner.systemPrompt.context(
                {"name": "scoped-context", "order": 0, "text": "scoped context"}
            )
            inner.systemPrompt.tools(
                lambda _c: {
                    "schemas": [
                        {"name": "scoped-tool", "description": "", "parameters": {}}
                    ]
                }
            )
            inner.systemPrompt.variable("scoped_var", lambda _c: "v")

    fiber = await mount(ctx, _Contributor)
    await ctx.flush()
    before = await ctx.systemPrompt.assemble()
    assert len(_contributed(before)) == 1
    assert len(before["contexts"]) == 1
    assert before["variables"] == {"scoped_var": "v"}

    await fiber.dispose()
    after = await ctx.systemPrompt.assemble()
    assert len(_contributed(after)) == 0
    assert len(after["contexts"]) == 0
    assert [s["name"] for s in after["sections"]] == list(BUILT_IN)
    assert after["tools"] == []
    assert after["variables"] == {}


@pytest.mark.asyncio
async def test_validation_and_change_listener_rollback() -> None:
    """Rejects duplicates/non-finite orders and rolls back on change error."""
    ctx = await _mount()

    # Duplicate section / context / bad order
    ctx.systemPrompt.section({"name": "dup", "order": 0, "text": "first"})
    with pytest.raises(ValueError, match="already registered"):
        ctx.systemPrompt.section({"name": "dup", "order": 1, "text": "second"})
    with pytest.raises(TypeError, match="order must be a finite number"):
        ctx.systemPrompt.section({"name": "bad", "order": math.nan, "text": "x"})
    with pytest.raises(TypeError, match="order must be a finite number"):
        ctx.systemPrompt.section({"name": "bad_bool", "order": True, "text": "x"})  # type: ignore[typeddict-item]

    ctx.systemPrompt.context({"name": "policy", "order": 1, "text": "first"})
    with pytest.raises(ValueError, match="already registered"):
        ctx.systemPrompt.context({"name": "policy", "order": 2, "text": "second"})
    with pytest.raises(TypeError, match="order must be a finite number"):
        ctx.systemPrompt.context({"name": "bad-ctx", "order": math.inf, "text": "x"})

    # Rollback when system-prompt/change listener throws (section, tool, variable)
    change_count = 0

    def _on_change() -> None:
        nonlocal change_count
        change_count += 1

    off_count = ctx.on("system-prompt/change", _on_change)
    disp_t = ctx.systemPrompt.tools(lambda _c: {"schemas": []})
    disp_c = ctx.systemPrompt.context({"name": "c2", "order": 2, "text": "x"})
    assert change_count == 2
    disp_t()
    disp_c()
    assert change_count == 4
    off_count()

    # Rollback cleanly when the Context.emit seam raises on change broadcast.
    threw = False

    def _raise_once(*_args: Any, **_kwargs: Any) -> None:
        nonlocal threw
        if not threw:
            threw = True
            raise RuntimeError("boom change listener")

    with (
        patch.object(ctx, "emit", side_effect=_raise_once),
        pytest.raises(RuntimeError, match="boom change listener"),
    ):
        ctx.systemPrompt.section({"name": "p", "order": 0, "text": "persona"})
    assert [s["name"] for s in _contributed(await ctx.systemPrompt.assemble())] == [
        "dup"
    ]

    threw = False
    with (
        patch.object(ctx, "emit", side_effect=_raise_once),
        pytest.raises(RuntimeError, match="boom change listener"),
    ):
        ctx.systemPrompt.tools(lambda _c: {"schemas": [_schema("t")]})
    assert (await ctx.systemPrompt.assemble())["tools"] == []

    threw = False
    with (
        patch.object(ctx, "emit", side_effect=_raise_once),
        pytest.raises(RuntimeError, match="boom change listener"),
    ):
        ctx.systemPrompt.variable("v", lambda _c: "x")
    assert (await ctx.systemPrompt.assemble())["variables"] == {}


@pytest.mark.asyncio
async def test_waterfall_complete_section_and_immutability() -> None:
    """Waterfall order, short-circuit, complete section restoration, immutability."""
    ctx = await _mount()
    ctx.systemPrompt.section({"name": "base", "order": 10, "text": "base"})

    seen_contexts: list[dict[str, Any]] = []
    seen_names: list[list[str]] = []

    async def _listener_a(
        assembly: dict[str, Any], context: dict[str, Any], next_fn: Any
    ) -> Any:
        seen_contexts.append(context)
        assembly["sections"].append({"name": "from-a", "text": "a"})
        return await next_fn()

    async def _listener_b(
        assembly: dict[str, Any], _context: dict[str, Any], next_fn: Any
    ) -> Any:
        seen_names.append([s["name"] for s in assembly["sections"]])
        return await next_fn()

    off_a = ctx.on("system-prompt/assemble", _listener_a, mode=EventMode.WATERFALL)
    off_b = ctx.on("system-prompt/assemble", _listener_b, mode=EventMode.WATERFALL)
    passed: dict[str, Any] = {}
    assembly = await ctx.systemPrompt.assemble(passed)
    assert seen_names == [
        [
            "harness:identity",
            PERSONA_PREFIX_SECTION,
            "base",
            PERSONA_SUFFIX_SECTION,
            "from-a",
        ]
    ]
    assert [s["name"] for s in assembly["sections"]] == seen_names[0]
    assert seen_contexts[0] is passed
    off_a()
    off_b()

    # Short-circuit listener
    async def _short_circuit(
        _assembly: dict[str, Any], _context: dict[str, Any], _next: Any
    ) -> Any:
        return {"sections": [], "contexts": [], "variables": {}}

    off_sc = ctx.on("system-prompt/assemble", _short_circuit, mode=EventMode.WATERFALL)
    assert (await ctx.systemPrompt.assemble())["sections"] == []
    off_sc()

    # Complete section restores exact text; multiple complete sections throw.
    d1 = ctx.systemPrompt.section(
        {"name": "complete", "order": 5, "text": "Exact prompt.", "complete": True}
    )

    async def _mutate_complete(
        assembly: dict[str, Any], _context: dict[str, Any], next_fn: Any
    ) -> Any:
        target = next(s for s in assembly["sections"] if s["name"] == "complete")
        target["text"] = "mutated"
        assembly["sections"].append({"name": "late", "text": "late"})
        return await next_fn()

    off_mut = ctx.on(
        "system-prompt/assemble", _mutate_complete, mode=EventMode.WATERFALL
    )
    assert (await ctx.systemPrompt.assemble())["sections"] == [
        {"name": "complete", "text": "Exact prompt."}
    ]
    off_mut()

    d2 = ctx.systemPrompt.section(
        {"name": "second", "order": 20, "text": "second", "complete": True}
    )
    with pytest.raises(
        ValueError,
        match='multiple complete prompt sections are active: "complete", "second"',
    ):
        await ctx.systemPrompt.assemble()
    d1()
    d2()

    # Immutability: caller mutations on assembled snapshots do not leak.
    ctx.systemPrompt.tools(lambda _c: {"schemas": [_schema("t", "tool")]})
    first = await ctx.systemPrompt.assemble()
    first["sections"][0]["name"] = "mutated"
    first["sections"][0]["text"] = "mutated"
    first["contexts"].append({"name": "mutated", "text": "mutated"})
    first["tools"][0]["description"] = "mutated"
    first["tools"][0]["parameters"]["properties"]["leak"] = {"type": "string"}

    second = await ctx.systemPrompt.assemble()
    assert second["sections"][0]["text"] == DEFAULT_HARNESS_IDENTITY
    assert second["contexts"] == []
    assert second["tools"] == [_schema("t", "tool")]


@pytest.mark.asyncio
async def test_prompt_variables_and_interpolation_diagnostics() -> None:
    """Variables, interpolation, non-rescanning, and exact error diagnostics."""
    ctx = await _mount({"personaPrefix": "You run on {{model}} in {{cwd}}."})
    changes = 0
    ctx.on("system-prompt/change", lambda: _inc())

    def _inc() -> None:
        nonlocal changes
        changes += 1

    disp = ctx.systemPrompt.variable("who", lambda c: c.get("who"))
    assert changes == 1
    assert (await ctx.systemPrompt.assemble({"who": "alice"}))["variables"] == {
        "who": "alice"
    }
    assert (await ctx.systemPrompt.assemble())["variables"] == {"who": None}
    disp()
    assert changes == 2

    # Live iteration within same assemble call.
    added = False

    vctx = await _mount()

    def _first(_c: dict[str, Any]) -> str:
        nonlocal added
        if not added:
            added = True
            vctx.systemPrompt.variable("late", lambda _c2: "second value")
        return "first value"

    vctx.systemPrompt.variable("first", _first)
    assert (await vctx.systemPrompt.assemble())["variables"] == {
        "first": "first value",
        "late": "second value",
    }

    # Invalid/duplicate variable rejection
    ctx.systemPrompt.variable("model", lambda _c: "m1")
    with pytest.raises(
        ValueError, match='prompt variable "model" is already registered'
    ):
        ctx.systemPrompt.variable("model", lambda _c: "m2")
    with pytest.raises(ValueError, match='invalid prompt variable name "Not Valid"'):
        ctx.systemPrompt.variable("Not Valid", lambda _c: "x")

    ctx.systemPrompt.variable("cwd", lambda _c: "/work")
    assert render_prompt(await ctx.systemPrompt.assemble()) == (
        f"{DEFAULT_HARNESS_IDENTITY}\n\nYou run on m1 in /work."
    )

    # interpolate=False with complete and dynamic combinations
    for complete in (False, True):
        for dynamic in (False, True):
            lctx = await _mount(
                {"includeHarnessIdentity": False, "personaPrefix": "{{model}}"}
            )
            lctx.systemPrompt.variable("model", lambda _c: "actual-model")
            raw = "{{item}} {{model}} {{ model }} {{nested{{item}}}}"
            lctx.systemPrompt.section(
                {
                    "name": "literal",
                    "order": 1,
                    "text": (lambda _c, r=raw: r) if dynamic else raw,
                    "interpolate": False,
                    "complete": complete,
                }
            )
            rendered = render_prompt(await lctx.systemPrompt.assemble())
            assert rendered == (raw if complete else f"actual-model\n\n{raw}")

    # Error diagnostics
    with pytest.raises(
        ValueError,
        match='unknown prompt variable "{{x}}" in section "s"; '
        r"registered variables: \(none\)",
    ):
        render_prompt(
            {
                "sections": [{"name": "s", "text": "{{x}}"}],
                "contexts": [],
                "tools": [],
                "variables": {},
            }
        )

    with pytest.raises(
        ValueError,
        match='unknown prompt variable "{{modle}}" in section "persona"; '
        "registered variables: model",
    ):
        render_prompt(
            {
                "sections": [{"name": "persona", "text": "on {{modle}}"}],
                "contexts": [],
                "tools": [],
                "variables": {"model": "m"},
            }
        )

    with pytest.raises(
        ValueError,
        match='prompt variable "{{cwd}}" has no value for this assembly '
        r'\(section "persona"\)',
    ):
        render_prompt(
            {
                "sections": [{"name": "persona", "text": "in {{cwd}}"}],
                "contexts": [],
                "tools": [],
                "variables": {"cwd": None},
            }
        )

    with pytest.raises(
        ValueError,
        match='malformed prompt variable reference "{{ model }}" in section "s"',
    ):
        render_prompt(
            {
                "sections": [{"name": "s", "text": "on {{ model }}"}],
                "contexts": [],
                "tools": [],
                "variables": {"model": "m"},
            }
        )

    # Lone {{ without later }} stays literal; substituted {{}} is not re-scanned.
    assert (
        render_prompt(
            {
                "sections": [{"name": "s", "text": "shell ${X:-{{fallback} stays"}],
                "contexts": [],
                "tools": [],
                "variables": {},
            }
        )
        == "shell ${X:-{{fallback} stays"
    )
    assert (
        render_prompt(
            {
                "sections": [{"name": "s", "text": "v = {{model}}!"}],
                "contexts": [],
                "tools": [],
                "variables": {"model": "literal {{sneaky}} inside"},
            }
        )
        == "v = literal {{sneaky}} inside!"
    )

    for mangled in ("{{{model}}}", "x {{a{b}} y {{model}}"):
        with pytest.raises(ValueError, match="malformed prompt variable reference at "):
            render_prompt(
                {
                    "sections": [{"name": "s", "text": mangled}],
                    "contexts": [],
                    "tools": [],
                    "variables": {"model": "m"},
                }
            )


@pytest.mark.asyncio
async def test_tool_order_sorting_validation_and_errors() -> None:
    """toolOrder validates at init and assembly, inserting unlisted at rest marker."""
    assert TOOL_ORDER_REST == "<unlisted-tools>"

    ctx = await _mount()
    ctx.systemPrompt.tools(
        lambda _c: {"schemas": [_schema("charlie"), _schema("alpha")]}
    )
    ctx.systemPrompt.tools(lambda _c: {"schemas": [_schema("bravo")]})
    assert [t["name"] for t in (await ctx.systemPrompt.assemble())["tools"]] == [
        "alpha",
        "bravo",
        "charlie",
    ]

    ordered_ctx = await _mount({"toolOrder": ["todo_write", TOOL_ORDER_REST, "bash"]})
    ordered_ctx.systemPrompt.tools(
        lambda _c: {
            "schemas": [
                _schema("bash"),
                _schema("echo_b"),
                _schema("todo_write"),
                _schema("echo_a"),
            ]
        }
    )
    assert [
        t["name"] for t in (await ordered_ctx.systemPrompt.assemble())["tools"]
    ] == [
        "todo_write",
        "echo_a",
        "echo_b",
        "bash",
    ]

    # Restricted away in knownNames is fine; unregistered throws.
    restricted_ctx = await _mount({"toolOrder": ["bash", TOOL_ORDER_REST]})
    restricted_ctx.systemPrompt.tools(
        lambda _c: {"schemas": [_schema("read")], "knownNames": ["read", "bash"]}
    )
    assert [
        t["name"] for t in (await restricted_ctx.systemPrompt.assemble())["tools"]
    ] == ["read"]

    bad_ctx = await _mount({"toolOrder": ["basj", TOOL_ORDER_REST]})
    bad_ctx.systemPrompt.tools(
        lambda _c: {"schemas": [_schema("read")], "knownNames": ["read", "bash"]}
    )
    with pytest.raises(
        ValueError,
        match='toolOrder lists unregistered tool "basj"; known tools: bash, read',
    ):
        await bad_ctx.systemPrompt.assemble()

    none_known_ctx = await _mount({"toolOrder": ["ghost", TOOL_ORDER_REST]})
    with pytest.raises(
        ValueError,
        match=r'toolOrder lists unregistered tool "ghost"; known tools: \(none\)',
    ):
        await none_known_ctx.systemPrompt.assemble()

    reserved_ctx = await _mount()
    reserved_ctx.systemPrompt.tools(lambda _c: {"schemas": [_schema(TOOL_ORDER_REST)]})
    with pytest.raises(
        ValueError,
        match=f'tool provider returned reserved tool name "{TOOL_ORDER_REST}"',
    ):
        await reserved_ctx.systemPrompt.assemble()

    # Load-time validation rejects missing rest entry or duplicates
    for bad_order, pat in (
        ([], f'must contain the "{TOOL_ORDER_REST}" rest entry'),
        (["bash", "todo_write"], f'must contain the "{TOOL_ORDER_REST}" rest entry'),
        (["bash", "bash", TOOL_ORDER_REST], 'lists "bash" more than once'),
        ([TOOL_ORDER_REST, "bash", TOOL_ORDER_REST], f'lists "{TOOL_ORDER_REST}" more'),
    ):
        fail_ctx = Context()
        fib = fail_ctx.use(SystemPromptService, {"toolOrder": bad_order})
        await fail_ctx.flush()
        with pytest.raises(ValueError, match=pat):
            await fib.await_settle()


@pytest.mark.asyncio
async def test_scoped_sections_variables_contexts_tools_and_assemble_listener() -> None:
    """Scoped overlays shadow globals, respect parent chains, and clean on dispose."""
    ctx = await _mount({"personaPrefix": "You are the deployment."})
    scope = await _mint_scope(ctx, "child")
    key = _scope_key(scope)

    global_calls = 0

    def _global_text(_c: dict[str, Any]) -> str:
        nonlocal global_calls
        global_calls += 1
        return "global"

    scope.ctx.systemPrompt.section(
        {"name": PERSONA_PREFIX_SECTION, "order": 0, "text": "You run tests."}
    )
    ctx.systemPrompt.section({"name": "shared", "order": 1, "text": _global_text})
    scope.ctx.systemPrompt.section(
        {"name": "shared", "order": 1, "text": lambda _c: "scoped"}
    )

    scoped_rendered = render_prompt(await ctx.systemPrompt.assemble({"scope": key}))
    assert "You run tests." in scoped_rendered
    assert "You are the deployment." not in scoped_rendered
    assert global_calls == 0  # global shadowed before evaluation for scoped call
    global_rendered = render_prompt(await ctx.systemPrompt.assemble())
    assert "You are the deployment." in global_rendered
    assert "You run tests." not in global_rendered
    assert global_calls == 1

    # Duplicate messages distinguish global vs scoped
    ctx.systemPrompt.section({"name": "x", "order": 1, "text": "a"})
    with pytest.raises(ValueError, match=r"agent\.ctx"):
        ctx.systemPrompt.section({"name": "x", "order": 1, "text": "b"})
    with pytest.raises(ValueError, match="already registered in this scope"):
        scope.ctx.systemPrompt.section({"name": "shared", "order": 1, "text": "b"})

    # Scoped assemble waterfall listener affects only its scope
    shaped: list[Any] = []

    async def _shape(
        assembly: dict[str, Any], context: dict[str, Any], next_fn: Any
    ) -> Any:
        shaped.append(context.get("scope"))
        result = await next_fn()
        result["sections"].append({"name": "listener:extra", "text": "listener text"})
        return result

    scope.ctx.on("system-prompt/assemble", _shape, mode=EventMode.WATERFALL)
    scoped_after = await ctx.systemPrompt.assemble({"scope": key})
    global_after = await ctx.systemPrompt.assemble()
    assert any(s["name"] == "listener:extra" for s in scoped_after["sections"])
    assert not any(s["name"] == "listener:extra" for s in global_after["sections"])
    assert len(shaped) == 1

    # Scoped context & suppression + refresh
    current_ctx = "before"
    ctx.systemPrompt.context({"name": "fact", "order": 0, "text": "global"})
    scope.ctx.systemPrompt.context(
        {
            "name": "fact",
            "order": 0,
            "required": True,
            "text": lambda _c: current_ctx,
        }
    )
    scope.ctx.systemPrompt.context({"name": "optional", "order": 10, "text": "opt"})
    accepted = await ctx.systemPrompt.assemble({"scope": key})
    current_ctx = "after"
    restore = scope.ctx.systemPrompt.suppress_runtime_context()
    assert ctx.systemPrompt.refresh_context(accepted, {"scope": key})["contexts"] == [
        {"name": "fact", "text": "after"}
    ]
    restore()

    # Scoped variable self-dispose defers replacement to next generation
    calls: list[str] = []
    scope.ctx.systemPrompt.section(
        {"name": "use_mode", "order": 2, "text": "Mode={{mode}}"}
    )
    disp_mode: list[Any] = []

    def _first_mode(_c: dict[str, Any]) -> str:
        calls.append("first")
        disp_mode[0]()
        scope.ctx.systemPrompt.variable("mode", lambda _c2: _second_mode())
        return "first"

    def _second_mode() -> str:
        calls.append("replacement")
        return "replacement"

    disp_mode.append(scope.ctx.systemPrompt.variable("mode", _first_mode))
    assert "Mode=first" in render_prompt(
        await ctx.systemPrompt.assemble({"scope": key})
    )
    assert calls == ["first"]
    assert "Mode=replacement" in render_prompt(
        await ctx.systemPrompt.assemble({"scope": key})
    )
    assert calls == ["first", "replacement"]

    # Disposal removes all scoped overlays
    await scope.dispose()
    assert "You are the deployment." in render_prompt(
        await ctx.systemPrompt.assemble({"scope": key})
    )


@pytest.mark.asyncio
async def test_legacy_register_section_and_render_bridge() -> None:
    """Legacy register_section/render helpers continue functioning cleanly."""
    ctx = Context()
    service = SystemPromptService(ctx, {"includeHarnessIdentity": False})

    service.register_section("header", "# System Persona: Magitech AI")
    service.register_section("time", lambda: "Time: 2026")

    async def _env() -> str:
        await asyncio.sleep(0.001)
        return "OS: Linux"

    service.register_section("env", _env)
    rendered = await service.render()
    assert "# System Persona: Magitech AI" in rendered
    assert "Time: 2026" in rendered
    assert "OS: Linux" in rendered

    # Re-registering replaces rather than throwing
    disp = service.register_section("header", "Replaced")
    assert "Replaced" in await service.render()
    assert "# System Persona: Magitech AI" not in await service.render()
    disp()


@pytest.mark.asyncio
async def test_snake_case_config_empty_context_and_legacy_cleanup() -> None:
    """Snake_case config keys work, empty contexts drop, and old legacy disposer is safe."""
    ctx = Context()
    service = SystemPromptService(ctx, {"include_harness_identity": False})

    # Sync legacy only (no async sections): render skips async path cleanly
    old_disp = service.register_section("slot", "First")
    new_disp = service.register_section("slot", "Second")
    old_disp()
    assert await service.render() == "Second"
    new_disp()
    assert await service.render() == ""

    # Empty dynamic context string is dropped from rendered context sections
    service.context({"name": "empty_ctx", "order": 1, "text": lambda _c: ""})
    service.context({"name": "real_ctx", "order": 2, "text": lambda _c: "Present"})
    assembled = await service.assemble()
    assert render_context_sections(assembled) == [
        {"name": "real_ctx", "text": "Present"}
    ]


@pytest.mark.asyncio
async def test_config_validation_teardown_and_optional_edge_cases() -> None:
    """Config validates before registration, teardown clears slots, and optional None keys work."""
    # Type exports exist and match expected type aliases
    assert Config is SystemPromptConfig
    sec_order_name: PromptSectionOrderName = "HARNESS_IDENTITY"
    ctx_order_name: PromptContextOrderName = "WORKING_DIRECTORY"
    assert sec_order_name == "HARNESS_IDENTITY"
    assert ctx_order_name == "WORKING_DIRECTORY"

    # Invalid config types fail before registering systemPrompt
    bad_configs: list[tuple[dict[str, Any], str]] = [
        ({"includeHarnessIdentity": "yes"}, "includeHarnessIdentity must be a boolean"),
        ({"includeRuntimeContext": 1}, "includeRuntimeContext must be a boolean"),
        ({"personaPrefix": 123}, "personaPrefix must be a string"),
        ({"personaSuffix": ["bad"]}, "personaSuffix must be a string"),
        ({"toolOrder": "bash"}, "toolOrder must be a sequence of strings"),
        (
            {"toolOrder": [1, TOOL_ORDER_REST]},
            "toolOrder must be a sequence of strings",
        ),
    ]
    for bad_cfg, pattern in bad_configs:
        bad_ctx = Context()
        with pytest.raises(TypeError, match=pattern):
            SystemPromptService(bad_ctx, bad_cfg)
        assert "systemPrompt" not in bad_ctx.reflect.store
        assert "system_prompt" not in bad_ctx

    # Fiber disposal removes camelCase `systemPrompt` as well as `system_prompt`
    mounted_ctx = Context()
    fiber = mounted_ctx.use(SystemPromptService, {})
    await mounted_ctx.flush()
    assert "systemPrompt" in mounted_ctx.reflect.store
    await fiber.dispose()
    assert "systemPrompt" not in mounted_ctx.reflect.store
    assert "system_prompt" not in mounted_ctx.reflect.store
    assert mounted_ctx.get("systemPrompt", strict=False) is None
    with pytest.raises(AttributeError):
        _ = mounted_ctx.systemPrompt
    with pytest.raises(AttributeError):
        _ = mounted_ctx.system_prompt

    # Multiple unknown tools trigger plural error formatting
    plural_ctx = await _mount({"toolOrder": ["ghost_a", "ghost_b", TOOL_ORDER_REST]})
    with pytest.raises(
        ValueError,
        match='toolOrder lists unregistered tools "ghost_a", "ghost_b"',
    ):
        await plural_ctx.systemPrompt.assemble()

    # None optional keys (interpolate / knownNames) behave like omitted keys,
    # mid-assemble tool registration defers to next assemble, and 1-arg next_fn works.
    edge_ctx = await _mount({"includeHarnessIdentity": False})
    edge_ctx.systemPrompt.variable("name", lambda _c: "world")
    edge_ctx.systemPrompt.tools(
        lambda _c: {"schemas": [_schema("initial_tool")], "knownNames": None}
    )

    registered_mid_turn = False

    def _dynamic_section(_c: dict[str, Any]) -> str:
        nonlocal registered_mid_turn
        if not registered_mid_turn:
            registered_mid_turn = True
            edge_ctx.systemPrompt.tools(
                lambda _inner: {"schemas": [_schema("late_tool")]}
            )
        return "hi {{name}}"

    edge_ctx.systemPrompt.section(
        {"name": "s1", "order": 1, "text": _dynamic_section, "interpolate": None}
    )
    edge_ctx.systemPrompt.context(
        {"name": "c1", "order": 1, "text": "ctx {{name}}", "interpolate": None}
    )

    async def _one_arg_waterfall(
        presentation: dict[str, Any], _context: dict[str, Any], next_fn: Any
    ) -> Any:
        updated = dict(presentation)
        updated["sections"] = [
            *presentation["sections"],
            {"name": "from-1arg", "text": "ok"},
        ]
        return await next_fn(updated)

    edge_ctx.on("system-prompt/assemble", _one_arg_waterfall, mode=EventMode.WATERFALL)

    first = await edge_ctx.systemPrompt.assemble()
    assert [t["name"] for t in first["tools"]] == ["initial_tool"]
    assert "from-1arg" in [s["name"] for s in first["sections"]]
    assert render_prompt(first) == "hi world\n\nok"
    assert render_context_sections(first) == [{"name": "c1", "text": "ctx world"}]

    second = await edge_ctx.systemPrompt.assemble()
    assert [t["name"] for t in second["tools"]] == ["initial_tool", "late_tool"]
