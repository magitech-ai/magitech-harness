"""Unit tests for scoped contexts and scope-filtered event routing."""

from __future__ import annotations

import asyncio
import gc
import weakref
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from anima import Context, EventMode, Service
from anima.core import _access_context

from magitech_harness.scope import (
    CreateScopeOptions,
    Scope,
    ScopeKey,
    bind_scope_parent,
    carrier_key_of,
    create_scope,
    emit_scoped,
    is_scope_carrier,
    scope_chain_of,
    scope_of,
    scope_parent_of,
    scope_target,
    waterfall_scoped,
)


async def _mint_scope(
    ctx: Context,
    key: ScopeKey,
    options: CreateScopeOptions | dict[str, Any] | None = None,
) -> Scope:
    """Mount a helper component and mint a scope inside it."""
    minted: Scope | None = None

    def _mount(inner: Context) -> None:
        nonlocal minted
        minted = create_scope(inner, key, options)

    await ctx.use(_mount)
    assert minted is not None
    return minted


@pytest.mark.asyncio
async def test_create_scope_tags_contexts_with_nearest_tag_winning() -> None:
    """Nested scopes tag contexts and derived contexts with nearest winning."""
    ctx = Context()
    outer_key = {"name": "outer"}
    inner_key = {"name": "inner"}
    outer = await _mint_scope(ctx, outer_key)
    inner = create_scope(outer.ctx, inner_key)

    assert scope_of(ctx) is None
    assert scope_of(outer.ctx) is outer_key
    assert scope_of(outer.ctx.extend({})) is outer_key
    assert scope_of(outer.ctx.filter(lambda _c: True)) is outer_key
    assert scope_of(outer.ctx.isolate("svc")) is outer_key
    assert scope_of(outer.ctx.intercept("svc", {})) is outer_key
    assert scope_of(inner.ctx) is inner_key
    assert scope_of(inner.ctx.extend({"fiber": inner.ctx.fiber})) is inner_key

    await inner.dispose()
    await outer.dispose()


@pytest.mark.asyncio
async def test_create_scope_usable_synchronously_before_backing_fiber_activates() -> (
    None
):
    """Effects registered before the backing fiber activates clean up on dispose."""
    ctx = Context()
    events: list[str] = []
    scope: Scope | None = None

    def _setup(inner: Context) -> None:
        nonlocal scope
        scope = create_scope(inner, {"name": "sync"})
        scope.ctx.effect(lambda: lambda: events.append("disposed"))
        events.append("registered")

    await ctx.use(_setup)
    assert scope is not None
    assert events == ["registered"]
    await scope.dispose()
    assert events == ["registered", "disposed"]


@pytest.mark.asyncio
async def test_create_scope_shares_quiescence_across_repeat_and_raw_dispose() -> None:
    """Repeat and raw-disposer-first calls share completion quiescence."""
    ctx = Context()
    scope = await _mint_scope(ctx, {"name": "quiescence"})
    loop = asyncio.get_running_loop()
    gate: asyncio.Future[None] = loop.create_future()
    finished = False

    async def _async_cleanup() -> None:
        nonlocal finished
        await gate
        finished = True

    scope.ctx.effect(lambda: _async_cleanup)

    raw_task = asyncio.create_task(scope.raw_dispose())
    public_task = asyncio.create_task(scope.dispose())
    await asyncio.sleep(0)
    assert finished is False
    gate.set_result(None)
    await asyncio.gather(raw_task, public_task, scope.dispose())
    assert finished is True


@pytest.mark.asyncio
async def test_create_scope_exposes_raw_disposer_for_ordered_teardown() -> None:
    """Raw disposer participates in nested composite effect LIFO teardown."""
    ctx = Context()
    order: list[str] = []
    disposers: list[Any] = []

    def _setup(inner: Context) -> None:
        def _composite() -> Any:
            order_push_outer = lambda: order.append("outer")
            scope = create_scope(inner, {"name": "nested"})
            scope.ctx.effect(lambda: lambda: order.append("scope"))
            yield order_push_outer
            yield scope.raw_dispose
            yield lambda: order.append("inner")

        disposers.append(inner.fiber.effect(_composite))

    ctx.use(_setup)
    await ctx.flush()
    await disposers[0]()
    assert order == ["inner", "scope", "outer"]


@pytest.mark.asyncio
async def test_scoped_context_supports_generator_effects() -> None:
    """Generator functions passed to scope.ctx.effect register yielded cleanups."""
    ctx = Context()
    scope = await _mint_scope(ctx, {"name": "gen"})
    events: list[str] = []

    def _gen_effect() -> Any:
        yield lambda: events.append("first")
        yield lambda: events.append("second")

    scope.ctx.effect(_gen_effect, label="gen-test")
    await scope.dispose()
    assert events == ["second", "first"]


@pytest.mark.asyncio
async def test_scope_target_routes_scoped_listeners_and_global_untagged() -> None:
    """Scoped listeners hear matching keys; untagged listeners hear all."""
    ctx = Context()
    key_a = {"name": "A"}
    key_b = {"name": "B"}
    scope_a = await _mint_scope(ctx, key_a)
    scope_b = await _mint_scope(ctx, key_b)
    heard: list[str] = []

    ctx.on("scope-test/ping", lambda value: heard.append(f"global:{value}"))
    scope_a.ctx.on("scope-test/ping", lambda value: heard.append(f"A:{value}"))
    scope_b.ctx.on("scope-test/ping", lambda value: heard.append(f"B:{value}"))

    emit_scoped(ctx, scope_target(ctx, key_a), "scope-test/ping", "a")
    emit_scoped(ctx, scope_target(ctx, key_b), "scope-test/ping", "b")
    emit_scoped(ctx, scope_target(ctx, None), "scope-test/ping", "none")

    assert heard == ["global:a", "A:a", "global:b", "B:b", "global:none"]
    await asyncio.gather(scope_a.dispose(), scope_b.dispose())


@pytest.mark.asyncio
async def test_scope_target_preserves_base_filter_and_receiver() -> None:
    """Base Context.symbols['filter'], Service._filter, and Context.match are kept."""
    ctx = Context()
    key = {"name": "A"}
    scope = await _mint_scope(ctx, key)
    heard: list[str] = []
    ctx.on("scope-test/ping", lambda value: heard.append(f"global:{value}"))
    scope.ctx.on("scope-test/ping", lambda value: heard.append(f"A:{value}"))
    receiver_matches = False

    class _BaseWithSymbolFilter:
        def check_filter(self, _ctx: Context) -> bool:
            nonlocal receiver_matches
            receiver_matches = self is base_sym
            return False

    base_sym = _BaseWithSymbolFilter()
    setattr(base_sym, Context.SYMBOL_FILTER, base_sym.check_filter)

    emit_scoped(ctx, scope_target(base_sym, key), "scope-test/ping", "vetoed")
    assert heard == []
    assert receiver_matches is True

    # Context.filter predicate is also preserved on base context carriers.
    filtered_base = ctx.filter(lambda c: c is scope.ctx)
    emit_scoped(ctx, scope_target(filtered_base, key), "scope-test/ping", "allowed")
    assert heard == ["A:allowed"]

    await scope.dispose()


@pytest.mark.asyncio
async def test_scope_target_retains_global_listener_semantics() -> None:
    """Listeners registered with global_=True receive foreign and unkeyed events."""
    ctx = Context()
    scope = await _mint_scope(ctx, {"name": "A"})
    heard: list[str] = []
    scope.ctx.on("scope-test/ping", lambda value: heard.append(value), global_=True)
    emit_scoped(ctx, scope_target(ctx, {"name": "other"}), "scope-test/ping", "foreign")
    emit_scoped(ctx, scope_target(ctx, None), "scope-test/ping", "none")
    assert heard == ["foreign", "none"]
    await scope.dispose()


def test_scope_target_opaque_carrier_and_metadata() -> None:
    """Carrier is opaque with separately tracked identity key."""
    key = {"name": "key"}
    subject = {"value": 1}
    carrier = scope_target(subject, key)
    assert is_scope_carrier(carrier) is True
    assert carrier_key_of(carrier) is key
    assert is_scope_carrier(subject) is False
    assert carrier_key_of(subject) is None
    assert carrier_key_of(None) is None
    assert not hasattr(carrier, "value")


def test_scope_parent_links_chains_and_rejects_cycles() -> None:
    """Parent chain links at mint, walks rootward, and rejects cycles."""
    ctx = Context()
    preset = {"kind": "preset"}
    agent = {"kind": "agent"}
    create_scope(ctx, preset)
    create_scope(ctx, agent, CreateScopeOptions(parent=preset))

    assert scope_parent_of(agent) is preset
    assert scope_parent_of(preset) is None
    assert scope_chain_of(agent) == [agent, preset]
    assert scope_chain_of(None) == []
    with pytest.raises(ValueError, match="cycle"):
        bind_scope_parent(preset, agent)
    with pytest.raises(ValueError, match="cycle"):
        bind_scope_parent(preset, preset)


def test_scope_parent_relinks_only_through_binding() -> None:
    """Re-linking requires the returned binding and preserves cycle checks."""
    ctx = Context()
    preset_a = {"id": "a"}
    preset_b = {"id": "b"}
    agent = {"id": "agent"}
    create_scope(ctx, preset_a)
    create_scope(ctx, preset_b)
    binding = bind_scope_parent(agent, preset_a)
    create_scope(ctx, agent, {"parent": None})

    with pytest.raises(ValueError, match="already bound"):
        bind_scope_parent(agent, preset_b)

    binding.rebind(preset_b)
    assert scope_chain_of(agent) == [agent, preset_b]

    child = {"id": "child"}
    bind_scope_parent(child, agent)
    with pytest.raises(ValueError, match="cycle"):
        binding.rebind(child)


@pytest.mark.asyncio
async def test_scope_admits_ancestor_listeners_never_descendants() -> None:
    """Events flow up the scope parent chain, never down."""
    ctx = Context()
    preset = {"kind": "preset"}
    agent = {"kind": "agent"}
    other = {"kind": "other-preset"}
    preset_scope = await _mint_scope(ctx, preset)
    agent_scope = await _mint_scope(ctx, agent, {"parent": preset})
    other_scope = await _mint_scope(ctx, other)

    seen: list[str] = []
    ctx.on("probe/event", lambda: seen.append("untagged"))
    preset_scope.ctx.on("probe/event", lambda: seen.append("preset"))
    agent_scope.ctx.on("probe/event", lambda: seen.append("agent"))
    other_scope.ctx.on("probe/event", lambda: seen.append("other"))

    emit_scoped(ctx, scope_target(ctx, agent), "probe/event")
    assert sorted(seen) == ["agent", "preset", "untagged"]

    seen.clear()
    emit_scoped(ctx, scope_target(ctx, preset), "probe/event")
    assert sorted(seen) == ["preset", "untagged"]


@pytest.mark.asyncio
async def test_waterfall_scoped_filters_listeners() -> None:
    """waterfall_scoped dispatches only to matching-scope and global listeners."""
    ctx = Context()
    key_a = {"id": "a"}
    key_b = {"id": "b"}
    scope_a = await _mint_scope(ctx, key_a)
    scope_b = await _mint_scope(ctx, key_b)

    async def _add_a(payload: list[str], next_fn: Any) -> list[str]:
        payload.append("A")
        return await next_fn(payload)

    async def _add_b(payload: list[str], next_fn: Any) -> list[str]:
        payload.append("B")
        return await next_fn(payload)

    scope_a.ctx.on("probe/waterfall", _add_a, mode=EventMode.WATERFALL)
    scope_b.ctx.on("probe/waterfall", _add_b, mode=EventMode.WATERFALL)

    res = await waterfall_scoped(
        ctx,
        scope_target(ctx, key_a),
        "probe/waterfall",
        [],
        default_handler=lambda items: items,
    )
    assert res == ["A"]


@pytest.mark.asyncio
async def test_scoped_context_extend_and_base_filter_variants() -> None:
    """_ScopedContext.extend sets non-fiber attrs and scope_target checks base filters."""
    ctx = Context()
    key = {"id": "k"}
    scope = await _mint_scope(ctx, key)

    extended = scope.ctx.extend({"tag": "custom"})
    assert scope_of(extended) is key
    assert extended.tag == "custom"

    class _MatchBase:
        def __init__(self, expected: Context) -> None:
            self.expected = expected

        def match(self, listener_ctx: Context) -> bool:
            return listener_ctx is self.expected

    class _FilterBase:
        def __init__(self, allow: bool) -> None:
            self._allow = allow

        def _filter(self, _listener_ctx: Context) -> bool:
            return self._allow

    seen: list[str] = []
    scope.ctx.on("probe/base", lambda: seen.append("scoped"))

    emit_scoped(ctx, scope_target({}, key), "probe/base")
    assert seen == ["scoped"]

    seen.clear()
    emit_scoped(ctx, scope_target(_MatchBase(scope.ctx), key), "probe/base")
    assert seen == ["scoped"]

    seen.clear()
    emit_scoped(ctx, scope_target(_MatchBase(ctx), key), "probe/base")
    assert seen == []

    seen.clear()
    emit_scoped(ctx, scope_target(_FilterBase(True), key), "probe/base")
    assert seen == ["scoped"]

    seen.clear()
    emit_scoped(ctx, scope_target(_FilterBase(False), key), "probe/base")
    assert seen == []


@pytest.mark.asyncio
async def test_nested_derived_scoped_context_service_resolution_and_isolation() -> None:
    """Nested and derived _ScopedContexts resolve ancestor services and honor isolation."""

    class _BoxService(Service):
        def __init__(self, target_ctx: Context, tag: str) -> None:
            super().__init__(target_ctx, "box")
            target_ctx.reflect.provide("boxCamel", self)
            self.tag = tag

        def ping(self) -> str:
            return self.tag

    ctx = Context()
    _BoxService(ctx, "root-box")
    ctx.provide("plainCfg", {"mode": "strict"})
    ctx.customAttr = "root-attr"

    outer = await _mint_scope(ctx, {"id": "outer"})
    inner = create_scope(outer.ctx, {"id": "inner"})
    derived = create_scope(ctx.extend(), {"id": "derived"})

    # Missing private attributes must not walk ancestors
    with pytest.raises(AttributeError):
        _ = inner.ctx._nonexistent_private
    assert "_nonexistent_private" not in inner.ctx
    assert "_fiber" in inner.ctx

    # Nested and extend()-derived scopes resolve parent services & plain provides
    assert inner.ctx.box.ping() == "root-box"
    assert inner.ctx.boxCamel.ping() == "root-box"
    assert inner.ctx.plainCfg == {"mode": "strict"}
    assert "boxCamel" in inner.ctx
    assert "plainCfg" in inner.ctx
    assert "customAttr" in inner.ctx
    assert "reflect" in inner.ctx
    assert "missingSvc" not in inner.ctx
    assert inner.ctx.has("boxCamel") is True
    assert inner.ctx.has("missingSvc") is False
    assert inner.ctx.get("boxCamel").ping() == "root-box"
    assert inner.ctx.get("missingSvc", "fallback") == "fallback"
    assert (
        inner.ctx.get("reflect") is ctx.reflect or inner.ctx.get("reflect") is not None
    )
    assert "boxCamel" in inner.ctx.services
    assert "plainCfg" in inner.ctx.services
    assert derived.ctx.boxCamel.ping() == "root-box"

    # Non-existent attribute still fails loud
    with pytest.raises(AttributeError):
        _ = inner.ctx.missingSvc

    # Isolation boundary blocks direct access on iso as well as iso_child
    iso = outer.ctx.isolate("boxCamel")
    iso_child = create_scope(iso, {"id": "iso-child"})
    assert iso.has("boxCamel") is False
    assert "boxCamel" not in iso
    assert "boxCamel" not in iso.services
    assert iso.get("boxCamel", "blocked") == "blocked"
    with pytest.raises(AttributeError):
        _ = iso.boxCamel
    assert iso_child.ctx.has("boxCamel") is False
    assert "boxCamel" not in iso_child.ctx
    assert "boxCamel" not in iso_child.ctx.services
    assert iso_child.ctx.get("boxCamel", "blocked") == "blocked"
    with pytest.raises(AttributeError):
        _ = iso_child.ctx.boxCamel

    # Re-providing on isolated parent exposes the isolated service directly and to children
    iso.provide("boxCamel", {"local": True})
    assert iso.boxCamel == {"local": True}
    assert iso_child.ctx.has("boxCamel") is True
    assert iso_child.ctx.boxCamel == {"local": True}
    assert iso_child.ctx.get("boxCamel") == {"local": True}
    assert iso_child.ctx.services["boxCamel"] == {"local": True}

    # Check predicate gating (`impl.check`) is enforced across nested scopes,
    # while unbound method `check(self)` falls back to True per reflect contract.
    gate_open = False
    ctx.reflect.provide("gatedSvc", {"secret": 1}, check=lambda: gate_open)
    with pytest.raises(AttributeError):
        _ = outer.ctx.gatedSvc
    with pytest.raises(AttributeError):
        _ = inner.ctx.gatedSvc
    gate_open = True
    assert outer.ctx.gatedSvc == {"secret": 1}
    assert inner.ctx.gatedSvc == {"secret": 1}

    class _MethodCheckService(Service):
        def __init__(self, target_ctx: Context) -> None:
            super().__init__(target_ctx, "methodCheckSvc")

        def check(
            self,
        ) -> bool:  # pragma: no cover - signature triggers TypeError in 0-arg check()
            return False

    _MethodCheckService(ctx)
    assert inner.ctx.methodCheckSvc is not None

    # Check non-existent handler_has walk across isolation
    iso_plain = outer.ctx.isolate("plainCfg")
    assert "plainCfg" not in iso_plain
    await iso_child.dispose()
    await inner.dispose()
    await derived.dispose()
    await outer.dispose()


@pytest.mark.asyncio
async def test_child_and_grandchild_plugin_fibers_in_scope() -> None:
    """scope.ctx.use(Plugin) keeps child effects/injections scoped to the child fiber."""
    cleaned: list[str] = []
    observed: list[str] = []

    class _DemoService(Service):
        def __init__(self, target_ctx: Context) -> None:
            super().__init__(target_ctx, "demoSvc")

        def who(self) -> Any:
            return scope_of(_access_context.get() or self.ctx)

    class _GrandChildPlugin:
        inject: ClassVar[list[str]] = ["demoSvc"]

        def __init__(self, child_ctx: Context) -> None:
            observed.append(str(child_ctx.demoSvc.who()["id"]))

            def _gc_effect() -> Any:
                return lambda: cleaned.append("grandchild")

            child_ctx.effect(_gc_effect, label="gc-effect")

    class _ChildPlugin:
        inject: ClassVar[list[str]] = ["demoSvc"]

        def __init__(self, child_ctx: Context) -> None:
            ext = child_ctx.extend()
            observed.append(f"child:{child_ctx.demoSvc.who()['id']}")
            observed.append(f"ext:{ext.demoSvc.who()['id']}")
            child_ctx.effect(
                lambda: lambda: cleaned.append("child-effect"),
                label="child-effect",
            )
            child_ctx.use(_GrandChildPlugin)

    ctx = Context()
    _DemoService(ctx)
    scope = await _mint_scope(ctx, {"id": "scope-a"})
    await scope._fiber

    class _LocalService(Service):
        def __init__(self, target_ctx: Context) -> None:
            super().__init__(target_ctx, "localSvc")

    local_svc = _LocalService(scope.ctx)
    assert scope.ctx.get("localSvc") is local_svc

    def _scope_gen_effect() -> Any:
        yield lambda: cleaned.append("scope-gen")

    scope.ctx.effect(_scope_gen_effect, label="scope-gen")
    assert callable(scope.rawDispose)
    assert scope.rawDispose == scope.raw_dispose

    child_fiber = scope.ctx.use(_ChildPlugin)
    await scope.ctx.flush()
    assert observed == ["child:scope-a", "ext:scope-a", "scope-a"]
    assert cleaned == []

    # Disposing child_fiber removes child & grandchild without disposing scope itself
    await child_fiber.dispose()
    assert sorted(cleaned) == ["child-effect", "grandchild"]

    # Disposing the scope later cleans up scope's own generator effect
    await scope.rawDispose()
    assert cleaned[-1] == "scope-gen"


@pytest.mark.asyncio
async def test_scope_weakref_release_and_non_weakrefable_keys() -> None:
    """Weakrefable scope keys and carriers are released; dict keys also remain clean."""

    @dataclass
    class _UnhashableKey:
        name: str

    class _WeakKey:
        pass

    ctx = Context()
    wk_parent = _WeakKey()
    wk_child: _WeakKey | None = _WeakKey()
    other_parent = _WeakKey()

    # Duplicate weak-key check, rebind + GC cleanup
    binding = bind_scope_parent(wk_child, wk_parent)
    assert scope_parent_of(wk_child) is wk_parent
    with pytest.raises(ValueError, match="already bound"):
        bind_scope_parent(wk_child, _WeakKey())
    binding.rebind(other_parent)
    assert scope_parent_of(wk_child) is other_parent

    # Unhashable weakrefable keys use object identity (`is`), not value equality (`==`)
    k_eq1 = _UnhashableKey("same")
    k_eq2 = _UnhashableKey("same")
    assert k_eq1 == k_eq2
    bind_scope_parent(k_eq1, wk_parent)
    assert scope_parent_of(k_eq1) is wk_parent
    assert scope_parent_of(k_eq2) is None
    with pytest.raises(ValueError, match="already bound"):
        bind_scope_parent(k_eq1, other_parent)

    key_ref = weakref.ref(wk_child)
    carrier = scope_target(ctx, wk_parent)
    carrier_ref = weakref.ref(carrier)
    assert is_scope_carrier(carrier) is True
    assert is_scope_carrier("not-a-carrier") is False
    assert carrier_key_of(None) is None
    assert carrier_key_of(scope_target(ctx, None)) is None

    # Dict (non-weakrefable) scope keys work for bind/lookup/duplicate too
    dict_parent = {"id": "dp"}
    dict_child = {"id": "dc"}
    bind_scope_parent(dict_child, dict_parent)
    assert scope_parent_of(dict_child) is dict_parent
    assert scope_parent_of({"id": "unbound"}) is None
    with pytest.raises(ValueError, match="already bound"):
        bind_scope_parent(dict_child, {"id": "other"})

    del binding
    wk_child = None
    del carrier
    gc.collect()
    assert key_ref() is None
    assert carrier_ref() is None
