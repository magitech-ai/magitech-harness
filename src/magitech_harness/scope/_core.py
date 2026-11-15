"""Core scoped-context primitives and scope-filtered event routing."""

from __future__ import annotations

import asyncio
import inspect
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from anima import Context, Disposer, Fiber
from anima.core import Service, _BoundService, _current_fiber

ScopeKey = object

_carriers: weakref.WeakSet[Scoped[Any]] = weakref.WeakSet()
_scope_parents: dict[int, tuple[weakref.ref[Any] | ScopeKey, ScopeKey]] = {}


def _scope_noop(_ctx: Context) -> None:
    """Shared no-op component used as the backing scope fiber."""


class ScopeParentBinding:
    """Privileged handle to re-link one scope key to a new parent."""

    def __init__(self, key: ScopeKey) -> None:
        self._key = key

    def rebind(self, parent: ScopeKey) -> None:
        """Re-link the bound key to a different enclosing scope key."""
        _link_scope_parent(self._key, parent)


def _has_scope_parent(key: ScopeKey) -> bool:
    entry = _scope_parents.get(id(key))
    if entry is None:
        return False
    stored = entry[0]
    target = stored() if isinstance(stored, weakref.ref) else stored
    return target is key


def _link_scope_parent(key: ScopeKey, parent: ScopeKey) -> None:
    cursor: ScopeKey | None = parent
    while cursor is not None:
        if cursor is key:
            raise ValueError("scope: scope parent link would form a cycle")
        cursor = scope_parent_of(cursor)
    key_id = id(key)

    def _evict(_ref: weakref.ref[Any]) -> None:
        _scope_parents.pop(key_id, None)

    try:
        stored: weakref.ref[Any] | ScopeKey = weakref.ref(key, _evict)
    except TypeError:
        stored = key
    _scope_parents[key_id] = (stored, parent)


def bind_scope_parent(key: ScopeKey, parent: ScopeKey) -> ScopeParentBinding:
    """Bind `parent` as `key`'s enclosing scope once and return its rebind handle."""
    if _has_scope_parent(key):
        raise ValueError(
            "scope: scope key is already bound to a parent; "
            "re-linking requires the binding returned by the original bind"
        )
    _link_scope_parent(key, parent)
    return ScopeParentBinding(key)


def scope_parent_of(key: ScopeKey) -> ScopeKey | None:
    """Read one scope key's enclosing parent key, or None for a root scope."""
    entry = _scope_parents.get(id(key))
    if entry is None:
        return None
    stored, parent = entry
    target = stored() if isinstance(stored, weakref.ref) else stored
    return parent if target is key else None


def scope_chain_of(key: ScopeKey | None) -> list[ScopeKey]:
    """Return the chain from `key` to its root ancestor, nearest first."""
    chain: list[ScopeKey] = []
    cursor = key
    while cursor is not None:
        chain.append(cursor)
        cursor = scope_parent_of(cursor)
    return chain


@dataclass(frozen=True)
class CreateScopeOptions:
    """Options accepted by :func:`create_scope`."""

    parent: ScopeKey | None = None


@dataclass
class Scope:
    """A minted registration scope and its quiescent disposal boundaries."""

    ctx: Context
    raw_dispose: Callable[[], Awaitable[None]]
    _fiber: Fiber
    _disposing: asyncio.Task[None] | None = None

    @property
    def rawDispose(self) -> Callable[[], Awaitable[None]]:
        """Alias for :attr:`raw_dispose`."""
        return self.raw_dispose

    async def dispose(self) -> None:
        """Dispose every scope-owned registration, sharing quiescence across calls."""
        if self._disposing is None:
            self._disposing = asyncio.ensure_future(_quiesce_fiber(self._fiber))
        await self._disposing


async def _quiesce_fiber(fiber: Fiber) -> None:
    await fiber.dispose()
    while fiber.inertia is not None:
        await fiber.inertia


def _impl_passes_check(impl: Any) -> bool:
    check_fn = impl.check
    if check_fn is None:
        return True
    try:
        return bool(check_fn())
    except TypeError:
        return True


def _is_descendant_fiber(candidate: Fiber | None, ancestor: Fiber) -> bool:
    cur = candidate.parent_fiber if candidate is not None else None
    while cur is not None:
        if cur is ancestor:
            return True
        cur = cur.parent_fiber
    return False


class _ScopedContext(Context):
    """Child Context carrying its own backing fiber across derived contexts."""

    def __init__(
        self,
        parent: Context,
        *,
        fiber: Fiber,
        scope_key: ScopeKey | None = None,
        isolated_services: dict[str, Any] | None = None,
        intercept_configs: dict[str, dict[str, Any]] | None = None,
        filter_predicate: Callable[[Context], bool] | None = None,
    ) -> None:
        super().__init__(
            parent=parent,
            isolated_services=isolated_services,
            intercept_configs=intercept_configs,
            fiber=fiber,
            filter_predicate=filter_predicate,
        )
        if scope_key is not None:
            object.__setattr__(self, "_mth_scope", scope_key)

    @property
    def _backing_fiber(self) -> Fiber:
        return cast(Fiber, object.__getattribute__(self, "_fiber"))

    @property
    def fiber(self) -> Fiber:
        """Return active descendant child fiber when inside one, else the backing scope fiber."""
        backing = self._backing_fiber
        current = _current_fiber.get()
        if current is not None and _is_descendant_fiber(current, backing):
            return current
        return backing

    def effect(self, func: Callable[[], Any], label: str | None = None) -> Disposer:
        """Attach effects to the active descendant or scope backing fiber."""
        target_fiber = self.fiber
        if inspect.isgeneratorfunction(func):
            return cast(Disposer, target_fiber.effect(func, label=label or "anonymous"))
        token = _current_fiber.set(target_fiber)
        try:
            return super().effect(func, label=label)
        finally:
            _current_fiber.reset(token)

    def _bind_service(self, svc: Any) -> Any:
        if isinstance(svc, _BoundService):
            svc = object.__getattribute__(svc, "_svc")
        if isinstance(svc, Service) and getattr(svc, "ctx", None) is not self:
            return _BoundService(svc, self)
        return svc

    def has(self, name: str) -> bool:
        """Check whether `name` is registered on this scope or an un-isolated ancestor."""
        curr: Context = self
        while True:
            if name in curr._services:
                return True
            if curr.reflect._get_impl(name, strict=False) is not None:
                return True
            parent = curr._parent
            if parent is None or (
                curr._isolated_services.get(name) != parent._isolated_services.get(name)
            ):
                return False
            curr = parent

    def get(self, name: str, default: Any = None, strict: bool = True) -> Any:
        """Resolve a service up the un-isolated ancestor chain and bind to this context."""
        curr: Context = self
        while True:
            val = curr.reflect.get(name, strict=strict)
            if val is not None:
                return self._bind_service(val)
            parent = curr._parent
            if parent is None or (
                curr._isolated_services.get(name) != parent._isolated_services.get(name)
            ):
                return default
            curr = parent

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattr__(name)
        if self._isolated_services.get(name) == self.fiber.ctx._isolated_services.get(
            name
        ):
            try:
                return self._bind_service(super().__getattr__(name))
            except AttributeError:
                pass
        curr: Context = self
        while True:
            impl = curr.reflect._get_impl(name, strict=False)
            if impl is not None:
                if not _impl_passes_check(impl):
                    raise AttributeError(
                        f"Service '{name}' is not registered on Context. "
                        f"Available services: {list(self.services.keys())}"
                    )
                return self._bind_service(impl.value)
            if name in curr._services:
                return self._bind_service(curr._services[name])
            parent = curr._parent
            if parent is None or (
                curr._isolated_services.get(name) != parent._isolated_services.get(name)
            ):
                raise AttributeError(
                    f"Service '{name}' is not registered on Context. "
                    f"Available services: {list(self.services.keys())}"
                )
            curr = parent

    def __contains__(self, name: str) -> bool:
        if name.startswith("_"):
            return hasattr(self, name)
        if self.has(name):
            return True
        curr: Context = self
        while True:
            if curr.reflect.handler_has(curr, name):
                return True
            parent = curr._parent
            if parent is None or (
                curr._isolated_services.get(name) != parent._isolated_services.get(name)
            ):
                return False
            curr = parent

    @property
    def services(self) -> dict[str, Any]:
        """Dictionary of all services accessible across un-isolated ancestors."""
        chain: list[Context] = []
        curr: Context | None = self
        while curr is not None:
            chain.append(curr)
            curr = curr._parent
        chain.reverse()
        all_svcs: dict[str, Any] = {}
        prev: Context | None = None
        for node in chain:
            if prev is not None:
                for key in list(all_svcs):
                    if node._isolated_services.get(key) != prev._isolated_services.get(
                        key
                    ):
                        del all_svcs[key]
            all_svcs.update(node._services)
            for impl in node.reflect.store.values():
                all_svcs[impl.name] = impl.value
            prev = node
        return all_svcs

    def extend(self, meta: dict[str, Any] | None = None) -> Context:
        """Create a child context preserving scope identity and backing fiber."""
        child = _ScopedContext(
            self,
            fiber=self._backing_fiber,
            isolated_services=dict(self._isolated_services),
            intercept_configs=None,
            filter_predicate=self._filter_predicate,
        )
        if meta:
            for attr, value in meta.items():
                if attr == "fiber":
                    object.__setattr__(child, "_fiber", value)
                else:
                    setattr(child, attr, value)
        return child

    def isolate(self, service_name: str, label: Any | None = None) -> Context:
        """Create an isolated child context preserving scope backing fiber."""
        isolated = dict(self._isolated_services)
        isolated[service_name] = (
            label if label is not None else f"__symbol__{service_name}"
        )
        return _ScopedContext(
            self,
            fiber=self._backing_fiber,
            isolated_services=isolated,
            intercept_configs=None,
            filter_predicate=self._filter_predicate,
        )

    def intercept(self, service_name: str, config: dict[str, Any]) -> Context:
        """Create an intercepted child context preserving scope backing fiber."""
        return _ScopedContext(
            self,
            fiber=self._backing_fiber,
            isolated_services=dict(self._isolated_services),
            intercept_configs={service_name: dict(config)},
            filter_predicate=self._filter_predicate,
        )


def _extract_parent(
    options: CreateScopeOptions | Mapping[str, Any] | None,
) -> ScopeKey | None:
    if options is None:
        return None
    if isinstance(options, CreateScopeOptions):
        return options.parent
    return options.get("parent")


def create_scope(
    ctx: Context,
    key: ScopeKey,
    options: CreateScopeOptions | Mapping[str, Any] | None = None,
) -> Scope:
    """Mint a registration scope under `ctx` tagged with `key`."""
    parent = _extract_parent(options)
    if parent is not None:
        bind_scope_parent(key, parent)
    fiber = ctx.use(_scope_noop)
    scoped_ctx = _ScopedContext(
        fiber.ctx,
        fiber=fiber,
        scope_key=key,
        isolated_services=dict(fiber.ctx._isolated_services),
        filter_predicate=fiber.ctx._filter_predicate,
    )
    raw_dispose = fiber.dispose
    return Scope(ctx=scoped_ctx, raw_dispose=raw_dispose, _fiber=fiber)


def scope_of(ctx: Context) -> ScopeKey | None:
    """Read the nearest scope key inherited by `ctx`, or None when unscoped."""
    curr: Any = ctx
    while curr is not None:
        if "_mth_scope" in curr.__dict__:
            return cast(ScopeKey | None, curr.__dict__["_mth_scope"])
        curr = getattr(curr, "_parent", None)
    return None


def _base_allows(base: object, listener_ctx: Context) -> bool:
    symbol_filter = getattr(base, Context.SYMBOL_FILTER, None)
    if callable(symbol_filter):
        return bool(symbol_filter(listener_ctx))
    svc_filter = getattr(base, "_filter", None)
    if callable(svc_filter):
        return bool(svc_filter(listener_ctx))
    match_fn = getattr(base, "match", None)
    if callable(match_fn):
        return bool(match_fn(listener_ctx))
    return True


class Scoped[T: object]:
    """Opaque routing-only event carrier preserving a base filter and scope key."""

    def __init__(self, base: T, key: ScopeKey | None) -> None:
        self._base = base
        self._key = key
        setattr(self, Context.SYMBOL_FILTER, self.match)

    def match(self, listener_ctx: Context) -> bool:
        """Check base filter admission and upward scope-chain membership."""
        if not _base_allows(self._base, listener_ctx):
            return False
        tag = scope_of(listener_ctx)
        if tag is None:
            return True
        cursor = self._key
        while cursor is not None:
            if cursor is tag:
                return True
            cursor = scope_parent_of(cursor)
        return False


def scope_target[T: object](base: T, key: ScopeKey | None) -> Scoped[T]:
    """Build an opaque scope-routed event carrier."""
    carrier = Scoped(base, key)
    _carriers.add(carrier)
    return carrier


def is_scope_carrier(value: object) -> bool:
    """Test whether `value` was created by :func:`scope_target`."""
    return isinstance(value, Scoped) and value in _carriers


def carrier_key_of(value: object) -> ScopeKey | None:
    """Read a carrier's scope key, or None for non-carriers or unkeyed carriers."""
    if not isinstance(value, Scoped) or value not in _carriers:
        return None
    return value._key


def emit_scoped(
    ctx: Context, target: Scoped[Any], event: str, *args: Any, **kwargs: Any
) -> None:
    """Synchronously broadcast `event` filtered by `target`."""
    ctx.events.emit(event, *args, this_arg=target, **kwargs)


async def waterfall_scoped(
    ctx: Context,
    target: Scoped[Any],
    event: str,
    *args: Any,
    default_handler: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Run an async waterfall over `event` filtered by `target`."""
    return await ctx.events.waterfall(
        event, *args, this_arg=target, default_handler=default_handler, **kwargs
    )
