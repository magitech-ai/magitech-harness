"""Insertion-ordered storage and effect ownership for scope-aware registries."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

from animas import Context, Disposer
from animas.core import _current_fiber

from magitech_harness.scope._core import ScopeKey, scope_chain_of, scope_of


class ScopeLayer(Protocol):
    """One scope's aggregate contribution to a registry."""

    def is_empty(self) -> bool:
        """Whether every table in this layer is empty."""
        ...


class LayerEffectOptions(TypedDict, total=False):
    """Options controlling lifecycle effect labels and notifications."""

    label: str
    notify: bool


@dataclass
class _EntryNode[V]:
    key: Any
    value: V
    alive: bool = True


class _EntryGeneration[V]:
    def __init__(self) -> None:
        self.nodes: list[_EntryNode[V]] = []
        self.by_key: dict[Any, _EntryNode[V]] = {}


def _iter_generation[V](
    gen: _EntryGeneration[V], pick: Callable[[_EntryNode[V]], Any]
) -> Iterator[Any]:
    idx = 0
    while idx < len(gen.nodes):
        node = gen.nodes[idx]
        idx += 1
        if node.alive:
            yield pick(node)


class NamedEntries[V]:
    """Insertion-ordered named entries with caller-owned duplicate diagnostics."""

    def __init__(self, duplicate_error: Callable[[str], Exception]) -> None:
        self._duplicate_error = duplicate_error
        self._gen: _EntryGeneration[V] = _EntryGeneration()

    def insert(self, name: str, value: V) -> Callable[[], None]:
        """Insert one unique name and return an idempotent undo closure."""
        gen = self._gen
        if name in gen.by_key:
            raise self._duplicate_error(name)
        node: _EntryNode[V] = _EntryNode(key=name, value=value)
        gen.nodes.append(node)
        gen.by_key[name] = node
        active = True

        def _undo() -> None:
            nonlocal active
            if not active:
                return
            active = False
            node.alive = False
            gen.by_key.pop(name, None)
            if not gen.by_key and self._gen is gen:
                self._gen = _EntryGeneration()

        return _undo

    def get(self, name: str) -> V | None:
        """Read one named value, or None when absent."""
        node = self._gen.by_key.get(name)
        return node.value if node is not None else None

    def has(self, name: str) -> bool:
        """Test whether the table contains `name`."""
        return name in self._gen.by_key

    def keys(self) -> Iterator[str]:
        """Iterate live names in insertion order."""
        return cast(Iterator[str], _iter_generation(self._gen, lambda n: n.key))

    def entries(self) -> Iterator[tuple[str, V]]:
        """Iterate live (name, value) pairs in insertion order."""
        return cast(
            Iterator[tuple[str, V]],
            _iter_generation(self._gen, lambda n: (n.key, n.value)),
        )

    def values(self) -> Iterator[V]:
        """Iterate live values in insertion order."""
        return cast(Iterator[V], _iter_generation(self._gen, lambda n: n.value))

    def is_empty(self) -> bool:
        """Test whether this table has no entries."""
        return not self._gen.by_key


class AnonymousEntries[V]:
    """Insertion-ordered anonymous entries with independent identity."""

    def __init__(self) -> None:
        self._gen: _EntryGeneration[V] = _EntryGeneration()

    def append(self, value: V) -> Callable[[], None]:
        """Append one independently owned value and return its undo closure."""
        gen = self._gen
        token = object()
        node: _EntryNode[V] = _EntryNode(key=token, value=value)
        gen.nodes.append(node)
        gen.by_key[token] = node
        active = True

        def _undo() -> None:
            nonlocal active
            if not active:
                return
            active = False
            node.alive = False
            gen.by_key.pop(token, None)
            if not gen.by_key and self._gen is gen:
                self._gen = _EntryGeneration()

        return _undo

    def values(self) -> Iterator[V]:
        """Iterate live values in insertion order."""
        return cast(Iterator[V], _iter_generation(self._gen, lambda n: n.value))

    def is_empty(self) -> bool:
        """Test whether this table has no entries."""
        return not self._gen.by_key


class ScopedLayers[L: ScopeLayer]:
    """Own global and exact-scope overlay layers for one registry."""

    def __init__(
        self,
        create_layer: Callable[[ScopeKey | None], L],
        on_change: Callable[[], None],
    ) -> None:
        self._create_layer = create_layer
        self._on_change = on_change
        self.global_layer: L = create_layer(None)
        self._scoped: dict[int, tuple[ScopeKey, L]] = {}

    def peek(self, scope: ScopeKey | None) -> L | None:
        """Read an existing exact-scope overlay without creating one."""
        if scope is None:
            return None
        entry = self._scoped.get(id(scope))
        return entry[1] if entry is not None else None

    def chain_layers(self, scope: ScopeKey | None) -> list[L]:
        """Existing overlays along the parent chain, farthest ancestor first."""
        layers: list[L] = []
        for key in reversed(scope_chain_of(scope)):
            entry = self._scoped.get(id(key))
            if entry is not None:
                layers.append(entry[1])
        return layers

    def merge[V](
        self,
        scope: ScopeKey | None,
        pick: Callable[[L], NamedEntries[V]],
    ) -> dict[str, V]:
        """Materialize global named entries shadowed farthest-ancestor-first."""
        merged = dict(pick(self.global_layer).entries())
        for layer in self.chain_layers(scope):
            for name, value in pick(layer).entries():
                merged[name] = value
        return merged

    def effect(
        self,
        ctx: Context,
        action: Callable[[L], Callable[[], None]],
        options: LayerEffectOptions,
    ) -> Disposer:
        """Attach one synchronous layer mutation to its registration context."""
        scope = scope_of(ctx)
        label = options.get("label", "anonymous")
        notify = options.get("notify", True)

        def _setup() -> Disposer:
            created = False
            if scope is None:
                layer = self.global_layer
            else:
                existing = self._scoped.get(id(scope))
                if existing is None:
                    layer = self._create_layer(scope)
                    self._scoped[id(scope)] = (scope, layer)
                    created = True
                else:
                    layer = existing[1]

            try:
                undo = action(layer)
            except Exception:
                if scope is not None and created and layer.is_empty():
                    self._scoped.pop(id(scope), None)
                raise

            def _teardown() -> None:
                undo()
                if scope is not None and layer.is_empty():
                    self._scoped.pop(id(scope), None)
                if notify:
                    self._on_change()

            if notify:
                try:
                    self._on_change()
                except Exception:
                    _teardown()
                    raise

            return _teardown

        # Attach to the context's fiber even outside component init, while
        # still letting mock contexts with only `.effect()` intercept verbatim.
        if isinstance(ctx, Context) and _current_fiber.get() is None:
            token = _current_fiber.set(ctx.fiber)
            try:
                return ctx.effect(_setup, label)
            finally:
                _current_fiber.reset(token)
        return ctx.effect(_setup, label)
