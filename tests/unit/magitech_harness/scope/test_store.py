"""Unit tests for insertion-ordered live tables and ScopedLayers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from animas import Context

from magitech_harness.scope import (
    AnonymousEntries,
    NamedEntries,
    Scope,
    ScopedLayers,
    ScopeKey,
    ScopeLayer,
    create_scope,
)
from tests._mount import mount


class _TestLayer(ScopeLayer):
    def __init__(self, scope: ScopeKey | None) -> None:
        self.named: NamedEntries[int] = NamedEntries(
            lambda name: ValueError(
                f"{'global' if scope is None else 'scoped'} duplicate: {name}"
            )
        )
        self.anonymous: AnonymousEntries[str] = AnonymousEntries()

    def is_empty(self) -> bool:
        return self.named.is_empty() and self.anonymous.is_empty()


async def _mint_scope(ctx: Context, key: ScopeKey) -> Scope:
    scope: Scope | None = None

    def _setup(inner: Context) -> None:
        nonlocal scope
        scope = create_scope(inner, key)

    await mount(ctx, _setup)
    assert scope is not None
    return scope


def test_named_entries_lifecycle_live_iteration_and_idempotent_undo() -> None:
    """NamedEntries owns duplicate diagnostics, live iterators, and exact undo."""
    duplicate = ValueError("caller duplicate")
    duplicate_error = MagicMock(return_value=duplicate)
    entries: NamedEntries[int] = NamedEntries(duplicate_error)

    undo_a = entries.insert("a", 1)
    values = entries.values()
    assert next(values) == 1
    undo_b = entries.insert("b", 2)

    assert list(values) == [2]
    assert list(entries.keys()) == ["a", "b"]
    assert list(entries.entries()) == [("a", 1), ("b", 2)]
    assert entries.get("a") == 1
    assert entries.get("missing") is None
    assert entries.has("b") is True
    assert entries.has("missing") is False
    assert entries.is_empty() is False

    with pytest.raises(ValueError, match="caller duplicate"):
        entries.insert("a", 3)
    duplicate_error.assert_called_once_with("a")

    undo_a()
    entries.insert("a", 3)
    undo_a()
    assert entries.get("a") == 3
    undo_b()
    assert list(entries.entries()) == [("a", 3)]


def test_named_entries_starts_fresh_generation_after_drain() -> None:
    """Draining NamedEntries detaches live iterators from subsequent insertions."""
    entries: NamedEntries[int] = NamedEntries(lambda name: ValueError(name))
    undo = entries.insert("first", 1)
    values = entries.values()

    assert next(values) == 1
    undo()
    entries.insert("replacement", 2)

    assert list(values) == []
    assert list(entries.values()) == [2]


def test_anonymous_entries_independent_identity_and_drain_generation() -> None:
    """AnonymousEntries retains equal values separately with live iterators."""
    entries: AnonymousEntries[object] = AnonymousEntries()
    value = object()
    undo_first = entries.append(value)
    values = entries.values()
    assert next(values) is value
    undo_second = entries.append(value)

    assert list(values) == [value]
    assert list(entries.values()) == [value, value]
    undo_first()
    undo_first()
    assert list(entries.values()) == [value]
    undo_second()
    assert entries.is_empty() is True

    # Drain generation detaches active iterators.
    entries_num: AnonymousEntries[int] = AnonymousEntries()
    undo_one = entries_num.append(1)
    it = entries_num.values()
    assert next(it) == 1
    undo_one()
    entries_num.append(2)
    assert list(it) == []
    assert list(entries_num.values()) == [2]


def test_scoped_layers_non_creating_reads_and_merge_order() -> None:
    """Global layer constructs eagerly while reads stay non-creating."""
    created: list[ScopeKey | None] = []

    def _create(scope: ScopeKey | None) -> _TestLayer:
        created.append(scope)
        return _TestLayer(scope)

    layers: ScopedLayers[_TestLayer] = ScopedLayers(_create, MagicMock())
    key = object()
    layers.global_layer.named.insert("a", 1)
    layers.global_layer.named.insert("shared", 2)

    assert created == [None]
    assert layers.peek(None) is None
    assert layers.peek(key) is None
    assert list(layers.merge(key, lambda layer: layer.named).items()) == [
        ("a", 1),
        ("shared", 2),
    ]
    assert created == [None]


@pytest.mark.asyncio
async def test_scoped_layers_visibility_ownership_and_reclaim() -> None:
    """ScopedLayers creates lazily on write and reclaims only empty layers."""
    ctx = Context()
    key = object()
    scope = await _mint_scope(ctx, key)
    changed = MagicMock()
    created: list[ScopeKey | None] = []

    def _create(selected: ScopeKey | None) -> _TestLayer:
        created.append(selected)
        return _TestLayer(selected)

    layers: ScopedLayers[_TestLayer] = ScopedLayers(_create, changed)
    layers.global_layer.named.insert("a", 1)
    layers.global_layer.named.insert("shared", 1)

    remove_named = layers.effect(
        scope.ctx,
        lambda layer: layer.named.insert("shared", 2),
        {"label": "test.named", "notify": False},
    )
    remove_tail = layers.effect(
        scope.ctx,
        lambda layer: layer.named.insert("c", 3),
        {"label": "test.tail", "notify": False},
    )
    remove_anon = layers.effect(
        scope.ctx,
        lambda layer: layer.anonymous.append("kept"),
        {"label": "test.anonymous", "notify": False},
    )

    assert created == [None, key]
    assert list(layers.merge(key, lambda layer: layer.named).items()) == [
        ("a", 1),
        ("shared", 2),
        ("c", 3),
    ]
    changed.assert_not_called()

    remove_named()
    assert layers.peek(key) is not None
    assert list(layers.merge(key, lambda layer: layer.named).items()) == [
        ("a", 1),
        ("shared", 1),
        ("c", 3),
    ]

    remove_tail()
    assert layers.peek(key) is not None
    remove_anon()
    assert layers.peek(key) is None
    await scope.dispose()


def test_scoped_layers_order_idempotence_and_effect_label() -> None:
    """Runs action, notify, undo, notify in order with Context effect labels."""
    ctx = Context()
    events: list[str] = []
    layers: ScopedLayers[_TestLayer] = ScopedLayers(
        lambda scope: _TestLayer(scope),
        lambda: events.append("notify"),
    )

    def _action(layer: _TestLayer) -> Any:
        events.append("action")
        undo = layer.named.insert("x", 1)

        def _teardown() -> None:
            events.append("undo")
            undo()

        return _teardown

    dispose = layers.effect(ctx, _action, {"label": "store.order"})
    assert events == ["action", "notify"]
    assert any(
        meta and meta["label"] == "store.order" for meta in ctx.fiber.getEffects()
    )

    dispose()
    dispose()
    assert events == ["action", "notify", "undo", "notify"]
    assert layers.global_layer.is_empty() is True


def test_scoped_layers_returns_exact_context_effect_disposer() -> None:
    """Returns exact disposer from mock ctx.effect without calling action."""
    raw_dispose = MagicMock()
    effect = MagicMock(return_value=raw_dispose)

    class _MockCtx:
        def __init__(self) -> None:
            self.effect = effect

    mock_ctx: Any = _MockCtx()
    action = MagicMock(return_value=MagicMock())
    layers: ScopedLayers[_TestLayer] = ScopedLayers(
        lambda scope: _TestLayer(scope), MagicMock()
    )

    returned = layers.effect(
        mock_ctx, action, {"label": "store.identity", "notify": False}
    )

    assert returned is raw_dispose
    effect.assert_called_once()
    assert effect.call_args[0][1] == "store.identity"
    action.assert_not_called()


@pytest.mark.asyncio
async def test_scoped_layers_cleans_failed_factory_and_failed_actions() -> None:
    """Failed factories or actions clean empty layers without losing prior entries."""
    ctx = Context()
    key = object()
    scope = await _mint_scope(ctx, key)
    fail_factory = True

    def _create(selected: ScopeKey | None) -> _TestLayer:
        if selected is not None and fail_factory:
            raise RuntimeError("factory failed")
        return _TestLayer(selected)

    layers: ScopedLayers[_TestLayer] = ScopedLayers(_create, MagicMock())

    with pytest.raises(RuntimeError, match="factory failed"):
        layers.effect(
            scope.ctx,
            lambda layer: layer.named.insert("never", 1),
            {"label": "store.factory", "notify": False},
        )
    assert layers.peek(key) is None

    fail_factory = False

    def _boom(_layer: _TestLayer) -> Any:
        raise RuntimeError("action failed")

    with pytest.raises(RuntimeError, match="action failed"):
        layers.effect(scope.ctx, _boom, {"label": "store.action", "notify": False})
    assert layers.peek(key) is None

    dispose = layers.effect(
        scope.ctx,
        lambda layer: layer.named.insert("kept", 1),
        {"label": "store.kept", "notify": False},
    )

    def _boom2(_layer: _TestLayer) -> Any:
        raise RuntimeError("second action failed")

    with pytest.raises(RuntimeError, match="second action failed"):
        layers.effect(
            scope.ctx, _boom2, {"label": "store.existing-action", "notify": False}
        )
    kept_layer = layers.peek(key)
    assert kept_layer is not None
    assert kept_layer.named.get("kept") == 1
    dispose()
    await scope.dispose()


@pytest.mark.asyncio
async def test_scoped_layers_rolls_back_insertion_when_notify_throws() -> None:
    """Notification errors undo the insertion and re-notify on rollback."""
    ctx = Context()
    key = object()
    scope = await _mint_scope(ctx, key)
    events: list[str] = []
    notifications = 0

    def _on_change() -> None:
        nonlocal notifications
        events.append("notify")
        notifications += 1
        if notifications == 1:
            raise RuntimeError("change failed")

    layers: ScopedLayers[_TestLayer] = ScopedLayers(
        lambda selected: _TestLayer(selected), _on_change
    )

    def _action(layer: _TestLayer) -> Any:
        undo = layer.named.insert("rollback", 1)

        def _teardown() -> None:
            events.append("undo")
            undo()

        return _teardown

    with pytest.raises(RuntimeError, match="change failed"):
        layers.effect(scope.ctx, _action, {"label": "store.rollback"})

    assert events == ["notify", "undo", "notify"]
    assert layers.peek(key) is None
    await scope.dispose()


@pytest.mark.asyncio
async def test_scoped_layers_chain_layers_ancestors_farthest_first() -> None:
    """chain_layers and merge shadow farthest-ancestor-first up to exact scope."""
    ctx = Context()
    root_key = {"level": "root"}
    child_key = {"level": "child"}
    root_scope = await _mint_scope(ctx, root_key)
    child_scope: Scope | None = None

    def _mount_child(inner: Context) -> None:
        nonlocal child_scope
        child_scope = create_scope(inner, child_key, {"parent": root_key})

    await mount(ctx, _mount_child)
    assert child_scope is not None

    layers: ScopedLayers[_TestLayer] = ScopedLayers(
        lambda selected: _TestLayer(selected), MagicMock()
    )
    layers.global_layer.named.insert("a", 1)
    layers.global_layer.named.insert("b", 1)
    layers.effect(
        root_scope.ctx,
        lambda layer: layer.named.insert("b", 2),
        {"label": "root.b", "notify": False},
    )
    layers.effect(
        child_scope.ctx,
        lambda layer: layer.named.insert("b", 3),
        {"label": "child.b", "notify": False},
    )

    merged = layers.merge(child_key, lambda layer: layer.named)
    assert list(merged.items()) == [("a", 1), ("b", 3)]
    assert [layer is not None for layer in layers.chain_layers(child_key)] == [
        True,
        True,
    ]
