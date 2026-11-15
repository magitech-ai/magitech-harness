"""Scoped job-event hub delivering filtered lifecycle notifications."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from anima import Context

from magitech_harness.jobs.types import JobEvent, JobEventFilter
from magitech_harness.scope import (
    AnonymousEntries,
    ScopedLayers,
    ScopeKey,
    scope_of,
)


class JobLayer:
    """Per-scope controllers and ``owners: 'scope'`` subscriptions."""

    def __init__(self, _scope: ScopeKey | None = None) -> None:
        self.controllers: AnonymousEntries[object] = AnonymousEntries()
        self.scoped: AnonymousEntries[_Subscription] = AnonymousEntries()

    def is_empty(self) -> bool:
        """Whether this layer holds no controllers and no subscriptions."""
        return self.controllers.is_empty() and self.scoped.is_empty()


@dataclass(eq=False, slots=True)
class _Subscription:
    filter_spec: JobEventFilter
    listener: Callable[[JobEvent], Any]


class JobEventHub:
    """Delivers job events across unscoped and owner-scoped subscription layers."""

    def __init__(
        self,
        layers: ScopedLayers[JobLayer],
        warn: Callable[[str], None],
    ) -> None:
        self._layers = layers
        self._warn = warn
        self._unscoped: list[_Subscription] = []

    def subscribe(
        self,
        registrar: Context,
        filter_spec: JobEventFilter | None,
        listener: Callable[[JobEvent], Any],
    ) -> Callable[[], Awaitable[None] | None]:
        """Attach a filtered subscription bound to *registrar* fiber lifecycle."""
        effective: JobEventFilter = (
            {"owners": "all"} if filter_spec is None else filter_spec
        )
        sub = _Subscription(filter_spec=effective, listener=listener)
        if effective.get("owners") == "scope":
            return cast(
                Callable[[], Awaitable[None] | None],
                self._layers.effect(
                    registrar,
                    lambda layer: layer.scoped.append(sub),
                    {"label": "jobs.events.subscribe()", "notify": False},
                ),
            )

        def _register_unscoped() -> Callable[[], None]:
            self._unscoped.append(sub)

            def _remove() -> None:
                self._unscoped = [s for s in self._unscoped if s is not sub]

            return _remove

        return cast(
            Callable[[], Awaitable[None] | None],
            registrar.effect(_register_unscoped, "jobs.events.subscribe()"),
        )

    def _deliver(self, sub: _Subscription, event: JobEvent) -> None:
        try:
            sub.listener(event)
        except Exception as exc:  # noqa: BLE001 - isolate listeners
            self._warn(f"jobs: event listener threw on {event['type']}: {exc}")

    def emit(self, event: JobEvent, owner: Any) -> None:
        """Dispatch *event* to matching listeners; listener exceptions are logged."""
        owner_id = getattr(owner, "id", None) if owner is not None else None
        for sub in list(self._unscoped):
            if (
                "owner" in sub.filter_spec
                and owner_id is not None
                and owner_id != sub.filter_spec["owner"]
            ):
                continue
            self._deliver(sub, event)

        for sub in self._layers.global_layer.scoped.values():
            self._deliver(sub, event)

        owner_ctx = getattr(owner, "ctx", None) if owner is not None else None
        scope = scope_of(owner_ctx) if owner_ctx is not None else None
        for layer in self._layers.chain_layers(scope):
            for sub in layer.scoped.values():
                self._deliver(sub, event)


__all__ = ["JobEventHub", "JobLayer"]
