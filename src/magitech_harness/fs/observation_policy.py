"""Event-only filesystem observation policy enforcing read-before-edit/delete."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any, cast
from weakref import ReferenceType, ref

from anima import Context, EventMode

from magitech_harness.fs.types import (
    FsError,
    FsObservation,
    FsTarget,
    FsVersion,
    FsWriteIntent,
)


def _owner_of(actor: Any) -> object | None:
    if actor is None:
        return None
    agent = (
        actor.get("agent")
        if isinstance(actor, Mapping)
        else getattr(actor, "agent", None)
    )
    if agent is None:
        return None
    session = (
        agent.get("session")
        if isinstance(agent, Mapping)
        else getattr(agent, "session", None)
    )
    return cast(object | None, session)


class ObservedStateGate:
    """Per-session recorded target presence/absence state."""

    def __init__(self) -> None:
        self._observed: dict[
            int, tuple[ReferenceType[object], dict[str, FsObservation]]
        ] = {}

    def clear(self) -> None:
        """Drop all recorded state (HMR safety / disposal)."""
        self._observed.clear()

    def _get_observations(
        self, owner: object | None
    ) -> dict[str, FsObservation] | None:
        if owner is None:
            return None
        entry = self._observed.get(id(owner))
        if entry is None or entry[0]() is not owner:
            return None
        return entry[1]

    def write_intent(self, target: FsTarget, actor: Any) -> FsWriteIntent:
        """Decide guarded write intent from prior observation."""
        owner = _owner_of(actor)
        bucket = self._get_observations(owner)
        prior = bucket.get(target["targetKey"]) if bucket is not None else None
        if prior is not None and prior["kind"] == "present":
            return {"kind": "replaceIfVersion", "version": prior["version"]}
        return {"kind": "createIfAbsent"}

    def edit_intent(self, target: FsTarget, actor: Any) -> dict[str, FsVersion]:
        """Require prior present observation before editing."""
        return self._require_present(target, actor, "edit")

    def delete_intent(self, target: FsTarget, actor: Any) -> dict[str, FsVersion]:
        """Require prior present observation before deleting."""
        return self._require_present(target, actor, "delete")

    def _require_present(
        self, target: FsTarget, actor: Any, verb: str
    ) -> dict[str, FsVersion]:
        owner = _owner_of(actor)
        bucket = self._get_observations(owner)
        prior = bucket.get(target["targetKey"]) if bucket is not None else None
        if owner is None or prior is None:
            raise FsError(
                f'{verb} requires reading "{target["displayPath"]}" first',
                "FS_NOT_OBSERVED",
            )
        if prior["kind"] == "absent":
            raise FsError(
                f'cannot {verb} "{target["displayPath"]}": not found',
                "FS_NOT_FOUND",
            )
        return {"version": prior["version"]}

    def observe(self, target: FsTarget, observation: FsObservation, actor: Any) -> None:
        """Record an authoritative presence or absence observation for *actor*."""
        owner = _owner_of(actor)
        if owner is None:
            return
        key = id(owner)
        by_target = self._get_observations(owner)
        if by_target is None:
            by_target = {}
            store = self._observed
            store[key] = (
                ref(
                    owner,
                    cast(
                        Callable[[ReferenceType[Any]], Any],
                        partial(store.pop, key, None),
                    ),
                ),
                by_target,
            )
        by_target[target["targetKey"]] = observation


def apply(ctx: Context, config: dict[str, Any] | None = None) -> None:
    """Install event-only prior-observation write/edit/delete guards onto *ctx*."""
    del config
    gate = ObservedStateGate()

    ctx.effect(
        lambda: gate.clear, label="fs-observation-policy observed-state teardown"
    )

    async def _on_write(
        target: FsTarget,
        actor: Any,
        next_fn: Callable[[], Awaitable[Any]],
    ) -> FsWriteIntent:
        del next_fn
        return gate.write_intent(target, actor)

    async def _on_edit(
        target: FsTarget,
        actor: Any,
        next_fn: Callable[[], Awaitable[Any]],
    ) -> dict[str, FsVersion]:
        del next_fn
        return gate.edit_intent(target, actor)

    async def _on_delete(
        target: FsTarget,
        actor: Any,
        next_fn: Callable[[], Awaitable[Any]],
    ) -> dict[str, FsVersion]:
        del next_fn
        return gate.delete_intent(target, actor)

    def _on_observed(target: FsTarget, observation: FsObservation, actor: Any) -> None:
        gate.observe(target, observation, actor)

    ctx.on("fs/write-intent", _on_write, mode=EventMode.WATERFALL)
    ctx.on("fs/edit-intent", _on_edit, mode=EventMode.WATERFALL)
    ctx.on("fs/delete-intent", _on_delete, mode=EventMode.WATERFALL)
    ctx.on("fs/observed", _on_observed)


__all__ = ["ObservedStateGate", "apply"]
