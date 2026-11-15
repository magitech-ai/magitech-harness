"""Credential-scoped /models cache with single-flight loads and last-known-good."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypedDict

from magitech_harness.agent.cancellation import (
    MAX_TIMER_DELAY_MS,
    AbortController,
    AbortError,
    AbortSignal,
    deadline,
)
from magitech_harness.llm.listing_config import LISTING_TIMEOUT, ListingSettings
from magitech_harness.llm.values import deep_equal_json


class ListingCacheDependencies(TypedDict):
    """Host-owned inputs driving one ModelListingCache instance."""

    settings: Callable[[], ListingSettings | None]
    credential: Callable[[str], str | None | Awaitable[str | None]]
    load: Callable[[str, str, AbortSignal], Awaitable[Any]]
    now: Callable[[], float]
    info: Callable[[str], None]
    warn: Callable[[str], None]
    changed: Callable[[], None]


class ListingView(TypedDict):
    """Snapshot returned by ModelListingCache.current() and view()."""

    authenticated: bool
    listing: Any | None


async def _abandonable(
    task: asyncio.Task[None],
    signal: AbortSignal | None,
) -> None:
    """Await *task* unless *signal* aborts first; *task* keeps running."""
    if signal is None:
        await task
        return
    signal.throw_if_aborted()
    loop = asyncio.get_running_loop()
    aborted_fut: asyncio.Future[None] = loop.create_future()

    def _on_abort() -> None:
        msg = (
            f"Operation was aborted: {signal.reason}"
            if signal.reason
            else "Operation was aborted."
        )
        aborted_fut.set_exception(AbortError(msg))

    unlisten = signal.on_abort(_on_abort)
    try:
        done, _ = await asyncio.wait(
            {task, aborted_fut},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if aborted_fut in done:
            aborted_fut.result()
        else:
            task.result()
    finally:
        unlisten()


class ModelListingCache:
    """Credential-scoped model listing cache with single-flight background refresh."""

    def __init__(self, dependencies: ListingCacheDependencies) -> None:
        self._dependencies = dependencies
        self._scope: str | None = None
        self._listing: Any | None = None
        self._settled_at: float | None = None
        self._inflight: asyncio.Task[None] | None = None
        self._request: AbortController | None = None
        self._generation = 0
        self._failing = False
        self._closed = False
        self._timer: asyncio.TimerHandle | None = None
        self._pending: set[asyncio.Task[None]] = set()

    @staticmethod
    def scope_key(url: str, credential: str) -> str:
        """Compute canonical endpoint + SHA-256(credential) cache scope key."""
        digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()
        return f"{url}\n{digest}"

    def current(self) -> ListingView:
        """Return the last published view without performing I/O."""
        return {
            "authenticated": self._scope is not None,
            "listing": self._listing,
        }

    async def view(
        self,
        settings: ListingSettings,
        signal: AbortSignal | None = None,
    ) -> ListingView:
        """Resolve view for current credential, waiting on first load only."""
        raw_cred = self._dependencies["credential"](settings["url"])
        cred = await raw_cred if inspect.isawaitable(raw_cred) else raw_cred
        if signal is not None:
            signal.throw_if_aborted()
        if self._closed:
            return self.current()
        scope = None if cred is None else self.scope_key(settings["url"], cred)
        if scope != self._scope:
            self._rescope(scope)
        if cred is not None:
            if self._settled_at is None:
                await _abandonable(self._start(settings, cred), signal)
            elif self._dependencies["now"]() - self._settled_at >= settings["maxAgeMs"]:
                self._start(settings, cred)
        return self.current()

    def invalidate(self) -> None:
        """Drop cached listing and abort any active load."""
        self._rescope(self._scope)

    async def dispose(self) -> None:
        """Stop automatic refresh, abort in-flight loads, and wait for tasks."""
        self._closed = True
        self._generation += 1
        self._cancel_timer()
        if self._request is not None:
            self._request.abort()
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _rescope(self, scope: str | None) -> None:
        self._cancel_timer()
        if self._request is not None:
            self._request.abort()
        self._scope = scope
        self._listing = None
        self._settled_at = None
        self._inflight = None
        self._failing = False
        self._generation += 1

    def _start(
        self,
        settings: ListingSettings,
        credential: str,
    ) -> asyncio.Task[None]:
        if self._inflight is not None:
            return self._inflight
        self._cancel_timer()
        self._generation += 1
        generation = self._generation
        request = AbortController()
        self._request = request
        limit = deadline(request.signal, settings["timeoutMs"], LISTING_TIMEOUT)
        loop = asyncio.get_running_loop()

        async def _run() -> None:
            try:
                listing = await self._dependencies["load"](
                    settings["url"],
                    credential,
                    limit.signal,
                )
            except Exception as exc:  # noqa: BLE001 - retain last-known-good catalog
                if generation != self._generation:
                    return
                if not self._failing:
                    fallback = (
                        "using the bundled catalog"
                        if self._listing is None
                        else "keeping the previous catalog"
                    )
                    self._dependencies["warn"](
                        f"model listing from {settings['url']} failed ({exc}); "
                        f"{fallback}"
                    )
                self._settled_at = self._dependencies["now"]()
                self._failing = True
            else:
                if generation != self._generation:
                    return
                changed = not deep_equal_json(self._listing, listing)
                self._listing = listing
                self._settled_at = self._dependencies["now"]()
                self._failing = False
                entries = (
                    listing.get("entries", ()) if isinstance(listing, Mapping) else ()
                )
                skipped = (
                    listing.get("skipped", 0) if isinstance(listing, Mapping) else 0
                )
                self._dependencies["info"](
                    f"model listing from {settings['url']} loaded {len(entries)} "
                    f"model(s), skipped {skipped} row(s)"
                )
                if changed:
                    self._notify()
            finally:
                limit.dispose()
                if self._inflight is task:
                    self._inflight = None
                if generation == self._generation:
                    self._schedule(settings)

        task = loop.create_task(_run())
        self._inflight = task
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    def _notify(self) -> None:
        try:
            self._dependencies["changed"]()
        except Exception as exc:  # noqa: BLE001 - observer error containment
            self._dependencies["warn"](
                f"model listing change notification failed ({exc}); "
                "reload the model catalog"
            )

    def _schedule(self, settings: ListingSettings) -> None:
        self._cancel_timer()
        if self._closed or settings["maxAgeMs"] == 0:
            return
        age = (
            0.0
            if self._settled_at is None
            else self._dependencies["now"]() - self._settled_at
        )
        delay_ms = min(MAX_TIMER_DELAY_MS, max(1.0, settings["maxAgeMs"] - age))
        loop = asyncio.get_running_loop()

        def _fire() -> None:
            self._timer = None
            op = loop.create_task(self._refresh(settings))
            self._pending.add(op)
            op.add_done_callback(self._pending.discard)

        self._timer = loop.call_later(delay_ms / 1000.0, _fire)

    async def _refresh(self, previous: ListingSettings) -> None:
        settings = previous
        try:
            next_settings = self._dependencies["settings"]()
            if next_settings is None:
                self.invalidate()
                self._notify()
                return
            settings = next_settings
            was_authenticated = self.current()["authenticated"]
            await self.view(settings)
            if self._closed:
                return
            if was_authenticated and not self.current()["authenticated"]:
                self._notify()
        except Exception as exc:  # noqa: BLE001 - refresh failure containment
            if self._closed:
                return
            if not self._failing:
                self._dependencies["warn"](
                    f"model listing refresh failed ({exc}); keeping the previous "
                    "catalog; check listing configuration and credentials"
                )
            self._failing = True
            self._settled_at = self._dependencies["now"]()
        if self._inflight is None:
            self._schedule(settings)


__all__ = [
    "ListingCacheDependencies",
    "ListingSettings",
    "ListingView",
    "ModelListingCache",
]
