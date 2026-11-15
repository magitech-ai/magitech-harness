"""Unit tests for credential-scoped ModelListingCache."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest

from magitech_harness.agent.cancellation import AbortController, AbortError
from magitech_harness.llm.listing_cache import ListingSettings, ModelListingCache


@pytest.mark.asyncio
async def test_listing_cache_first_load_single_flight_and_scoping() -> None:
    """First access awaits load with single-flight dedupe and SHA-256 credential scope."""
    clock = 1_000.0
    cred: str | None = "secret-a"
    active_settings: ListingSettings | None = {
        "url": "https://api.example.test/models",
        "timeoutMs": 5_000,
        "maxAgeMs": 600_000,
    }
    infos: list[str] = []
    warns: list[str] = []
    changes: int = 0
    calls: list[tuple[str, str]] = []
    gate = asyncio.Event()

    async def _load(url: str, credential: str, signal: Any) -> dict[str, Any]:
        del signal
        calls.append((url, credential))
        await gate.wait()
        return {"entries": [{"id": f"m-{credential}"}], "skipped": 1}

    cache = ModelListingCache(
        {
            "settings": lambda: active_settings,
            "credential": lambda _url: cred,
            "load": _load,
            "now": lambda: clock,
            "info": infos.append,
            "warn": warns.append,
            "changed": lambda: _inc_changes(),
        }
    )

    def _inc_changes() -> None:
        nonlocal changes
        changes += 1

    assert cache.current() == {"authenticated": False, "listing": None}

    task1 = asyncio.create_task(cache.view(active_settings))
    await asyncio.sleep(0)
    task2 = asyncio.create_task(cache.view(active_settings))
    await asyncio.sleep(0)
    assert len(calls) == 1

    gate.set()
    view1, view2 = await asyncio.gather(task1, task2)
    assert view1 == {
        "authenticated": True,
        "listing": {"entries": [{"id": "m-secret-a"}], "skipped": 1},
    }
    assert view2 == view1
    assert changes == 1
    assert len(infos) == 1
    assert warns == []

    # Same credential digest + URL within maxAgeMs does NOT reload
    assert await cache.view(active_settings) == view1
    assert len(calls) == 1
    assert cache.scope_key("https://api.example.test/models", "secret-a") == (
        "https://api.example.test/models\n" + hashlib.sha256(b"secret-a").hexdigest()
    )

    # Credential change rescopes immediately and forgets prior catalog
    cred = "secret-b"
    view_b = await cache.view(active_settings)
    assert len(calls) == 2
    assert view_b == {
        "authenticated": True,
        "listing": {"entries": [{"id": "m-secret-b"}], "skipped": 1},
    }
    assert changes == 2

    # Missing credential clears scope and unauthenticates
    cred = None
    unauth = await cache.view(active_settings)
    assert unauth == {"authenticated": False, "listing": None}

    await cache.dispose()


@pytest.mark.asyncio
async def test_listing_cache_background_refresh_last_known_good_and_abandonable() -> (
    None
):
    """Covers background refresh, failing streak retention, abandonable wait, and dispose."""
    clock = 10_000.0
    settings: ListingSettings | None = {
        "url": "https://api.example.test/models",
        "timeoutMs": 5_000,
        "maxAgeMs": 1_000,
    }
    infos: list[str] = []
    warns: list[str] = []
    changes = 0
    mode = "ok-v1"
    blocker = asyncio.Event()

    async def _load(url: str, credential: str, signal: Any) -> dict[str, Any]:
        del url, credential
        if mode == "block":
            while not signal.aborted and not blocker.is_set():
                await asyncio.sleep(0.01)
            return {"entries": [{"id": "late"}], "skipped": 0}
        if mode == "fail":
            raise RuntimeError("upstream 500")
        if mode == "ok-v2":
            return {"entries": [{"id": "v2"}], "skipped": 0}
        return {"entries": [{"id": "v1"}], "skipped": 0}

    def _changed() -> None:
        nonlocal changes
        changes += 1
        if mode == "throw-changed":
            raise RuntimeError("listener boom")

    cache = ModelListingCache(
        {
            "settings": lambda: settings,
            "credential": lambda _u: "k1",
            "load": _load,
            "now": lambda: clock,
            "info": infos.append,
            "warn": warns.append,
            "changed": _changed,
        }
    )

    # First access can be abandoned via signal while shared load continues
    mode = "block"
    ac = AbortController()
    waiter = asyncio.create_task(cache.view(settings, ac.signal))
    await asyncio.sleep(0.02)
    ac.abort("caller-cancelled")
    with pytest.raises(AbortError, match="caller-cancelled"):
        await waiter
    blocker.set()
    await asyncio.sleep(0.02)
    assert cache.current()["listing"] == {"entries": [{"id": "late"}], "skipped": 0}

    # Invalidate resets scope and allows clean v1 load
    cache.invalidate()
    mode = "ok-v1"
    assert await cache.view(settings) == {
        "authenticated": True,
        "listing": {"entries": [{"id": "v1"}], "skipped": 0},
    }

    # First access with non-aborted signal resolves normally via task.result()
    cache.invalidate()
    ac_ok = AbortController()
    view_ok_sig = await cache.view(settings, ac_ok.signal)
    assert view_ok_sig["listing"] == {"entries": [{"id": "v1"}], "skipped": 0}

    # Same payload reload on stale access skips change notification
    clock += 1_500.0
    prev_changes_same = changes
    await cache.view(settings)
    await asyncio.sleep(0.02)
    assert changes == prev_changes_same

    # Stale access triggers background refresh, returning stale immediately
    clock += 1_500.0
    mode = "fail"
    stale_view = await cache.view(settings)
    assert stale_view["listing"] == {"entries": [{"id": "v1"}], "skipped": 0}
    await asyncio.sleep(0.02)
    assert len(warns) == 1
    assert "keeping the previous catalog" in warns[0]

    # Repeated failures in streak do NOT spam warns
    clock += 1_500.0
    await cache.view(settings)
    await asyncio.sleep(0.02)
    assert len(warns) == 1

    # Recovery to v2 emits change notification once
    prev_changes = changes
    clock += 1_500.0
    mode = "ok-v2"
    await cache.view(settings)
    await asyncio.sleep(0.02)
    assert changes == prev_changes + 1
    assert cache.current()["listing"] == {"entries": [{"id": "v2"}], "skipped": 0}

    # Listener exception during notification is contained and logged
    clock += 1_500.0
    mode = "throw-changed"
    await cache.view(settings)
    await asyncio.sleep(0.02)
    assert any("change notification failed" in w for w in warns)

    await cache.dispose()
    assert await cache.view(settings) == cache.current()


@pytest.mark.asyncio
async def test_listing_cache_auto_refresh_invalidate_races_and_dispose() -> None:
    """Covers auto-refresh timer paths, generation races, maxAgeMs=0, and active dispose."""
    clock = 0.0
    cred_mode = "ok"
    settings: ListingSettings | None = {
        "url": "https://api.example.test/models",
        "timeoutMs": 5_000,
        "maxAgeMs": 20,
    }
    warns: list[str] = []
    changes = 0
    slow_gate = asyncio.Event()
    slow_fail_gate = asyncio.Event()

    async def _cred(_url: str) -> str | None:
        if cred_mode == "none":
            return None
        if cred_mode == "raise":
            raise RuntimeError("vault locked")
        if cred_mode == "raise-on-close":
            await asyncio.sleep(0.02)
            raise RuntimeError("late cred err")
        return "tok-1"

    async def _load(url: str, credential: str, signal: Any) -> dict[str, Any]:
        del url, credential, signal
        if cred_mode == "slow-resolve":
            await slow_gate.wait()
        elif cred_mode == "slow-reject":
            await slow_fail_gate.wait()
            raise RuntimeError("late reject")
        return {"entries": [{"id": "m1"}], "skipped": 0}

    cache = ModelListingCache(
        {
            "settings": lambda: settings,
            "credential": _cred,
            "load": _load,
            "now": lambda: clock,
            "info": lambda _m: None,
            "warn": warns.append,
            "changed": lambda: _bump(),
        }
    )

    def _bump() -> None:
        nonlocal changes
        changes += 1

    # Disposing an untouched cache is a safe no-op
    await cache.dispose()

    # Re-create active cache and test maxAgeMs=0 (no auto-refresh scheduled)
    cache2 = ModelListingCache(
        {
            "settings": lambda: settings,
            "credential": _cred,
            "load": _load,
            "now": lambda: clock,
            "info": lambda _m: None,
            "warn": warns.append,
            "changed": _bump,
        }
    )
    settings_zero: ListingSettings = {
        "url": "https://api.example.test/models",
        "timeoutMs": 5_000,
        "maxAgeMs": 0,
    }
    assert await cache2.view(settings_zero) == {
        "authenticated": True,
        "listing": {"entries": [{"id": "m1"}], "skipped": 0},
    }

    # Test stale generation skip on both resolve and reject
    cache2.invalidate()
    cred_mode = "slow-resolve"
    t_res = asyncio.create_task(cache2.view(settings))
    await asyncio.sleep(0.01)
    cache2.invalidate()
    slow_gate.set()
    await asyncio.sleep(0.01)
    t_res.cancel()

    cred_mode = "slow-reject"
    t_rej = asyncio.create_task(cache2.view(settings))
    await asyncio.sleep(0.01)
    cache2.invalidate()
    slow_fail_gate.set()
    await asyncio.sleep(0.01)
    t_rej.cancel()

    # Auto-refresh timer fires and sees credential error streak + recovery + unauth + settings=None
    cred_mode = "ok"
    settings = {
        "url": "https://api.example.test/models",
        "timeoutMs": 5_000,
        "maxAgeMs": 15,
    }
    cache2.invalidate()
    await cache2.view(settings)
    changes_before = changes
    cred_mode = "raise"
    await asyncio.sleep(0.03)
    assert any("refresh failed" in w for w in warns)
    warn_count_before = len(warns)
    await asyncio.sleep(0.03)
    assert len(warns) == warn_count_before  # failing streak does not spam

    # Refresh when credential disappears notifies change
    cred_mode = "none"
    await asyncio.sleep(0.03)
    assert changes >= changes_before + 1

    # Refresh when settings becomes None invalidates & notifies
    cred_mode = "ok"
    await cache2.view(settings)
    changes_before2 = changes
    settings = None
    await asyncio.sleep(0.03)
    assert changes >= changes_before2 + 1
    assert cache2.current() == {"authenticated": True, "listing": None}
    await cache2.dispose()

    # Refresh closes cleanly if disposed mid-refresh (returning or raising)
    async def _run_mid_close(final_mode: str) -> None:
        local_mode = "ok"
        local_settings: ListingSettings = {
            "url": "https://api.example.test/models",
            "timeoutMs": 5_000,
            "maxAgeMs": 10,
        }

        async def _cred_c(_u: str) -> str | None:
            if local_mode == "close-mid-return":
                await asyncio.sleep(0.015)
                return "tok-late"
            if local_mode == "raise-on-close":
                await asyncio.sleep(0.015)
                raise RuntimeError("late cred err")
            return "tok-1"

        cache_c = ModelListingCache(
            {
                "settings": lambda: local_settings,
                "credential": _cred_c,
                "load": _load,
                "now": lambda: clock,
                "info": lambda _m: None,
                "warn": warns.append,
                "changed": _bump,
            }
        )
        await cache_c.view(local_settings)
        local_mode = final_mode
        await asyncio.sleep(0.015)
        await cache_c.dispose()

    await _run_mid_close("close-mid-return")
    await _run_mid_close("raise-on-close")

    # Dispose while load is in flight waits for pending tasks; also cover concurrent refresh
    settings4: ListingSettings = {
        "url": "https://api.example.test/models",
        "timeoutMs": 5_000,
        "maxAgeMs": 10,
    }

    async def _wait_load(url: str, cred: str, signal: Any) -> dict[str, Any]:
        del url, cred, signal
        await asyncio.sleep(0.02)
        return {"entries": [], "skipped": 0}

    cache4 = ModelListingCache(
        {
            "settings": lambda: settings4,
            "credential": lambda _u: "k4",
            "load": _wait_load,
            "now": lambda: clock,
            "info": lambda _m: None,
            "warn": warns.append,
            "changed": _bump,
        }
    )
    t4 = asyncio.create_task(cache4.view(settings4))
    await asyncio.sleep(0.005)
    await cache4.dispose()
    await t4

    # Cover _refresh completing while another inflight load is already active
    cred_step = 0

    async def _cred5(_u: str) -> str | None:
        nonlocal cred_step
        cred_step += 1
        if cred_step == 2:
            await asyncio.sleep(0.02)
            raise RuntimeError("concurrent refresh fail")
        return "k5"

    cache5 = ModelListingCache(
        {
            "settings": lambda: settings4,
            "credential": _cred5,
            "load": _wait_load,
            "now": lambda: clock,
            "info": lambda _m: None,
            "warn": warns.append,
            "changed": _bump,
        }
    )
    await cache5.view(settings4)
    await asyncio.sleep(0.015)
    cache5.invalidate()
    concurrent_t = asyncio.create_task(cache5.view(settings4))
    await asyncio.sleep(0.03)
    await concurrent_t
    await cache5.dispose()
