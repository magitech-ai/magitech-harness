"""Animas component entry point for the pi-ai provider."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from animas import Context, Disposer, EventMode

from magitech_harness.llm.pi_ai.adapter import PiAiAdapter
from magitech_harness.llm.pi_ai.config import ResolvedProfile, resolve_profiles
from magitech_harness.llm.pi_ai.provider_data import LANES, PROVIDER_DATA
from magitech_harness.llm.types import LlmConfigurableProvider

logger = logging.getLogger("magitech_harness.llm.pi_ai")

SETTINGS_NS = "llm-pi-ai"


def directory_entries(
    profiles: Mapping[str, ResolvedProfile],
    settings_ns: str = SETTINGS_NS,
) -> list[LlmConfigurableProvider]:
    """Build configurable-provider directory entries for every known provider."""
    entries: dict[str, LlmConfigurableProvider] = {}
    for provider, data in PROVIDER_DATA.items():
        entry: LlmConfigurableProvider = {
            "provider": provider,
            "displayName": provider,
            "settingsNs": settings_ns,
            "settingsPath": ["providers", provider],
            "declared": provider not in LANES,
            "nativeAuth": {"env": list(data.get("env", [])), "oauth": False},
        }
        default_base_url = data.get("defaultBaseURL")
        if default_base_url:
            entry["defaultBaseURL"] = default_base_url
        base_url_env = data.get("baseURLEnv")
        if base_url_env:
            entry["baseURLEnv"] = list(base_url_env)
        entries[provider] = entry
    for provider, profile in profiles.items():
        declared_entry: LlmConfigurableProvider = {
            "provider": provider,
            "displayName": profile.display_name,
            "settingsNs": settings_ns,
            "settingsPath": ["providers", provider],
            "declared": True,
            "nativeAuth": {
                "env": list(
                    PROVIDER_DATA.get(provider, {}).get("env", [])
                    or ([profile.api_key_env] if profile.api_key_env else [])
                ),
                "oauth": False,
            },
        }
        base_url = profile.profile.resolved_base_url()
        if base_url:
            declared_entry["defaultBaseURL"] = base_url
        entries[provider] = declared_entry
    return list(entries.values())


def _directory_facts(entries: Sequence[LlmConfigurableProvider]) -> tuple[Any, ...]:
    """Stable, comparable snapshot of directory facts for change detection."""
    return tuple(
        sorted(
            (
                entry.get("provider"),
                entry.get("displayName"),
                entry.get("settingsNs"),
                tuple(entry.get("settingsPath", ())),
            )
            for entry in entries
        )
    )


def _route_facts(resolved: Mapping[str, ResolvedProfile]) -> tuple[Any, ...]:
    """Stable, comparable snapshot of route facts for change detection."""
    return tuple(
        sorted(
            (
                provider,
                profile.retry_policy.get("mode") if profile.retry_policy else None,
            )
            for provider, profile in resolved.items()
        )
    )


def apply(
    ctx: Context,
    config: Mapping[str, Any] | None = None,
    providers: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Disposer:
    """Mount the pi-ai multi-provider adapter onto ctx.llm.

    The mount is reactive: whenever this fiber's config is updated through
    Animas's rc.13 native lifecycle (``Fiber.update`` -> ``internal/update``
    waterfall), the directory entries and adapter routes are atomically
    re-resolved from the fiber's live config and re-registered only when the
    facts actually changed (atomic replace, never dispose-then-register,
    matching the documented contract). A bare mount with zero routes stays
    dormant (directory only).

    Args:
        ctx: Active Animas Context.
        config: Optional component configuration carrying ``providers`` profiles.
        providers: Optional provider profiles keyed by route name.
        **kwargs: Unused extra component configuration.

    Returns:
        A disposer that withdraws every route and directory entry this mount
        registered.
    """
    raw: dict[str, Any] = {}
    if config:
        raw.update(config)
    if providers is not None:
        raw.setdefault("providers", dict(providers))
    if kwargs:
        raw.setdefault("providers", dict(kwargs))

    resolved = resolve_profiles(raw)
    adapter = PiAiAdapter(profiles=lambda: resolved)

    llm = ctx.get("llm", None, strict=False)
    if llm is None:
        raise RuntimeError("pi-ai provider requires the llm service")

    directory_handle = llm.register_configurable_providers(directory_entries(resolved))
    registration = None
    routes = list(resolved)
    if routes:
        registration = llm.register_adapter(routes, adapter)

    current_dir_facts = _directory_facts(directory_entries(resolved))
    # Sentinel: "no route facts established yet" so an empty config update
    # on a dormant mount is still reconciled (branch coverage).
    current_route_facts: Any = None

    fiber = ctx.fiber

    def refresh() -> None:
        """Re-resolve from the fiber's live config and replace on change."""
        nonlocal directory_handle, registration
        nonlocal current_dir_facts, current_route_facts, resolved
        live = dict(fiber.raw_config or {})
        new_resolved = resolve_profiles(live)
        new_entries = directory_entries(new_resolved)
        new_dir_facts = _directory_facts(new_entries)
        new_routes = list(new_resolved)
        new_route_facts = _route_facts(new_resolved)
        if new_dir_facts != current_dir_facts:
            directory_handle.replace(new_entries)
            current_dir_facts = new_dir_facts
        if new_route_facts != current_route_facts:
            if registration is None:
                if new_routes:
                    registration = llm.register_adapter(new_routes, adapter)
            else:
                registration.replace(new_routes)
            current_route_facts = new_route_facts
        resolved = new_resolved

    def on_config_update(_resolved: Any) -> None:
        """Accept this fiber's config update in place and reconcile.

        Subscribes to the rc.13 ``internal/update`` waterfall scoped to this
        mount's fiber; the update is accepted in place (the default handler
        that restarts the fiber does not run) and the directory entries and
        adapter routes are atomically re-resolved from the fiber's live
        config, replacing only when the facts actually changed.

        A resolution failure leaves the previous registration serving (the
        reference logs the conflict and keeps prior state).
        """
        try:
            refresh()
        except Exception:  # observer must not crash the emitter
            logger.exception(
                "pi-ai: config update conflicts with existing provider route"
            )

    config_update_listener = ctx.on(
        "internal/update", on_config_update, mode=EventMode.WATERFALL
    )

    def _teardown() -> None:
        config_update_listener()
        if registration is not None:
            registration()
        directory_handle()

    return ctx.effect(lambda: _teardown)


apply.inject = ["llm"]  # type: ignore[attr-defined]
