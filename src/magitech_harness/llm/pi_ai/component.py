"""Anima component entry point for the pi-ai provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from anima import Context, Disposer

from magitech_harness.llm.pi_ai.adapter import PiAiAdapter
from magitech_harness.llm.pi_ai.config import ResolvedProfile, resolve_profiles
from magitech_harness.llm.pi_ai.provider_data import LANES, PROVIDER_DATA
from magitech_harness.llm.types import LlmConfigurableProvider

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


def apply(
    ctx: Context,
    config: Mapping[str, Any] | None = None,
    providers: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Disposer:
    """Mount the pi-ai multi-provider adapter onto ctx.llm.

    Args:
        ctx: Active Anima Context.
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

    def _teardown() -> None:
        if registration is not None:
            registration()
        directory_handle()

    return ctx.effect(lambda: _teardown)


apply.inject = ["llm"]  # type: ignore[attr-defined]
