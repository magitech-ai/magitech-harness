"""Unit tests for the pi-ai provider component."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from animas import Context

from magitech_harness.llm.pi_ai.component import apply, directory_entries
from magitech_harness.llm.pi_ai.config import resolve_profiles
from magitech_harness.llm.service import LLMService


def _ctx() -> Context:
    ctx = Context()
    ctx.llm = LLMService(ctx)
    return ctx


def test_directory_entries_cover_every_catalog_provider() -> None:
    """Verifies the directory offers the whole provider catalog."""
    entries = directory_entries({})
    assert len(entries) == 42
    providers = {entry["provider"] for entry in entries}
    assert "deepseek" in providers
    assert "openrouter" in providers
    by_provider = {entry["provider"]: entry for entry in entries}
    assert by_provider["deepseek"]["defaultBaseURL"] == "https://api.deepseek.com"
    assert by_provider["deepseek"]["nativeAuth"] == {
        "env": ["DEEPSEEK_API_KEY"],
        "oauth": False,
    }
    assert by_provider["openrouter"]["declared"] is False
    assert by_provider["deepseek"]["declared"] is True
    assert by_provider["azure"]["baseURLEnv"] == [
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_RESOURCE_NAME",
    ]


def test_directory_entries_include_configured_routes() -> None:
    """Verifies a configured route joins the directory with its own facts."""
    profiles = resolve_profiles(
        {
            "providers": {
                "acme": {
                    "displayName": "Acme Gateway",
                    "apiKeyEnv": "ACME_API_KEY",
                    "api": "openai-completions",
                    "baseURL": "https://gateway.acme.example/v1",
                    "models": [{"id": "acme-large"}],
                }
            }
        }
    )
    entries = {entry["provider"]: entry for entry in directory_entries(profiles)}
    assert entries["acme"]["displayName"] == "Acme Gateway"
    assert entries["acme"]["declared"] is True
    assert entries["acme"]["defaultBaseURL"] == "https://gateway.acme.example/v1"
    assert entries["acme"]["nativeAuth"]["env"] == ["ACME_API_KEY"]


def test_component_mounts_dormant_without_configuration() -> None:
    """Verifies the component registers no route until one is configured."""
    ctx = _ctx()
    apply(ctx)
    assert ctx.llm.list_providers() == []
    assert len(ctx.llm.list_configurable_providers()) == 42


def test_component_registers_configured_routes() -> None:
    """Verifies a configured route reaches the adapter registry."""
    ctx = _ctx()
    apply(ctx, config={"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
    providers = ctx.llm.list_providers()
    assert providers == [{"id": "groq", "name": "groq"}]
    assert ctx.llm.list_configurable_providers()


def test_component_accepts_providers_as_a_keyword() -> None:
    """Verifies the component reads provider profiles passed directly."""
    ctx = _ctx()
    apply(ctx, providers={"groq": {"apiKeyEnv": "GROQ_API_KEY"}})
    assert ctx.llm.list_providers() == [{"id": "groq", "name": "groq"}]


def test_component_accepts_provider_profiles_as_keyword_args() -> None:
    """Verifies the component reads provider profiles passed as keyword args."""
    ctx = _ctx()
    apply(ctx, groq={"apiKeyEnv": "GROQ_API_KEY"})
    assert ctx.llm.list_providers() == [{"id": "groq", "name": "groq"}]


def test_component_withdraws_its_routes_when_unmounted() -> None:
    """Verifies the component releases every route it registered."""
    ctx = _ctx()
    dispose = apply(ctx, config={"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
    assert ctx.llm.list_providers()
    dispose()
    assert ctx.llm.list_providers() == []


def test_component_withdraws_a_dormant_mount_when_unmounted() -> None:
    """Verifies a dormant mount releases its directory entries."""
    ctx = _ctx()
    dispose = apply(ctx)
    assert len(ctx.llm.list_configurable_providers()) == 42
    dispose()
    assert ctx.llm.list_configurable_providers() == []


def test_component_requires_the_llm_service() -> None:
    """Verifies the component refuses to mount without the llm service."""
    ctx = Context()
    with pytest.raises(RuntimeError, match="requires the llm service"):
        apply(ctx)


def test_component_entry_point_declares_its_injection() -> None:
    """Verifies the component declares the service it depends on."""
    assert apply.inject == ["llm"]


def test_component_refuses_an_unserviceable_route() -> None:
    """Verifies a route that cannot serve fails loudly at mount."""
    ctx = _ctx()
    config = {"providers": {"deepseek": {"apiKeyEnv": "DEEPSEEK_API_KEY"}}}
    with pytest.raises(Exception, match="resolves no models"):
        apply(ctx, config=config)


def test_component_configuration_reaches_the_adapter() -> None:
    """Verifies the mounted adapter serves the configured catalog."""
    ctx = _ctx()
    apply(ctx, config={"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
    registration = ctx.llm._adapters["groq"]
    adapter: Any = registration.adapter
    models = adapter._profiles()["groq"].models
    assert "llama-3.1-8b-instant" in models


@pytest.mark.asyncio
async def test_component_refreshes_routes_on_config_update() -> None:
    """A committed config atomically replaces the registered routes."""
    ctx = _ctx()
    dispose = apply(ctx, config={"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
    assert [p["id"] for p in ctx.llm.list_providers()] == ["groq"]

    await ctx.fiber.update(
        {
            "providers": {
                "groq": {"apiKeyEnv": "GROQ_API_KEY"},
                "anthropic": {"apiKeyEnv": "ANTHROPIC_API_KEY"},
            }
        }
    )
    assert sorted(p["id"] for p in ctx.llm.list_providers()) == ["anthropic", "groq"]
    dispose()


@pytest.mark.asyncio
async def test_component_ignores_unchanged_config_update() -> None:
    """A config commit with identical facts does not duplicate registration."""
    import magitech_harness.llm.service as svc

    calls: list[list[str]] = []
    original_replace = svc.AdapterRegistrationHandle.replace

    def spy_replace(self: Any, providers: Sequence[str]) -> None:
        calls.append(list(providers))
        return original_replace(self, providers)

    svc.AdapterRegistrationHandle.replace = spy_replace  # type: ignore[method-assign]
    try:
        ctx = _ctx()
        apply(ctx, config={"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
        assert calls == []  # registration created at mount, not replaced

        await ctx.fiber.update({"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
        # First reconcile: atomic replace of the identical route set.
        assert calls == [["groq"]]
        assert [p["id"] for p in ctx.llm.list_providers()] == ["groq"]

        await ctx.fiber.update({"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
        # Facts unchanged again: no further replace.
        assert calls == [["groq"]]
    finally:
        svc.AdapterRegistrationHandle.replace = original_replace


@pytest.mark.asyncio
async def test_component_clears_routes_when_config_empties() -> None:
    """A config commit that empties the routes withdraws them all."""
    ctx = _ctx()
    apply(ctx, config={"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
    assert [p["id"] for p in ctx.llm.list_providers()] == ["groq"]

    await ctx.fiber.update({"providers": {}})
    assert ctx.llm.list_providers() == []


@pytest.mark.asyncio
async def test_component_registers_routes_added_after_dormant_mount() -> None:
    """A dormant mount picks up routes added by a later config update."""
    ctx = _ctx()
    apply(ctx)  # dormant: directory only, no routes
    assert ctx.llm.list_providers() == []

    await ctx.fiber.update({"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
    assert [p["id"] for p in ctx.llm.list_providers()] == ["groq"]


@pytest.mark.asyncio
async def test_component_stays_dormant_on_empty_config_update() -> None:
    """A dormant mount with an empty config update stays dormant."""
    ctx = _ctx()
    apply(ctx)
    assert ctx.llm.list_providers() == []

    await ctx.fiber.update({"providers": {}})
    assert ctx.llm.list_providers() == []


@pytest.mark.asyncio
async def test_component_keeps_previous_routes_on_config_resolution_failure() -> None:
    """A config commit that cannot resolve leaves the prior routes serving."""
    ctx = _ctx()
    apply(ctx, config={"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
    assert [p["id"] for p in ctx.llm.list_providers()] == ["groq"]

    await ctx.fiber.update(
        {"providers": {"deepseek": {"apiKeyEnv": "DEEPSEEK_API_KEY"}}}
    )
    # deepseek declares no models in the catalog → resolution fails loud.
    # The observer swallows the failure; previous routes keep serving.
    assert [p["id"] for p in ctx.llm.list_providers()] == ["groq"]


@pytest.mark.asyncio
async def test_component_replaces_routes_on_existing_registration() -> None:
    """A config commit that swaps routes replaces through the live handle."""
    ctx = _ctx()
    apply(ctx, config={"providers": {"groq": {"apiKeyEnv": "GROQ_API_KEY"}}})
    assert [p["id"] for p in ctx.llm.list_providers()] == ["groq"]

    await ctx.fiber.update(
        {"providers": {"anthropic": {"apiKeyEnv": "ANTHROPIC_API_KEY"}}}
    )
    assert [p["id"] for p in ctx.llm.list_providers()] == ["anthropic"]


@pytest.mark.asyncio
async def test_component_replaces_directory_on_display_name_change() -> None:
    """A config commit that changes directory facts atomically replaces entries."""
    ctx = _ctx()
    apply(
        ctx,
        config={
            "providers": {
                "groq": {"apiKeyEnv": "GROQ_API_KEY", "displayName": "Groq One"}
            }
        },
    )
    before = {
        e["provider"]: e["displayName"] for e in ctx.llm.list_configurable_providers()
    }
    assert before["groq"] == "Groq One"

    await ctx.fiber.update(
        {
            "providers": {
                "groq": {"apiKeyEnv": "GROQ_API_KEY", "displayName": "Groq Two"}
            }
        }
    )
    after = {
        e["provider"]: e["displayName"] for e in ctx.llm.list_configurable_providers()
    }
    assert after["groq"] == "Groq Two"
