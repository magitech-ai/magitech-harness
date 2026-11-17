"""Unit tests for the pi-ai provider component."""

from __future__ import annotations

from typing import Any

import pytest
from anima import Context

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
