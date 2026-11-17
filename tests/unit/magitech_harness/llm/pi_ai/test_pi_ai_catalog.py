"""Unit tests for the pi-ai model catalog and route resolution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.catalog import (
    DEFAULT_INPUT,
    ModelSpec,
    _merge_model,
    lane_catalog,
    load_lane_catalog,
    model_info,
    provider_info,
    resolve_reasoning_effort,
    resolve_route_models,
    resolved_model_info,
    supported_thinking_levels,
)


def _spec(**overrides: object) -> ModelSpec:
    base = {
        "provider": "openai",
        "id": "gpt-test",
        "name": "GPT Test",
        "api": "openai-responses",
        "base_url": "https://api.openai.com/v1",
        "input": ("text",),
        "context_window": 1000,
        "max_tokens": 100,
        "reasoning": False,
    }
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]


def test_lane_catalogs_ship_every_vendored_model() -> None:
    """Verifies each default lane serves its vendored model catalog."""
    assert len(lane_catalog("openrouter")) == 396
    assert len(lane_catalog("opencode")) == 84
    assert len(lane_catalog("anthropic")) == 17
    assert len(lane_catalog("openai")) == 44
    assert len(lane_catalog("google")) == 23
    assert len(lane_catalog("groq")) == 7
    assert len(lane_catalog("xai")) == 4


def test_lane_catalog_ignores_unknown_lane() -> None:
    """Verifies a lane outside the catalog serves no models."""
    assert lane_catalog("deepseek") == {}
    assert load_lane_catalog("deepseek") == {}


def test_lane_catalog_models_carry_their_wire_protocol() -> None:
    """Verifies vendored models keep their own protocol and endpoint."""
    groq = lane_catalog("groq")
    assert groq["llama-3.1-8b-instant"].api == "openai-completions"
    assert groq["llama-3.1-8b-instant"].base_url == "https://api.groq.com/openai/v1"
    anthropic = lane_catalog("anthropic")
    assert anthropic["claude-haiku-4-5"].api == "anthropic-messages"
    google = lane_catalog("google")
    assert all(spec.api == "google-generative-ai" for spec in google.values())


def test_provider_and_model_metadata() -> None:
    """Verifies provider and model metadata reach the harness vocabulary."""
    assert provider_info("openai") == {"id": "openai", "name": "openai"}
    assert provider_info("openai", "Open AI") == {"id": "openai", "name": "Open AI"}
    info = model_info("openai", _spec(input=("text", "image")))
    assert info == {
        "provider": "openai",
        "id": "gpt-test",
        "name": "GPT Test",
        "inputModalities": ["text", "image"],
    }


def test_supported_thinking_levels_follow_the_model_map() -> None:
    """Verifies reasoning levels come from the model thinking-level map."""
    plain = _spec()
    assert supported_thinking_levels(plain) == ["off"]
    mapped = _spec(
        reasoning=True,
        thinking_level_map={
            "off": None,
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": None,
            "max": None,
        },
    )
    assert supported_thinking_levels(mapped) == ["minimal", "low", "medium", "high"]
    open_ended = _spec(reasoning=True, thinking_level_map={"xhigh": "xhigh"})
    assert supported_thinking_levels(open_ended) == [
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    ]


def test_reasoning_effort_resolution_rejects_unsupported_levels() -> None:
    """Verifies an unsupported reasoning effort fails loudly."""
    spec = _spec(reasoning=True, thinking_level_map={"high": "high"})
    assert resolve_reasoning_effort(spec, "high") == "high"
    assert resolve_reasoning_effort(spec, None) is None
    with pytest.raises(LlmError, match="does not support reasoning effort"):
        resolve_reasoning_effort(spec, "xhigh")


def test_resolved_model_info_reports_capability_and_default() -> None:
    """Verifies exact-route metadata carries context, modalities, reasoning."""
    spec = _spec(reasoning=True, thinking_level_map={"high": "high"})
    info = resolved_model_info("openai", spec, "high")
    assert info["provider"] == "openai"
    assert info["id"] == "gpt-test"
    assert info["context"] == {"contextWindow": 1000}
    assert info["inputModalities"] == ["text"]
    assert info["reasoning"] is not None
    assert info["reasoning"]["efforts"][0]["id"] == "off"
    assert info["reasoning"]["defaultEffort"] == "high"
    plain = resolved_model_info("openai", _spec())
    assert "reasoning" not in plain


def test_resolve_route_models_serves_the_vendored_catalog() -> None:
    """Verifies a configured lane serves its vendored catalog unchanged."""
    models = resolve_route_models("groq", {})
    assert "llama-3.1-8b-instant" in models
    assert models["llama-3.1-8b-instant"].api == "openai-completions"


def test_resolve_route_models_applies_configured_entries() -> None:
    """Verifies configured model entries replace the served catalog."""
    models = resolve_route_models(
        "groq",
        {
            "models": [
                {
                    "id": "custom",
                    "api": "openai-completions",
                    "baseURL": "https://x/v1",
                },
            ]
        },
    )
    assert list(models) == ["custom"]
    assert models["custom"].base_url == "https://x/v1"
    assert models["custom"].context_window == 1_000_000
    assert models["custom"].max_tokens == 128_000
    assert models["custom"].input == ("text",)


def test_resolve_route_models_merges_declared_entries_onto_the_catalog() -> None:
    """Verifies a declared entry inherits the catalog capabilities."""
    models = resolve_route_models(
        "openai",
        {
            "models": [
                {
                    "id": "gpt-4",
                    "name": "Renamed",
                    "contextWindow": 4096,
                    "input": ["text", "image"],
                    "reasoningEfforts": False,
                }
            ]
        },
    )
    assert models["gpt-4"].name == "Renamed"
    assert models["gpt-4"].context_window == 4096
    assert models["gpt-4"].input == ("text", "image")
    assert models["gpt-4"].reasoning is False


def test_resolve_route_models_applies_overrides_to_catalog_models() -> None:
    """Verifies a model override reshapes one catalog model."""
    models = resolve_route_models(
        "openai",
        {"modelOverrides": {"gpt-4": {"contextWindow": 1234}}},
    )
    assert models["gpt-4"].context_window == 1234
    assert "gpt-4o" in models


def test_resolve_route_models_repoints_a_whole_route() -> None:
    """Verifies route-level api and baseURL repoint every served model."""
    models = resolve_route_models(
        "groq",
        {"api": "openai-responses", "baseURL": "https://gateway.example/v1"},
    )
    assert all(spec.api == "openai-responses" for spec in models.values())
    assert all(
        spec.base_url == "https://gateway.example/v1" for spec in models.values()
    )


def test_resolve_route_models_refuses_an_unserviceable_route() -> None:
    """Verifies a route that cannot serve fails loudly."""
    with pytest.raises(LlmError, match="resolves no models"):
        resolve_route_models("deepseek", {})
    with pytest.raises(LlmError, match='lists model "a" more than once'):
        resolve_route_models(
            "deepseek",
            {
                "api": "openai-completions",
                "baseURL": "https://api.deepseek.com",
                "models": [
                    {"id": "a"},
                    {"id": "a"},
                ],
            },
        )
    with pytest.raises(LlmError, match="has a malformed model entry"):
        resolve_route_models(
            "deepseek",
            {
                "api": "openai-completions",
                "baseURL": "https://api.deepseek.com",
                "models": ["nope"],
            },
        )
    with pytest.raises(LlmError, match="needs an api"):
        resolve_route_models(
            "deepseek",
            {"baseURL": "https://api.deepseek.com", "models": [{"id": "a"}]},
        )
    with pytest.raises(LlmError, match="needs a baseURL"):
        resolve_route_models(
            "deepseek",
            {"api": "openai-completions", "models": [{"id": "a"}]},
        )
    with pytest.raises(LlmError, match="unsupported api"):
        resolve_route_models(
            "deepseek",
            {
                "api": "bedrock-converse-stream",
                "baseURL": "https://api.deepseek.com",
                "models": [{"id": "a"}],
            },
        )
    with pytest.raises(LlmError, match="modelOverrides"):
        resolve_route_models("deepseek", {"modelOverrides": {"a": {}}})
    with pytest.raises(LlmError, match="the catalog does not describe it"):
        resolve_route_models("openai", {"modelOverrides": {"nope": {}}})
    with pytest.raises(LlmError, match="malformed modelOverride"):
        resolve_route_models("openai", {"modelOverrides": {"gpt-4": "nope"}})
    with pytest.raises(LlmError, match="empty id"):
        resolve_route_models(
            "deepseek",
            {
                "api": "openai-completions",
                "baseURL": "https://api.deepseek.com",
                "models": [{"id": ""}],
            },
        )


def test_resolve_route_models_merges_compat_and_default_input() -> None:
    """Verifies configured compat and default input reach the model."""
    models = resolve_route_models(
        "deepseek",
        {
            "api": "openai-completions",
            "baseURL": "https://api.deepseek.com",
            "defaultInput": ["text", "image"],
            "models": [{"id": "a", "compat": {"supportsStore": True}}],
        },
    )
    assert models["a"].input == ("text", "image")
    assert models["a"].compat == {"supportsStore": True}


def test_model_info_omits_modalities_when_empty() -> None:
    """Verifies model listing metadata omits empty modality lists."""
    info = model_info("openai", _spec(input=()))
    assert info == {
        "provider": "openai",
        "id": "gpt-test",
        "name": "GPT Test",
    }


def test_reasoning_info_ignores_unknown_default_effort() -> None:
    """Verifies an unknown default effort is not reported."""
    spec = _spec(
        reasoning=True,
        thinking_level_map={"off": None, "high": "high"},
    )
    info = resolved_model_info("openai", spec, "bogus")
    assert info["reasoning"] is not None
    assert "defaultEffort" not in info["reasoning"]


def test_merge_model_accepts_reasoning_efforts_map() -> None:
    """Verifies a reasoning-efforts map reshapes a declared model."""
    spec = _merge_model(
        "deepseek",
        None,
        {
            "id": "a",
            "api": "openai-completions",
            "baseURL": "https://api.deepseek.com",
            "reasoningEfforts": {"off": None, "high": "high"},
        },
        route_api=None,
        route_base_url=None,
        default_input=DEFAULT_INPUT,
    )
    assert spec.reasoning is True
    assert spec.thinking_level_map == {"off": None, "high": "high"}


def test_load_lane_catalog_skips_unsupported_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a lane model with no usable protocol or endpoint is skipped."""

    def fake_models(lane: str) -> Mapping[str, Mapping[str, Any]]:
        return {
            "good": {
                "id": "good",
                "name": "Good",
                "api": "openai-completions",
                "baseUrl": "https://x/v1",
                "contextWindow": 1000,
                "maxTokens": 100,
                "input": ["text"],
            },
            "bad-api": {
                "id": "bad-api",
                "name": "Bad API",
                "api": "bedrock-converse-stream",
                "baseUrl": "https://x/v1",
            },
            "no-base": {
                "id": "no-base",
                "name": "No Base",
                "api": "openai-completions",
            },
        }

    monkeypatch.setattr("magitech_harness.llm.pi_ai.catalog._lane_models", fake_models)
    specs = load_lane_catalog("openai")
    assert set(specs) == {"good"}
