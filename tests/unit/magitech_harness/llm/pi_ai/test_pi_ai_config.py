"""Unit tests for the pi-ai provider configuration and profile resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.config import (
    DEFAULT_MAX_REQUEST_IMAGE_BYTES,
    DEFAULT_MAX_REQUEST_IMAGES,
    DEFAULT_STREAM_IDLE_TIMEOUT_MS,
    PiAiConfig,
    ProviderProfile,
    ResolvedProfile,
    parse_config,
    resolve_profiles,
)


def test_provider_profile_accepts_camel_case_and_snake_case() -> None:
    """Verifies a profile reads both the camelCase and snake_case spelling."""
    profile = ProviderProfile.model_validate(
        {
            "apiKeyEnv": "OPENAI_API_KEY",
            "displayName": "Open AI",
            "baseURL": "https://api.openai.com/v1",
            "defaultMaxTokens": 10,
            "defaultInput": ["text"],
        }
    )
    assert profile.api_key_env == "OPENAI_API_KEY"
    assert profile.display_name == "Open AI"
    assert profile.resolved_base_url() == "https://api.openai.com/v1"
    assert profile.default_max_tokens == 10
    assert profile.to_profile_mapping()["baseURL"] == "https://api.openai.com/v1"


def test_provider_profile_refuses_unknown_fields() -> None:
    """Verifies a mistyped profile field fails loudly."""
    with pytest.raises(ValidationError):
        ProviderProfile.model_validate({"apiKeyEnvironment": "OPENAI_API_KEY"})


def test_provider_profile_mapping_includes_optional_fields() -> None:
    """Verifies optional fields project into the catalog vocabulary."""
    profile = ProviderProfile.model_validate(
        {
            "apiKeyEnv": "OPENAI_API_KEY",
            "modelOverrides": {"gpt-4": {"contextWindow": 1234}},
            "compat": {"supportsStore": True},
            "defaultContextWindow": 4096,
        }
    )
    mapping = profile.to_profile_mapping()
    assert mapping["modelOverrides"] == {"gpt-4": {"contextWindow": 1234}}
    assert mapping["compat"] == {"supportsStore": True}
    assert mapping["defaultContextWindow"] == 4096


def test_parse_config_accepts_the_supported_shapes() -> None:
    """Verifies component configuration parses from every accepted shape."""
    assert parse_config(None).providers == {}
    assert parse_config({"providers": {}}).providers == {}
    parsed = PiAiConfig()
    assert parse_config(parsed) is parsed
    with pytest.raises(TypeError):
        parse_config(["providers"])


def test_resolve_profiles_serves_a_catalog_lane() -> None:
    """Verifies a configured lane resolves its profile and catalog."""
    profiles = resolve_profiles(
        {"providers": {"openai": {"apiKeyEnv": "OPENAI_API_KEY"}}}
    )
    profile = profiles["openai"]
    assert profile.display_name == "openai"
    assert profile.api_key_env == "OPENAI_API_KEY"
    assert "gpt-4o" in profile.models
    assert profile.retry_policy["mode"] == "normal"


def test_resolve_profiles_serves_a_declared_route() -> None:
    """Verifies a route outside the catalog serves when it declares models."""
    profiles = resolve_profiles(
        {
            "providers": {
                "deepseek": {
                    "apiKeyEnv": "DEEPSEEK_API_KEY",
                    "api": "openai-completions",
                    "baseURL": "https://api.deepseek.com",
                    "models": [{"id": "deepseek-chat"}],
                }
            }
        }
    )
    profile = profiles["deepseek"]
    assert list(profile.models) == ["deepseek-chat"]
    assert profile.models["deepseek-chat"].base_url == "https://api.deepseek.com"


def test_resolve_profiles_refuses_an_empty_route_name() -> None:
    """Verifies a route without a name fails loudly."""
    with pytest.raises(LlmError, match="provider names must be non-empty"):
        resolve_profiles({"providers": {"": {"apiKeyEnv": "X"}}})


def test_resolved_profile_defaults_every_knob() -> None:
    """Verifies the adapter-owned defaults land on a resolved profile."""
    profile = ProviderProfile()
    resolved = ResolvedProfile("openai", profile, {})
    assert resolved.display_name == "openai"
    assert resolved.stream_idle_timeout_ms == DEFAULT_STREAM_IDLE_TIMEOUT_MS
    assert resolved.max_request_image_bytes == DEFAULT_MAX_REQUEST_IMAGE_BYTES
    assert resolved.max_request_images == DEFAULT_MAX_REQUEST_IMAGES
    assert resolved.image_offload_count_quantum == 10
    assert resolved.request_image_pixel_budget == 2048 * 2048
    assert resolved.request_image_max_bytes == 1024 * 1024
    assert resolved.timeout_ms == 300_000
    assert resolved.headers == {}


def test_resolved_profile_honours_configured_knobs() -> None:
    """Verifies configured knobs win over the adapter-owned defaults."""
    profile = ProviderProfile.model_validate(
        {
            "maxRequestImages": 5,
            "imageOffloadCountQuantum": 2,
            "streamIdleTimeoutMs": 1000,
            "timeoutMs": 2000,
            "headers": {"x-test": "1"},
        }
    )
    resolved = ResolvedProfile("openai", profile, {})
    assert resolved.max_request_images == 5
    assert resolved.image_offload_count_quantum == 2
    assert resolved.stream_idle_timeout_ms == 1000
    assert resolved.timeout_ms == 2000
    assert resolved.headers == {"x-test": "1"}


def test_resolve_api_key_reads_the_configured_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the credential is resolved from the environment per request."""
    profile = ProviderProfile.model_validate({"apiKeyEnv": "PI_AI_TEST_KEY"})
    resolved = ResolvedProfile("openai", profile, {})
    monkeypatch.setenv("PI_AI_TEST_KEY", "  sk-secret  ")
    assert resolved.resolve_api_key() == "sk-secret"


def test_resolve_api_key_reports_a_missing_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a named but unset credential fails loudly."""
    profile = ProviderProfile.model_validate({"apiKeyEnv": "PI_AI_TEST_KEY"})
    resolved = ResolvedProfile("openai", profile, {})
    monkeypatch.delenv("PI_AI_TEST_KEY", raising=False)
    with pytest.raises(LlmError, match="no credential for provider route") as excinfo:
        resolved.resolve_api_key()
    assert excinfo.value.code == "MISSING_CREDENTIAL"
    monkeypatch.setenv("PI_AI_TEST_KEY", "")
    with pytest.raises(LlmError) as excinfo:
        resolved.resolve_api_key()
    assert excinfo.value.code == "MISSING_CREDENTIAL"


def test_resolve_api_key_rejects_an_unusable_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a credential no HTTP header can carry fails loudly."""
    profile = ProviderProfile.model_validate({"apiKeyEnv": "PI_AI_TEST_KEY"})
    resolved = ResolvedProfile("openai", profile, {})
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk\nsecret")
    with pytest.raises(LlmError) as excinfo:
        resolved.resolve_api_key()
    assert excinfo.value.code == "INVALID_CREDENTIAL"


def test_resolve_api_key_without_a_credential_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a route naming no credential asks for none."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolved = ResolvedProfile("openai", ProviderProfile(), {})
    assert resolved.resolve_api_key() is None


def test_retry_policy_follows_the_harness_policy() -> None:
    """Verifies a configured retry policy resolves through the harness rules."""
    profile = ProviderProfile.model_validate(
        {"retryPolicy": {"mode": "normal", "maxRetries": 2}}
    )
    resolved = ResolvedProfile("openai", profile, {})
    assert resolved.retry_policy["mode"] == "normal"
    assert resolved.retry_policy["maxRetries"] == 2
    assert resolved.retry_policy["retryableCodes"]
