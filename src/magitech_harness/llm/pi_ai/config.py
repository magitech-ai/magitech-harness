"""Configuration models and profile resolution for the pi-ai provider."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from magitech_harness.llm.adapter_failure import LlmError, assert_usable_api_key
from magitech_harness.llm.pi_ai.catalog import ModelSpec, resolve_route_models
from magitech_harness.llm.retry_policy import resolve_retry_policy

DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
DEFAULT_TIMEOUT_MS = 300_000
DEFAULT_MAX_REQUEST_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_REQUEST_IMAGES = 50
DEFAULT_IMAGE_OFFLOAD_COUNT_QUANTUM = 10
DEFAULT_REQUEST_IMAGE_PIXEL_BUDGET = 2048 * 2048
DEFAULT_REQUEST_IMAGE_MAX_BYTES = 1024 * 1024


class ProviderProfile(BaseModel):
    """User-configured route profile for one provider."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    api_key_env: str | None = Field(default=None, alias="apiKeyEnv")
    display_name: str | None = Field(default=None, alias="displayName")
    api: str | None = None
    base_url: str | None = Field(default=None, alias="baseURL")
    models: list[dict[str, Any]] | None = None
    model_overrides: dict[str, dict[str, Any]] | None = Field(
        default=None, alias="modelOverrides"
    )
    compat: dict[str, Any] | None = None
    default_input: list[str] | None = Field(default=None, alias="defaultInput")
    default_context_window: int | None = Field(
        default=None, alias="defaultContextWindow"
    )
    default_max_tokens: int | None = Field(default=None, alias="defaultMaxTokens")
    headers: dict[str, str] | None = None
    reasoning: str | None = None
    retry_policy: dict[str, Any] | None = Field(default=None, alias="retryPolicy")
    stream_idle_timeout_ms: int | None = Field(
        default=None, alias="streamIdleTimeoutMs"
    )
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    max_request_image_bytes: int | None = Field(
        default=None, alias="maxRequestImageBytes"
    )
    max_request_images: int | None = Field(default=None, alias="maxRequestImages")
    image_offload_count_quantum: int | None = Field(
        default=None, alias="imageOffloadCountQuantum"
    )
    request_image_pixel_budget: int | None = Field(
        default=None, alias="requestImagePixelBudget"
    )
    request_image_max_bytes: int | None = Field(
        default=None, alias="requestImageMaxBytes"
    )

    def resolved_base_url(self) -> str | None:
        """Return the configured endpoint, if any."""
        return self.base_url

    def to_profile_mapping(self) -> dict[str, Any]:
        """Project the profile into the catalog resolution vocabulary."""
        data: dict[str, Any] = {}
        if self.api is not None:
            data["api"] = self.api
        if self.base_url is not None:
            data["baseURL"] = self.base_url
        if self.models is not None:
            data["models"] = self.models
        if self.model_overrides is not None:
            data["modelOverrides"] = self.model_overrides
        if self.compat is not None:
            data["compat"] = self.compat
        if self.default_input is not None:
            data["defaultInput"] = self.default_input
        if self.default_context_window is not None:
            data["defaultContextWindow"] = self.default_context_window
        if self.default_max_tokens is not None:
            data["defaultMaxTokens"] = self.default_max_tokens
        return data


class PiAiConfig(BaseModel):
    """Top-level pi-ai provider component configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    providers: dict[str, ProviderProfile] = Field(default_factory=dict)


class ResolvedProfile:
    """Resolved runtime profile for one provider route."""

    def __init__(
        self,
        provider: str,
        profile: ProviderProfile,
        models: dict[str, ModelSpec],
    ) -> None:
        """Bind a validated profile to its resolved model catalog."""
        self.provider = provider
        self.profile = profile
        self.models = models
        self.display_name = profile.display_name or provider
        self.api_key_env = profile.api_key_env
        self.headers = dict(profile.headers or {})
        self.reasoning = profile.reasoning
        self.retry_policy = resolve_retry_policy(
            profile.retry_policy, f'pi-ai provider "{provider}" retryPolicy'
        )
        self.timeout_ms = (
            profile.timeout_ms if profile.timeout_ms is not None else DEFAULT_TIMEOUT_MS
        )
        self.stream_idle_timeout_ms = (
            profile.stream_idle_timeout_ms
            if profile.stream_idle_timeout_ms is not None
            else DEFAULT_STREAM_IDLE_TIMEOUT_MS
        )
        self.max_request_image_bytes = (
            profile.max_request_image_bytes
            if profile.max_request_image_bytes is not None
            else DEFAULT_MAX_REQUEST_IMAGE_BYTES
        )
        self.max_request_images = (
            profile.max_request_images
            if profile.max_request_images is not None
            else DEFAULT_MAX_REQUEST_IMAGES
        )
        self.image_offload_count_quantum = (
            profile.image_offload_count_quantum
            if profile.image_offload_count_quantum is not None
            else min(DEFAULT_IMAGE_OFFLOAD_COUNT_QUANTUM, self.max_request_images)
        )
        self.request_image_pixel_budget = (
            profile.request_image_pixel_budget
            if profile.request_image_pixel_budget is not None
            else DEFAULT_REQUEST_IMAGE_PIXEL_BUDGET
        )
        self.request_image_max_bytes = (
            profile.request_image_max_bytes
            if profile.request_image_max_bytes is not None
            else DEFAULT_REQUEST_IMAGE_MAX_BYTES
        )

    def resolve_api_key(self) -> str | None:
        """Resolve and validate the configured credential for one request."""
        env_name = self.api_key_env
        if env_name is None:
            return None
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            raise LlmError(
                f'pi-ai: no credential for provider route "{self.provider}"; set '
                f"{env_name} in the environment or remove apiKeyEnv",
                "MISSING_CREDENTIAL",
            )
        return assert_usable_api_key(raw, "pi-ai", env_name)


def parse_config(config: Any) -> PiAiConfig:
    """Parse component configuration into a PiAiConfig."""
    if config is None:
        return PiAiConfig()
    if isinstance(config, PiAiConfig):
        return config
    if isinstance(config, Mapping):
        return PiAiConfig.model_validate(dict(config))
    raise TypeError("pi-ai provider configuration must be a mapping")


def resolve_profiles(config: Any) -> dict[str, ResolvedProfile]:
    """Resolve provider profiles from component configuration."""
    parsed = parse_config(config)
    resolved: dict[str, ResolvedProfile] = {}
    for provider, profile in parsed.providers.items():
        if not provider:
            raise LlmError("pi-ai: provider names must be non-empty", "INVALID_CONFIG")
        models = resolve_route_models(provider, profile.to_profile_mapping())
        resolved[provider] = ResolvedProfile(provider, profile, models)
    return resolved
