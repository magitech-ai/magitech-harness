"""Adapter implementation for the pi-ai provider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, cast

import httpx

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.catalog import (
    ModelSpec,
    model_info,
    provider_info,
    resolve_reasoning_effort,
    resolved_model_info,
)
from magitech_harness.llm.pi_ai.config import ResolvedProfile
from magitech_harness.llm.pi_ai.protocols import STREAMERS
from magitech_harness.llm.service import LlmAdapter, PreparedAdapterCall
from magitech_harness.llm.types import (
    GenerateOptions,
    LlmModelInfo,
    LlmProviderInfo,
    LlmResolvedModelInfo,
    StreamChunk,
)

__all__ = ["PiAiAdapter"]


class PiAiAdapter(LlmAdapter):
    """Multi-provider adapter built on the four pi-ai wire protocols."""

    def __init__(
        self,
        profiles: Callable[[], Mapping[str, ResolvedProfile]],
        client_factory: Callable[[ResolvedProfile], httpx.AsyncClient] | None = None,
    ) -> None:
        """Bind the adapter to the profiles it serves.

        Args:
            profiles: Callable returning the currently resolved route profiles.
            client_factory: Optional factory producing the HTTP client used for
                one request; defaults to a client carrying the route timeout.
        """
        super().__init__()
        self._profiles = profiles
        self._client_factory = client_factory

    def _profile(self, provider: str) -> ResolvedProfile:
        profiles = self._profiles()
        profile = profiles.get(provider)
        if profile is None:
            raise LlmError(
                f'pi-ai adapter does not own provider "{provider}"', "NO_ADAPTER"
            )
        return profile

    def _spec(self, provider: str, model: str) -> ModelSpec:
        profile = self._profile(provider)
        spec = profile.models.get(model)
        if spec is None:
            raise LlmError(
                f'pi-ai provider "{provider}" has no configured model "{model}"',
                "UNKNOWN_MODEL",
            )
        return spec

    def provider_info(self, provider: str) -> LlmProviderInfo:
        profile = self._profiles().get(provider)
        display = profile.display_name if profile else provider
        return provider_info(provider, display)

    def provider_retry_policy(self, provider: str) -> dict[str, Any] | None:
        profile = self._profiles().get(provider)
        return cast(dict[str, Any], profile.retry_policy) if profile else None

    async def list_models(self, provider: str) -> Sequence[LlmModelInfo]:
        profile = self._profile(provider)
        try:
            profile.resolve_api_key()
        except LlmError as exc:
            if exc.code == "MISSING_CREDENTIAL":
                return []
            raise
        return [model_info(provider, spec) for spec in profile.models.values()]

    async def resolve_model(
        self,
        provider: str,
        model: str,
        signal: AbortSignal | None = None,
    ) -> LlmResolvedModelInfo:
        del signal
        profile = self._profile(provider)
        spec = self._spec(provider, model)
        return resolved_model_info(provider, spec, profile.reasoning)

    async def prepare_call(
        self,
        provider: str,
        model: str,
        signal: AbortSignal | None = None,
    ) -> PreparedAdapterCall:
        del signal
        profile = self._profile(provider)
        spec = self._spec(provider, model)
        resolved = resolved_model_info(provider, spec, profile.reasoning)

        async def _stream(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
            async for chunk in self._stream_call(provider, spec, profile, options):
                yield chunk

        return PreparedAdapterCall(model=resolved, stream=_stream)

    def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        provider = str(options.get("provider", ""))
        model = str(options.get("model", ""))
        profile = self._profile(provider)
        spec = self._spec(provider, model)

        async def _gen() -> AsyncIterator[StreamChunk]:
            async for chunk in self._stream_call(provider, spec, profile, options):
                yield chunk

        return _gen()

    def _client(self, profile: ResolvedProfile) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory(profile)
        timeout = httpx.Timeout(profile.timeout_ms / 1000)
        return httpx.AsyncClient(timeout=timeout)

    @staticmethod
    def _contains_image(options: GenerateOptions) -> bool:
        messages = options.get("messages") or []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "image":
                    return True
        return False

    async def _stream_call(
        self,
        provider: str,
        spec: ModelSpec,
        profile: ResolvedProfile,
        options: GenerateOptions,
    ) -> AsyncIterator[StreamChunk]:
        resolve_reasoning_effort(spec, cast(str | None, options.get("reasoningEffort")))
        api_key = profile.resolve_api_key()
        if self._contains_image(options) and "image" not in spec.input:
            raise LlmError(
                f'pi-ai model "{spec.id}" does not support image input',
                "UNSUPPORTED_CONTENT",
            )
        streamer = STREAMERS.get(spec.api)
        if streamer is None:
            raise LlmError(
                f'pi-ai provider "{provider}" uses unsupported api "{spec.api}"',
                "INVALID_CONFIG",
            )
        request_options = dict(options)
        if request_options.get("reasoningEffort") is None and profile.reasoning:
            request_options["reasoningEffort"] = profile.reasoning
        client = self._client(profile)
        try:
            async for chunk in streamer(
                spec,
                cast(GenerateOptions, request_options),
                api_key=api_key,
                headers=dict(profile.headers),
                client=client,
            ):
                yield chunk
        finally:
            await client.aclose()
