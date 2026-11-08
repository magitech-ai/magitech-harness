"""LLMService (ctx.llm): Pluggable model providers and streaming waterfall seam.

This module provides the central LLM orchestration service for Magitech Harness.
It decouples the runtime engine from specific model vendors by exposing a
pluggable provider registry and waterfall hooks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from anima import Context, Service
from anima.events import Disposer

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.slug import parse_model_slug
from magitech_harness.llm.types import ModelChunk, ModelResponse
from magitech_harness.session.events import ToolCallData


class NoProviderError(RuntimeError):
    """Raised when an LLM call is attempted without an active provider plugin."""


class LLMProvider(Protocol):
    """Protocol that any model provider plugin must implement."""

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generates a response from the underlying model.

        Args:
            messages: List of model-visible conversation messages.
            tools: Optional list of tool JSON schemas available to the model.
            system_prompt: Optional assembled system prompt string.
            **kwargs: Extra vendor-specific generation options.

        Returns:
            A ModelResponse instance containing generated text and/or tool calls.
        """
        ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        signal: AbortSignal | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunk]:
        """Streams response chunks from the underlying model."""
        ...


class MockLLMProvider:
    """Deterministic scriptable mock provider for offline tests and development."""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        """Initializes MockLLMProvider with an optional pre-queued list of responses."""
        self.responses: list[ModelResponse] = responses or []
        self.call_history: list[dict[str, Any]] = []

    def queue_response(
        self, text: str = "", tool_calls: list[ToolCallData] | None = None
    ) -> None:
        """Queues a mock response to be yielded by generate."""
        self.responses.append(
            ModelResponse(
                text=text,
                tool_calls=tool_calls or [],
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Yields the next queued response or a fallback default response."""
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse(text="Default mock response", tool_calls=[])

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        signal: AbortSignal | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunk]:
        """Yields mock response chunks as a stream."""
        resp = await self.generate(messages, tools, system_prompt, **kwargs)
        if signal and signal.is_aborted:
            raise AbortError("Operation was aborted.")
        if resp.text:
            yield ModelChunk(
                text=resp.text,
                tool_calls=[],
                finish_reason=None if resp.tool_calls else resp.finish_reason,
            )
        if resp.tool_calls:
            yield ModelChunk(
                text="",
                tool_calls=resp.tool_calls,
                finish_reason=resp.finish_reason,
            )


class LLMService(Service):
    """Service managing LLM execution, provider registries, and waterfall hooks."""

    def __init__(self, ctx: Context, provider: LLMProvider | None = None) -> None:
        """Initializes LLMService on the given context."""
        super().__init__(ctx, "llm")
        self._providers: dict[str, LLMProvider] = {}
        self._active_provider_name: str | None = None

        if provider is not None:
            self.register_provider("default", provider)

    def register_provider(self, name: str, provider: LLMProvider) -> Disposer:
        """Registers a named model provider plugin.

        Args:
            name: Unique slug identifier for the provider (e.g. 'openrouter').
            provider: The provider instance implementing LLMProvider protocol.

        Returns:
            A disposer callable that unregisters the provider upon disposal.
        """
        self._providers[name] = provider
        if self._active_provider_name is None:
            self._active_provider_name = name

        def _unregister() -> None:
            if name in self._providers:
                del self._providers[name]
                if self._active_provider_name == name:
                    self._active_provider_name = (
                        next(iter(self._providers.keys())) if self._providers else None
                    )

        self.ctx.effect(lambda: _unregister)
        return _unregister

    def get_provider(self, name: str) -> LLMProvider | None:
        """Retrieves a registered provider by name."""
        return self._providers.get(name)

    def set_active_provider(self, name: str) -> None:
        """Sets the active provider by name.

        Args:
            name: The registered provider name to activate.

        Raises:
            KeyError: If name is not a registered provider.
        """
        if name not in self._providers:
            raise KeyError(
                f"Provider '{name}' is not registered. "
                f"Available providers: {list(self._providers.keys())}"
            )
        self._active_provider_name = name

    @property
    def active_provider(self) -> LLMProvider | None:
        """Returns the currently active LLMProvider instance or None."""
        if self._active_provider_name is not None:
            return self._providers.get(self._active_provider_name)
        return None

    @property
    def providers(self) -> dict[str, LLMProvider]:
        """Returns a mapping of all registered providers."""
        return dict(self._providers)

    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Executes a model call through the llm/call waterfall pipeline.

        Args:
            messages: Conversation messages.
            tools: Optional tool definitions.
            system_prompt: Optional system prompt.
            model: Optional model slug (e.g. 'openrouter/meta-llama/llama-3-70b').
            **kwargs: Extra options forwarded to the provider.

        Returns:
            A ModelResponse object.

        Raises:
            NoProviderError: If no provider is available or registered.
        """
        provider_name, _ = parse_model_slug(model or "")
        target_provider: LLMProvider | None = None

        if provider_name and provider_name in self._providers:
            target_provider = self._providers[provider_name]
        elif self.active_provider is not None:
            target_provider = self.active_provider

        if target_provider is None:
            raise NoProviderError(
                "No active LLM provider plugin registered. Please install or configure "
                "a provider plugin (e.g. 'mth plugin install openrouter') or specify one "
                "in your profile.yaml."
            )

        async def _call_provider(
            msgs: list[dict[str, Any]],
            tls: list[dict[str, Any]] | None,
            sys_p: str | None,
        ) -> ModelResponse:
            return await target_provider.generate(
                messages=msgs, tools=tls, system_prompt=sys_p, model=model, **kwargs
            )

        response: ModelResponse = await self.ctx.waterfall(
            "llm/call",
            messages,
            tools,
            system_prompt,
            default_handler=_call_provider,
        )
        return response

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        signal: AbortSignal | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunk]:
        """Streams model output chunks from the resolved provider."""
        provider_name, _ = parse_model_slug(model or "")
        target_provider: LLMProvider | None = None

        if provider_name and provider_name in self._providers:
            target_provider = self._providers[provider_name]
        elif self.active_provider is not None:
            target_provider = self.active_provider

        if target_provider is None:
            raise NoProviderError(
                "No active LLM provider plugin registered. Please install or configure "
                "a provider plugin (e.g. 'mth plugin install openrouter') or specify one "
                "in your profile.yaml."
            )

        if signal:
            signal.throw_if_aborted()

        self.ctx.emit("llm/stream-start", {"messages": messages, "model": model})

        async for chunk in target_provider.stream(
            messages=messages,
            tools=tools,
            system_prompt=system_prompt,
            model=model,
            signal=signal,
            **kwargs,
        ):
            if signal:
                signal.throw_if_aborted()
            self.ctx.emit("llm/chunk", chunk)
            yield chunk

        self.ctx.emit("llm/stream-end", {"model": model})
