"""
LLMService (ctx.llm): Pluggable model providers and streaming waterfall seam.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol
from kernel import Context, Service
from harness.llm.types import ModelResponse
from harness.session.events import ToolCallData


class LLMProvider(Protocol):
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        ...


class MockLLMProvider:
    """Deterministic scriptable mock provider for offline tests and development."""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self.responses: list[ModelResponse] = responses or []
        self.call_history: list[dict[str, Any]] = []

    def queue_response(self, text: str = "", tool_calls: list[ToolCallData] | None = None) -> None:
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
    ) -> ModelResponse:
        self.call_history.append({
            "messages": messages,
            "tools": tools,
            "system_prompt": system_prompt,
        })
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse(text="Default mock response", tool_calls=[])


class LLMService(Service):
    """Service managing LLM execution and waterfall streaming interception."""

    def __init__(self, ctx: Context, provider: LLMProvider | None = None) -> None:
        super().__init__(ctx, "llm")
        self._provider = provider or MockLLMProvider()

    def set_provider(self, provider: LLMProvider) -> None:
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Execute model call through llm/stream waterfall pipeline."""
        async def _call_provider(msgs, tls, sys_p):
            return await self._provider.generate(
                messages=msgs, tools=tls, system_prompt=sys_p
            )

        # Allow waterfall listeners to route/intercept/log LLM requests
        response: ModelResponse = await self.ctx.waterfall(
            "llm/call",
            messages,
            tools,
            system_prompt,
            default_handler=_call_provider,
        )
        return response
