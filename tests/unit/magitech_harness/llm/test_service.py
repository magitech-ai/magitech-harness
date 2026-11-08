"""Unit tests for LLMService and MockLLMProvider."""

import pytest
from anima import Context, EventMode
from magitech_harness.llm.service import LLMService, MockLLMProvider
from magitech_harness.llm.types import ModelResponse


@pytest.mark.asyncio
async def test_mock_llm_provider_default_response() -> None:
    """Verifies MockLLMProvider returns default response when queue is exhausted."""
    provider = MockLLMProvider()
    resp = await provider.generate(messages=[{"role": "user", "content": "hi"}])
    assert resp.text == "Default mock response"
    assert resp.tool_calls == []
    assert len(provider.call_history) == 1


@pytest.mark.asyncio
async def test_llm_service_set_provider_and_call() -> None:
    """Verifies LLMService provider getter/setter and execution."""
    ctx = Context()
    service = LLMService(ctx)
    assert isinstance(service.provider, MockLLMProvider)

    custom_provider = MockLLMProvider()
    custom_provider.queue_response(text="custom reply")
    service.set_provider(custom_provider)
    assert service.provider is custom_provider

    res = await service.call(messages=[{"role": "user", "content": "test"}])
    assert res.text == "custom reply"


@pytest.mark.asyncio
async def test_llm_service_waterfall_interception() -> None:
    """Verifies that llm/call waterfall middleware can intercept and modify LLM output."""
    ctx = Context()
    ctx.plugin(LLMService)
    await ctx.flush()

    mock_llm: MockLLMProvider = ctx.llm.provider
    mock_llm.queue_response(text="raw response")

    async def rewrite_middleware(messages, tools, system_prompt, next_fn):
        original_res = await next_fn(messages, tools, system_prompt)
        return ModelResponse(text=f"[censored] {original_res.text}")

    ctx.on("llm/call", rewrite_middleware, mode=EventMode.WATERFALL)

    final_res = await ctx.llm.call(messages=[{"role": "user", "content": "hello"}])
    assert final_res.text == "[censored] raw response"
