"""Unit tests for LLMService, MockLLMProvider, and NoProviderError."""

import pytest
from anima import Context, EventMode

from magitech_harness.llm.service import LLMService, MockLLMProvider, NoProviderError
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
async def test_llm_service_no_provider_raises_error() -> None:
    """Verifies that calling LLMService with no provider raises NoProviderError."""
    ctx = Context()
    service = LLMService(ctx)
    with pytest.raises(
        NoProviderError, match="No active LLM provider plugin registered"
    ):
        await service.call(messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_llm_service_register_named_provider_and_routing() -> None:
    """Verifies registering named providers and routing calls via model slugs."""
    ctx = Context()
    service = LLMService(ctx)

    p_openrouter = MockLLMProvider()
    p_openrouter.queue_response(text="from openrouter")
    disposer = service.register_provider("openrouter", p_openrouter)

    p_anthropic = MockLLMProvider()
    p_anthropic.queue_response(text="from anthropic")
    service.register_provider("anthropic", p_anthropic)

    assert service.get_provider("openrouter") is p_openrouter
    assert service.get_provider("anthropic") is p_anthropic
    assert "openrouter" in service.providers

    # Route by slug prefix
    res1 = await service.call(
        messages=[{"role": "user", "content": "test"}],
        model="openrouter/meta-llama/llama-3-70b",
    )
    assert res1.text == "from openrouter"

    # Set active provider and call without prefix
    service.set_active_provider("anthropic")
    assert service.active_provider is p_anthropic

    res2 = await service.call(
        messages=[{"role": "user", "content": "test"}],
        model="claude-3-5-sonnet",
    )
    assert res2.text == "from anthropic"

    # Test unregister via disposer
    disposer()
    assert service.get_provider("openrouter") is None
    # Idempotent disposer call
    disposer()


@pytest.mark.asyncio
async def test_llm_service_waterfall_interception() -> None:
    """Verifies that llm/call waterfall middleware can intercept and modify LLM output."""
    ctx = Context()
    ctx.plugin(LLMService)
    await ctx.flush()

    mock_llm = MockLLMProvider()
    mock_llm.queue_response(text="raw response")
    ctx.llm.register_provider("mock", mock_llm)
    ctx.llm.set_active_provider("mock")

    async def rewrite_middleware(messages, tools, system_prompt, next_fn):
        original_res = await next_fn(messages, tools, system_prompt)
        return ModelResponse(text=f"[censored] {original_res.text}")

    ctx.on("llm/call", rewrite_middleware, mode=EventMode.WATERFALL)

    final_res = await ctx.llm.call(messages=[{"role": "user", "content": "hello"}])
    assert final_res.text == "[censored] raw response"


@pytest.mark.asyncio
async def test_llm_service_initial_provider_and_active_fallback() -> None:
    """Verifies passing provider to LLMService constructor and active fallback on unregister."""
    ctx = Context()
    initial_p = MockLLMProvider()
    service = LLMService(ctx, provider=initial_p)
    assert service.get_provider("default") is initial_p
    assert service.active_provider is initial_p

    # Register second provider
    second_p = MockLLMProvider()
    d2 = service.register_provider("second", second_p)

    # Active is default. Unregister default -> active should fall back to 'second'
    service.get_provider("default")
    del service._providers["default"]
    # Trigger unregister when active is being removed and other providers exist (line 122)
    service.set_active_provider("second")
    service.register_provider("third", MockLLMProvider())
    d2()  # unregister 'second' which was active -> active becomes 'third'
    assert service.active_provider is not None


def test_llm_service_set_unknown_active_provider_raises() -> None:
    """Verifies setting an unregistered provider as active raises KeyError."""
    ctx = Context()
    service = LLMService(ctx)
    with pytest.raises(KeyError, match="not registered"):
        service.set_active_provider("nonexistent")
