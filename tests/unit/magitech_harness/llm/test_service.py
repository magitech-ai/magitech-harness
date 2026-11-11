"""Unit tests for LLMService, MockLLMProvider, and NoProviderError."""

import pytest
from anima import Context, EventMode

from magitech_harness.llm.service import LLMService, MockLLMProvider, NoProviderError
from magitech_harness.llm.types import ModelResponse
from magitech_harness.session.events import ToolCallData


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
    service = LLMService(ctx, {"provider": initial_p})
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


@pytest.mark.asyncio
async def test_llm_service_stream_mock_provider() -> None:
    """Verifies streaming from active mock provider and event emission."""
    ctx = Context()
    service = LLMService(ctx)
    p = MockLLMProvider()
    tc = ToolCallData(call_id="call_1", name="bash", arguments={})
    p.queue_response(text="Streaming chunk 1", tool_calls=[tc])
    service.register_provider("mock", p)
    service.set_active_provider("mock")

    chunks = []
    emitted_chunks = []
    ctx.on("llm/chunk", lambda c: emitted_chunks.append(c))

    async for chunk in service.stream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].text == "Streaming chunk 1"
    assert len(chunks[1].tool_calls) == 1
    assert len(emitted_chunks) == 2


@pytest.mark.asyncio
async def test_llm_service_stream_no_provider_raises() -> None:
    """Verifies streaming raises NoProviderError if no provider is registered."""
    ctx = Context()
    service = LLMService(ctx)
    with pytest.raises(
        NoProviderError, match="No active LLM provider plugin registered"
    ):
        async for _ in service.stream(messages=[{"role": "user", "content": "hi"}]):
            pass


@pytest.mark.asyncio
async def test_llm_service_stream_with_aborted_signal() -> None:
    """Verifies streaming raises AbortError when AbortSignal is cancelled."""
    from magitech_harness.agent.cancellation import AbortError, AbortSignal

    ctx = Context()
    service = LLMService(ctx)
    p = MockLLMProvider()
    p.queue_response(text="test")
    service.register_provider("mock", p)
    service.set_active_provider("mock")

    signal = AbortSignal()
    signal.abort()

    with pytest.raises(AbortError, match="Operation was aborted"):
        async for _ in service.stream(
            messages=[{"role": "user", "content": "hi"}], signal=signal
        ):
            pass


@pytest.mark.asyncio
async def test_llm_service_stream_slug_routing() -> None:
    """Verifies streaming routes to specific provider by slug."""
    ctx = Context()
    service = LLMService(ctx)
    p = MockLLMProvider()
    p.queue_response(text="routed stream")
    service.register_provider("router", p)

    received = []
    async for chunk in service.stream(messages=[], model="router/some-model"):
        received.append(chunk)

    assert len(received) == 1
    assert received[0].text == "routed stream"


@pytest.mark.asyncio
async def test_mock_llm_provider_stream_direct_and_abort() -> None:
    """Verifies direct MockLLMProvider.stream calls, tool-only chunk, and abort signal."""
    from magitech_harness.agent.cancellation import AbortError, AbortSignal

    p = MockLLMProvider()
    tc = ToolCallData(call_id="c_direct", name="ls", arguments={})
    # Queue tool-only response (no text)
    p.queue_response(text="", tool_calls=[tc])

    chunks = []
    async for chunk in p.stream(messages=[]):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert len(chunks[0].tool_calls) == 1

    # Empty text and empty tool_calls branch
    p_empty = MockLLMProvider()
    p_empty.queue_response(text="", tool_calls=[])
    chunks_empty = []
    async for chunk in p_empty.stream(messages=[]):
        chunks_empty.append(chunk)
    assert len(chunks_empty) == 1
    assert chunks_empty[0].text == ""

    # Now test abort on direct stream
    signal = AbortSignal()
    signal.abort()
    with pytest.raises(AbortError, match="Operation was aborted"):
        async for _ in p.stream(messages=[], signal=signal):
            pass


@pytest.mark.asyncio
async def test_llm_service_stream_mid_stream_abort() -> None:
    """Verifies abort triggered mid-stream inside LLMService.stream loop."""
    from magitech_harness.agent.cancellation import AbortError, AbortSignal
    from magitech_harness.llm.types import ModelChunk

    class MultiChunkProvider:
        async def generate(self, *args, **kwargs):
            return ModelResponse(text="multi")

        async def stream(self, *args, **kwargs):
            yield ModelChunk(text="first")
            yield ModelChunk(text="second")

    ctx = Context()
    service = LLMService(ctx)
    service.register_provider("multi", MultiChunkProvider())
    service.set_active_provider("multi")

    signal = AbortSignal()

    with pytest.raises(AbortError, match="Operation was aborted"):
        async for chunk in service.stream(messages=[], signal=signal):
            if chunk.text == "first":
                signal.abort()
