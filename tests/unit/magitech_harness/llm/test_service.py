"""Unit tests for LLMService, MockLLMProvider, and NoProviderError."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from anima import Context, EventMode

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm import (
    AdapterRegistrationHandle,
    DirectoryRegistrationHandle,
    LlmAdapter,
    LlmError,
    LlmRuntime,
    PreparedAdapterCall,
    PreparedLlmCall,
    RemoteError,
    create_assistant_message,
    create_developer_message,
    create_user_message,
    deep_freeze,
    resolve_retry_policy,
)
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
        NoProviderError, match="No active LLM provider component registered"
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

    p_secondary = MockLLMProvider()
    p_secondary.queue_response(text="from secondary")
    service.register_provider("secondary", p_secondary)

    assert service.get_provider("openrouter") is p_openrouter
    assert service.get_provider("secondary") is p_secondary
    assert "openrouter" in service.providers

    # Route by slug prefix
    res1 = await service.call(
        messages=[{"role": "user", "content": "test"}],
        model="openrouter/meta-llama/llama-3-70b",
    )
    assert res1.text == "from openrouter"

    # Set active provider and call without prefix
    service.set_active_provider("secondary")
    assert service.active_provider is p_secondary

    res2 = await service.call(
        messages=[{"role": "user", "content": "test"}],
        model="model-beta",
    )
    assert res2.text == "from secondary"

    # Test unregister via disposer
    disposer()
    assert service.get_provider("openrouter") is None
    # Idempotent disposer call
    disposer()


@pytest.mark.asyncio
async def test_llm_service_waterfall_interception() -> None:
    """Verifies that llm/call waterfall middleware can intercept and modify LLM output."""
    ctx = Context()
    ctx.use(LLMService)
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
        NoProviderError, match="No active LLM provider component registered"
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


# ---------------------------------------------------------------------------
# Parity LLMService / LlmRuntime tests
# ---------------------------------------------------------------------------


class _ScriptedAdapter(LlmAdapter):
    """Minimal scripted adapter yielding fixed stream chunks."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = script

    async def stream(
        self, options: Any, signal: AbortSignal | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        del options, signal
        for chunk in self._script:
            yield chunk


class _RecordingAdapter(_ScriptedAdapter):
    """Scripted adapter recording the most recent dispatched GenerateOptions."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        super().__init__(script)
        self.last_options: dict[str, Any] | None = None

    async def stream(
        self, options: Any, signal: AbortSignal | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        self.last_options = dict(options)
        async for chunk in super().stream(options, signal):
            yield chunk


class _CatalogAdapter(_ScriptedAdapter):
    """Adapter serving configurable provider metadata, models, retry, & capabilities."""

    def __init__(
        self,
        script: list[dict[str, Any]],
        *,
        provider_info_map: dict[str, dict[str, Any]] | None = None,
        models: list[dict[str, Any]] | None = None,
        resolved: dict[str, dict[str, Any]] | None = None,
        retry: dict[str, Any] | None = None,
        pricing: dict[str, Any] | None = None,
        image_target: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(script)
        self._provider_info_map = provider_info_map or {}
        self._models = models or []
        self._resolved = resolved or {}
        self._retry = retry
        self._pricing = pricing
        self._image_target = image_target
        self.resolve_calls = 0

    def provider_info(self, provider: str) -> dict[str, Any]:
        return self._provider_info_map.get(provider, {"id": provider, "name": provider})

    def provider_retry_policy(self, provider: str) -> dict[str, Any] | None:
        del provider
        return self._retry

    def image_request_pricing(self, provider: str, model: str) -> dict[str, Any] | None:
        del provider, model
        return self._pricing

    def image_request_target(
        self, provider: str, model: str, image: dict[str, Any]
    ) -> dict[str, Any]:
        if self._image_target is not None:
            return self._image_target
        return super().image_request_target(provider, model, image)

    async def list_models(self, provider: str) -> list[dict[str, Any]]:
        del provider
        return self._models

    async def resolve_model(
        self, provider: str, model: str, signal: AbortSignal | None = None
    ) -> dict[str, Any]:
        del signal
        self.resolve_calls += 1
        return self._resolved.get(
            model,
            {"provider": provider, "id": model, "name": model},
        )


def _setup_runtime() -> tuple[Context, LLMService]:
    ctx = Context()
    llm = LLMService(ctx)
    return ctx, llm


def _sample_script() -> list[dict[str, Any]]:
    return [
        {"type": "text-delta", "text": "hello"},
        {
            "type": "finish",
            "reason": {"kind": "stop"},
            "usage": {"inputTokens": 3, "outputTokens": 2},
        },
    ]


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "provider": "openai",
        "displayName": "OpenAI",
        "settingsNs": "llm-provider",
        "settingsPath": ["providers", "openai"],
    }
    base.update(overrides)
    return base


def test_llm_adapter_base_defaults() -> None:
    """Base LlmAdapter exposes default provider info and raises on unimplemented hooks."""
    adapter = LlmAdapter()
    assert adapter.provider_info("p") == {"id": "p", "name": "p"}
    assert adapter.provider_retry_policy("p") is None
    assert adapter.image_request_pricing("p", "m") is None
    assert adapter.image_request_target(
        "p",
        "m",
        {
            "attachmentId": "att-1",
            "mediaType": "image/png",
            "bytes": 100,
            "width": 800,
            "height": 600,
        },
    ) == {"width": 800, "height": 600, "maxBytes": 100}
    with pytest.raises(NotImplementedError):
        adapter.stream({"provider": "p", "model": "m", "messages": []})


@pytest.mark.asyncio
async def test_llm_adapter_async_base_defaults() -> None:
    """Async defaults on LlmAdapter return empty catalog and basic resolved model info."""
    adapter = LlmAdapter()
    assert await adapter.list_models("p") == []
    assert await adapter.resolve_model("p", "m") == {
        "provider": "p",
        "id": "m",
        "name": "m",
    }
    prepared = await adapter.prepare_call("p", "m")
    assert isinstance(prepared, PreparedAdapterCall)
    assert prepared.model == {"provider": "p", "id": "m", "name": "m"}


def test_register_adapter_validation_disposal_and_replace() -> None:
    """Adapter registration enforces all-or-nothing rules, atomic replace, and fiber disposal."""
    ctx, llm = _setup_runtime()
    assert LlmRuntime is LLMService

    with pytest.raises(LlmError) as exc_info:
        llm.register_adapter(
            ["a"],
            SimpleNamespace(provider_info=lambda p: {"id": "wrong", "name": "Bad"}),
        )
    assert exc_info.value.code == "INVALID_ADAPTER"

    with pytest.raises(LlmError) as exc_bad_name:
        llm.register_adapter(
            ["a"],
            SimpleNamespace(provider_info=lambda p: {"id": "a", "name": ""}),
        )
    assert exc_bad_name.value.code == "INVALID_ADAPTER"

    with pytest.raises(LlmError) as exc_empty:
        llm.register_adapter([], _ScriptedAdapter([]))
    assert exc_empty.value.code == "INVALID_ADAPTER"

    with pytest.raises(LlmError) as exc_blank:
        llm.register_adapter(["valid", ""], _ScriptedAdapter([]))
    assert exc_blank.value.code == "INVALID_ADAPTER"
    assert llm.list_providers() == []

    with pytest.raises(LlmError) as exc_dup_in_call:
        llm.register_adapter(["a", "a"], _ScriptedAdapter([]))
    assert exc_dup_in_call.value.code == "DUPLICATE_ADAPTER"
    assert llm.list_providers() == []

    events: list[list[str]] = []
    ctx.on(
        "llm/adapters-updated",
        lambda: events.append([p["id"] for p in llm.list_providers()]),
    )

    handle = llm.register_adapter(["a", "b"], _ScriptedAdapter([]))
    assert isinstance(handle, AdapterRegistrationHandle)
    assert [p["id"] for p in llm.list_providers()] == ["a", "b"]

    other = llm.register_adapter(["owned"], _ScriptedAdapter([]))
    with pytest.raises(LlmError) as exc_collide:
        llm.register_adapter(["fresh", "owned"], _ScriptedAdapter([]))
    assert exc_collide.value.code == "DUPLICATE_ADAPTER"
    assert [p["id"] for p in llm.list_providers()] == ["a", "b", "owned"]

    with pytest.raises(LlmError) as exc_swap_collide:
        handle.replace(["owned"])
    assert exc_swap_collide.value.code == "DUPLICATE_ADAPTER"
    assert [p["id"] for p in llm.list_providers()] == ["a", "b", "owned"]

    events.clear()
    handle.replace(["a"])
    assert events == [["owned", "a"]]

    handle.replace([])
    assert [p["id"] for p in llm.list_providers()] == ["owned"]

    handle()
    handle()  # idempotent
    with pytest.raises(LlmError) as exc_disposed:
        handle.replace(["x"])
    assert exc_disposed.value.code == "REGISTRATION_DISPOSED"

    other()
    assert llm.list_providers() == []


@pytest.mark.asyncio
async def test_adapters_updated_containment_and_fiber_disposal() -> None:
    """Sync and async listener errors warn without breaking registration; fiber teardown unregisters."""
    ctx = Context()
    await ctx.use(LLMService)
    await ctx.flush()
    llm = ctx.llm
    warn_spy = MagicMock()
    ctx.logger.warning = warn_spy  # type: ignore[method-assign]

    later = MagicMock()

    def _boom() -> None:
        raise RuntimeError("sync listener failed")

    async def _async_boom() -> None:
        raise RuntimeError("async listener failed")

    ctx.on("llm/adapters-updated", _boom)
    ctx.on("llm/adapters-updated", _async_boom)
    ctx.on("llm/adapters-updated", later)

    fiber = ctx.inject(
        ["llm"],
        lambda child_ctx: child_ctx.llm.register_adapter(
            ["fiber-route"], _ScriptedAdapter([])
        ),
    )
    await ctx.flush()
    assert [p["id"] for p in llm.list_providers()] == ["fiber-route"]
    assert later.call_count == 1
    assert warn_spy.call_count == 4
    assert [call.args[0] for call in warn_spy.call_args_list].count(
        "llm: an llm/adapters-updated listener failed"
    ) == 2

    await fiber.dispose()
    assert llm.list_providers() == []


def test_adapters_updated_outside_event_loop_closes_async_listener() -> None:
    """Sync emission outside running loop warns and handles async listener cleanly."""
    ctx, llm = _setup_runtime()
    warn_spy = MagicMock()
    ctx.logger.warning = warn_spy  # type: ignore[method-assign]

    async def _async_listener() -> None:
        return None

    ctx.on("llm/adapters-updated", _async_listener)
    handle = llm.register_adapter(["sync-outside"], _ScriptedAdapter([]))
    assert [p["id"] for p in llm.list_providers()] == ["sync-outside"]
    handle()


@pytest.mark.asyncio
async def test_configurable_provider_directory_lifecycle_and_validation() -> None:
    """Configurable provider directory enforces atomic registration, detach, replace, and disposal."""
    ctx = Context()
    await ctx.use(LLMService)
    await ctx.flush()
    llm = ctx.llm
    events = MagicMock()
    ctx.on("llm/adapters-updated", events)

    env_list = ["OPENAI_API_KEY"]
    base_url_env = ["OPENAI_BASE_URL"]
    src = _entry(
        defaultBaseURL="https://api.openai.com/v1",
        baseURLEnv=base_url_env,
        nativeAuth={"env": env_list, "oauth": False},
    )
    h1 = llm.register_configurable_providers(
        [
            _entry(
                provider="magitech-official",
                displayName="Magitech",
                settingsNs="llm-magitech",
                settingsPath=[],
            ),
            src,
        ]
    )
    assert isinstance(h1, DirectoryRegistrationHandle)
    assert events.call_count == 1

    # Detach from caller & returned lists
    src["displayName"] = "mutated"
    env_list.append("MUTATED")
    base_url_env.append("MUTATED")
    listed = llm.list_configurable_providers()
    assert len(listed) == 2
    listed[0]["displayName"] = "mutated"
    listed[1]["settingsPath"].append("mutated")  # type: ignore[union-attr]
    listed[1]["nativeAuth"]["env"].append("mutated")  # type: ignore[index]
    listed[1]["baseURLEnv"].append("mutated")  # type: ignore[union-attr]
    fresh = llm.list_configurable_providers()
    assert fresh[0]["displayName"] == "Magitech"
    assert fresh[1]["settingsPath"] == ["providers", "openai"]
    assert fresh[1]["nativeAuth"] == {"env": ["OPENAI_API_KEY"], "oauth": False}
    assert fresh[1]["baseURLEnv"] == ["OPENAI_BASE_URL"]

    h1()
    assert llm.list_configurable_providers() == []
    assert events.call_count == 2

    # Invalid inputs
    with pytest.raises(LlmError, match="at least one provider"):
        llm.register_configurable_providers([])
    for bad in (
        _entry(provider=""),
        _entry(displayName=""),
        _entry(settingsNs=""),
        _entry(settingsPath=["providers", ""]),
    ):
        with pytest.raises(LlmError):
            llm.register_configurable_providers([_entry(provider="ok"), bad])
        assert llm.list_configurable_providers() == []

    # Duplicate intra-batch and cross-batch
    with pytest.raises(LlmError) as exc_dup:
        llm.register_configurable_providers([_entry(), _entry()])
    assert exc_dup.value.code == "DUPLICATE_DIRECTORY"

    handle = llm.register_configurable_providers([_entry(), _entry(provider="second")])
    other = llm.register_configurable_providers([_entry(provider="owned")])
    with pytest.raises(LlmError, match="already declared"):
        handle.replace([_entry(provider="owned")])
    assert sorted(e["provider"] for e in llm.list_configurable_providers()) == [
        "openai",
        "owned",
        "second",
    ]
    handle.replace([_entry(displayName="Renamed")])
    assert [
        e["displayName"]
        for e in llm.list_configurable_providers()
        if e["provider"] == "openai"
    ] == ["Renamed"]
    handle.replace([])
    assert [e["provider"] for e in llm.list_configurable_providers()] == ["owned"]
    handle()
    with pytest.raises(LlmError) as exc_dir_disposed:
        handle.replace([_entry()])
    assert exc_dir_disposed.value.code == "REGISTRATION_DISPOSED"
    other()

    # Fiber scoped
    fiber = ctx.inject(
        ["llm"],
        lambda c: c.llm.register_configurable_providers([_entry()]),
    )
    await ctx.flush()
    assert len(llm.list_configurable_providers()) == 1
    await fiber.dispose()
    assert llm.list_configurable_providers() == []


@pytest.mark.asyncio
async def test_model_discovery_sync_async_signal_and_remote_errors() -> None:
    """Model discovery handles sync/async hooks, dedup, validation, and RemoteError mapping."""
    _, llm = _setup_runtime()

    with pytest.raises(LlmError) as exc_ns:
        llm.register_model_discovery("", lambda req: [])
    assert exc_ns.value.code == "INVALID_DISCOVERY"

    calls: list[Any] = []

    def _sync_discover(req: dict[str, Any]) -> list[dict[str, Any]]:
        calls.append(req)
        return [
            {
                "id": "keep",
                "name": "Keep",
                "contextWindow": 1024,
                "maxTokens": 256,
                "inputModalities": ["text", "image"],
            },
            {"id": ""},
            {"id": "keep"},
            {"id": "bare"},
        ]

    dispose = llm.register_model_discovery("llm-example", _sync_discover)
    with pytest.raises(LlmError) as exc_dup:
        llm.register_model_discovery("llm-example", lambda req: [])
    assert exc_dup.value.code == "DUPLICATE_DISCOVERY"

    res = await llm.discover_models(
        "llm-example", {"baseURL": "https://gateway.example/v1"}
    )
    assert res == [
        {
            "id": "keep",
            "name": "Keep",
            "contextWindow": 1024,
            "maxTokens": 256,
            "inputModalities": ["text", "image"],
        },
        {"id": "bare"},
    ]
    assert calls == [{"baseURL": "https://gateway.example/v1"}]

    with pytest.raises(LlmError) as exc_missing:
        await llm.discover_models("absent", {"baseURL": "https://gateway.example/v1"})
    assert exc_missing.value.code == "NO_DISCOVERY"

    for invalid_req in ({}, {"baseURL": ""}, {"provider": "", "baseURL": ""}):
        with pytest.raises(LlmError) as exc_req:
            await llm.discover_models("llm-example", invalid_req)  # type: ignore[arg-type]
        assert exc_req.value.code == "INVALID_DISCOVERY"

    # Malformed entries (empty or non-string ids) skipped cleanly
    dispose()
    llm.register_model_discovery(
        "llm-bad",
        lambda req: [{"id": ""}, {"id": 123}, {"id": "usable"}],  # type: ignore[list-item,return-value]
    )
    assert await llm.discover_models("llm-bad", {"provider": "known"}) == [
        {"id": "usable"}
    ]

    # Async discover accepting signal + RemoteError wrapping
    outcomes: list[Any] = [
        [{"id": "ok"}],
        RuntimeError("endpoint offline"),
        ValueError("provider refused"),
    ]

    async def _async_discover(
        req: dict[str, Any], signal: AbortSignal | None = None
    ) -> list[dict[str, Any]]:
        assert signal is not None
        nxt = outcomes.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    llm.register_model_discovery("llm-remote", _async_discover)
    sig = AbortSignal()
    assert await llm.remote_discover_models(
        "llm-remote", {"baseURL": "https://gateway.example/v1"}, sig
    ) == [{"id": "ok"}]

    with pytest.raises(RemoteError) as exc_rem1:
        await llm.remote_discover_models(
            "llm-remote", {"baseURL": "https://gateway.example/v1"}, sig
        )
    assert exc_rem1.value.code == "llm/model-discovery-rejected"
    assert exc_rem1.value.details == {
        "settingsNs": "llm-remote",
        "baseURL": "https://gateway.example/v1",
    }

    with pytest.raises(RemoteError) as exc_rem2:
        await llm.remote_discover_models("llm-remote", {"provider": "route"}, sig)
    assert exc_rem2.value.code == "llm/model-discovery-rejected"
    assert exc_rem2.value.details == {"settingsNs": "llm-remote"}


@pytest.mark.asyncio
async def test_retry_policy_image_pricing_target_and_file_request_text() -> None:
    """Service resolves retry policies, image capabilities, and host-mapped file handles."""
    ctx, llm = _setup_runtime()
    pricing = {
        "priceImages": lambda blocks: [{"visualTokens": 85, "text": ""} for _ in blocks]
    }
    custom_target = {"width": 512, "height": 512, "maxBytes": 2048}
    policy_a = resolve_retry_policy({"mode": "normal", "maxRetries": 5})
    adapter = _CatalogAdapter(
        _sample_script(),
        retry=policy_a,
        pricing=pricing,
        image_target=custom_target,
    )
    dispose = llm.register_adapter(["a"], adapter)
    llm.register_adapter(["plain"], _ScriptedAdapter([]))

    assert llm.provider_retry_policy("a") == policy_a
    assert llm.provider_retry_policy("plain") == resolve_retry_policy(None)
    with pytest.raises(LlmError) as exc_retry_missing:
        llm.provider_retry_policy("missing")
    assert exc_retry_missing.value.code == "NO_ADAPTER"

    assert llm.image_request_pricing("a", "vision") is pricing
    assert llm.image_request_pricing("plain", "vision") is None
    assert llm.image_request_pricing("missing", "vision") is None

    img_ref = {
        "attachmentId": "att-img",
        "mediaType": "image/png",
        "bytes": 5000,
        "width": 1000,
        "height": 1000,
    }
    assert llm.image_request_target("a", "vision", img_ref) == custom_target
    with pytest.raises(LlmError) as exc_img_missing:
        llm.image_request_target("missing", "vision", img_ref)
    assert exc_img_missing.value.code == "NO_ADAPTER"

    dispose()
    assert llm.image_request_pricing("a", "vision") is None

    file_ref = {"attachmentId": "att-file", "name": "notes.md", "bytes": 42}
    # Without attachments service -> handle text has no path
    text_no_att = llm.file_request_text(file_ref)
    assert "42 bytes" in text_no_att and "notes.md" in text_no_att

    # Attachments raising -> swallowed safely
    ctx.provide(
        "attachments",
        SimpleNamespace(
            fileHostPath=lambda r: (_ for _ in ()).throw(ValueError("bad ref"))
        ),
    )
    assert llm.file_request_text(file_ref) == text_no_att

    # Attachments without fs -> no path
    ctx.provide(
        "attachments", SimpleNamespace(file_host_path=lambda r: "/host/notes.md")
    )
    assert llm.file_request_text(file_ref) == text_no_att

    # Both attachments and fs without callable mapper -> no path
    ctx.provide("fs", SimpleNamespace())
    assert llm.file_request_text(file_ref) == text_no_att

    # Both attachments and fs resolving mapped process path
    ctx.provide(
        "fs", SimpleNamespace(processPathFromHostPath=lambda p: "/workspace/notes.md")
    )
    text_with_path = llm.file_request_text(file_ref)
    assert "/workspace/notes.md" in text_with_path


@pytest.mark.asyncio
async def test_list_models_and_resolve_model_info_validation_and_detach() -> None:
    """Catalog discovery and exact model resolution validate metadata and detach structures."""
    _, llm = _setup_runtime()

    badge_translations = {"en": "Fast", "zh": "快"}
    modalities = ["text", "image"]
    adapter = _CatalogAdapter(
        _sample_script(),
        provider_info_map={"cat": {"id": "cat", "name": "Catalog Provider"}},
        models=[
            {
                "provider": "cat",
                "id": "m1",
                "name": "Model 1",
                "inputModalities": modalities,
                "badge": {"text": badge_translations, "tone": "info"},
            },
            {
                "provider": "cat",
                "id": "m2",
                "name": "Model 2",
                "badge": {"text": "New", "tone": "success"},
            },
        ],
        resolved={
            "m1": {
                "provider": "cat",
                "id": "m1",
                "name": "Model 1",
                "inputModalities": modalities,
                "context": {"contextWindow": 8192},
                "defaultMaxTokens": 1024,
                "reasoning": {
                    "efforts": [
                        {"id": "low", "name": "Low"},
                        {"id": "high", "name": "High"},
                    ],
                    "defaultEffort": "low",
                },
                "systemPromptUpdate": "in-history",
                "toolUpdate": "addition-only",
            }
        },
    )
    llm.register_adapter(["cat"], adapter)

    listed = await llm.list_models("cat")
    assert len(listed) == 2
    assert listed[0]["badge"] == {"text": {"en": "Fast", "zh": "快"}, "tone": "info"}
    badge_translations["en"] = "HACKED"
    modalities.append("HACKED")
    assert listed[0]["badge"]["text"]["en"] == "Fast"  # type: ignore[index]
    assert listed[0]["inputModalities"] == ["text", "image"]

    resolved = await llm.resolve_model_info("cat", "m1")
    assert resolved["context"] == {"contextWindow": 8192}
    assert resolved["defaultMaxTokens"] == 1024
    assert resolved["systemPromptUpdate"] == "in-history"
    assert resolved["toolUpdate"] == "addition-only"

    with pytest.raises(LlmError) as exc_missing:
        await llm.list_models("missing")
    assert exc_missing.value.code == "NO_ADAPTER"

    # Duplicate catalog ids & bad catalog
    bad_cat = _CatalogAdapter(
        [],
        models=[
            {"provider": "dup", "id": "x", "name": "X"},
            {"provider": "dup", "id": "x", "name": "X2"},
        ],
    )
    llm.register_adapter(["dup"], bad_cat)
    with pytest.raises(LlmError) as exc_dup_cat:
        await llm.list_models("dup")
    assert exc_dup_cat.value.code == "INVALID_CATALOG"

    # Validation checks on resolve_model_info
    bad_models = _CatalogAdapter(
        [],
        resolved={
            "bad-provider": {"provider": "other", "id": "bad-provider", "name": "Bad"},
            "bad-ctx": {
                "provider": "bad",
                "id": "bad-ctx",
                "name": "BadCtx",
                "context": {"contextWindow": 0},
            },
            "bad-max": {
                "provider": "bad",
                "id": "bad-max",
                "name": "BadMax",
                "defaultMaxTokens": -1,
            },
            "bad-reason": {
                "provider": "bad",
                "id": "bad-reason",
                "name": "BadReason",
                "reasoning": {"efforts": []},
            },
            "bad-default-effort": {
                "provider": "bad",
                "id": "bad-default-effort",
                "name": "BadDef",
                "reasoning": {
                    "efforts": [{"id": "low", "name": "Low"}],
                    "defaultEffort": "ultra",
                },
            },
            "bad-sys": {
                "provider": "bad",
                "id": "bad-sys",
                "name": "BadSys",
                "systemPromptUpdate": "out-of-band",
            },
            "bad-tool": {
                "provider": "bad",
                "id": "bad-tool",
                "name": "BadTool",
                "toolUpdate": "illegal",
            },
        },
    )
    llm.register_adapter(["bad"], bad_models)
    for model_id, expected_code in (
        ("bad-provider", "INVALID_MODEL_INFO"),
        ("bad-ctx", "INVALID_MODEL_CONTEXT"),
        ("bad-max", "INVALID_MODEL_MAX_TOKENS"),
        ("bad-reason", "INVALID_MODEL_REASONING"),
        ("bad-default-effort", "INVALID_MODEL_REASONING"),
        ("bad-sys", "INVALID_MODEL_INFO"),
        ("bad-tool", "INVALID_MODEL_INFO"),
    ):
        with pytest.raises(LlmError) as exc_val:
            await llm.resolve_model_info("bad", model_id)
        assert exc_val.value.code == expected_code


@pytest.mark.asyncio
async def test_resolve_call_config_and_prepare_call_one_shot_and_cancellation() -> None:
    """prepare_call binds registration & config once, applies defaults/configure, and guards drift."""
    _, llm = _setup_runtime()

    rec = _RecordingAdapter(_sample_script())

    prep_retry = resolve_retry_policy({"mode": "normal", "maxRetries": 4})

    class _PrepAdapter(_CatalogAdapter):
        def __init__(self) -> None:
            super().__init__(
                _sample_script(),
                retry=prep_retry,
                resolved={
                    "m": {
                        "provider": "p",
                        "id": "m",
                        "name": "M",
                        "context": {"contextWindow": 16384},
                        "defaultMaxTokens": 2048,
                        "reasoning": {
                            "efforts": [
                                {"id": "low", "name": "Low"},
                                {"id": "high", "name": "High"},
                            ],
                            "defaultEffort": "low",
                        },
                        "inputModalities": ["text"],
                        "systemPromptUpdate": "in-history",
                        "toolUpdate": "in-history",
                    }
                },
            )
            self.seen_opts: dict[str, Any] | None = None

        async def prepare_call(
            self, provider: str, model: str, signal: AbortSignal | None = None
        ) -> PreparedAdapterCall:
            info = await self.resolve_model(provider, model, signal)

            async def _stream(opts: Any) -> AsyncIterator[dict[str, Any]]:
                self.seen_opts = dict(opts)
                async for c in rec.stream(opts):
                    yield c

            return PreparedAdapterCall(model=info, stream=_stream)

    adapter = _PrepAdapter()
    reg = llm.register_adapter(["p"], adapter)

    resolved_cfg = await llm.resolve_call_config({"provider": "p", "model": "m"})
    assert resolved_cfg["maxTokens"] == 2048
    assert resolved_cfg["reasoningEffort"] == "low"

    explicit_cfg = await llm.resolve_call_config(
        {"provider": "p", "model": "m", "maxTokens": 512, "reasoningEffort": "high"}
    )
    assert explicit_cfg["maxTokens"] == 512
    assert explicit_cfg["reasoningEffort"] == "high"

    with pytest.raises(LlmError) as exc_effort:
        await llm.resolve_call_config(
            {"provider": "p", "model": "m", "reasoningEffort": "extreme"}
        )
    assert exc_effort.value.code == "UNSUPPORTED_REASONING_EFFORT"

    adapter.resolve_calls = 0
    prepared = await llm.prepare_call(
        {"provider": "p", "model": "m"},
        configure=lambda controls, info: {
            "reasoningEffort": "high",
            "temperature": 0.25,
        },
    )
    assert adapter.resolve_calls == 1
    assert isinstance(prepared, PreparedLlmCall)
    assert prepared.config["maxTokens"] == 2048
    assert prepared.config["reasoningEffort"] == "high"
    assert prepared.config["temperature"] == 0.25
    assert prepared.retryPolicy == prep_retry
    assert prepared.adapterDefaults == {"maxTokens": True}
    assert prepared.context == {"contextWindow": 16384}
    assert prepared.inputModalities == ["text"]
    assert prepared.systemPromptUpdate == "in-history"
    assert prepared.toolUpdate == "in-history"

    # Replace registration route away AFTER prepare_call; prepared call still binds old registration
    reg.replace(["p-renamed"])
    llm.register_adapter(
        ["p"], _ScriptedAdapter([{"type": "text-delta", "text": "wrong"}])
    )

    # Config drift rejected
    with pytest.raises(LlmError) as exc_drift:
        async for _ in prepared.stream(
            {"provider": "p", "model": "m", "maxTokens": 999, "messages": []}
        ):
            pass
    assert exc_drift.value.code == "INVALID_PREPARED_CALL"

    chunks = [
        c
        async for c in prepared.stream(
            {
                "provider": "p",
                "model": "m",
                "maxTokens": 2048,
                "reasoningEffort": "high",
                "temperature": 0.25,
                "messages": [],
            }
        )
    ]
    assert chunks[0]["text"] == "hello"
    assert adapter.seen_opts is not None

    # One-shot enforcement
    with pytest.raises(LlmError) as exc_reuse:
        async for _ in prepared.stream(
            {
                "provider": "p",
                "model": "m",
                "maxTokens": 2048,
                "reasoningEffort": "high",
                "temperature": 0.25,
                "messages": [],
            }
        ):
            pass
    assert exc_reuse.value.code == "INVALID_PREPARED_CALL"

    # Cancellation before prepare_call skips configure
    pre_aborted = AbortSignal()
    pre_aborted.abort("stopped")
    configure_spy = MagicMock()
    with pytest.raises(AbortError):
        await llm.prepare_call(
            {"provider": "p", "model": "m"}, signal=pre_aborted, configure=configure_spy
        )
    configure_spy.assert_not_called()

    # Cancellation raised inside configure
    mid_sig = AbortSignal()

    def _abort_during_configure(controls: Any, info: Any) -> dict[str, Any]:
        del controls, info
        mid_sig.abort()
        return {"temperature": 0.5}

    with pytest.raises(AbortError):
        await llm.prepare_call(
            {"provider": "p", "model": "m"},
            signal=mid_sig,
            configure=_abort_during_configure,
        )

    # Unsupported effort returned from configure
    with pytest.raises(LlmError) as exc_bad_cfg_eff:
        await llm.prepare_call(
            {"provider": "p", "model": "m"},
            configure=lambda c, i: {"reasoningEffort": "invalid-level"},
        )
    assert exc_bad_cfg_eff.value.code == "UNSUPPORTED_REASONING_EFFORT"


@pytest.mark.asyncio
async def test_stream_waterfall_routing_projections_and_replay_isolation() -> None:
    """stream handles waterfall routing, post-waterfall projections, and adapter-scoped replayState."""
    ctx, llm = _setup_runtime()

    adapter_a = _RecordingAdapter(_sample_script())
    adapter_b = _RecordingAdapter(_sample_script())
    adapter_b_cat = _CatalogAdapter(
        _sample_script(),
        resolved={
            "text-only": {
                "provider": "b",
                "id": "text-only",
                "name": "Text Only",
                "inputModalities": ["text"],
                "reasoning": {
                    "efforts": [{"id": "med", "name": "Medium"}],
                    "defaultEffort": "med",
                },
                "toolUpdate": "addition-only",
            }
        },
    )

    class _CapturingTextOnlyAdapter(_CatalogAdapter):
        def __init__(self) -> None:
            super().__init__(
                _sample_script(),
                resolved=adapter_b_cat._resolved,
            )
            self.last_options: dict[str, Any] | None = None

        async def stream(
            self, options: Any, signal: AbortSignal | None = None
        ) -> AsyncIterator[dict[str, Any]]:
            self.last_options = dict(options)
            async for chunk in super().stream(options, signal):
                yield chunk

    capturing_b = _CapturingTextOnlyAdapter()

    # Register two providers sharing adapter_a, plus provider "b" on a separate adapter instance
    llm.register_adapter(["a1", "a2"], adapter_a)
    llm.register_adapter(["b"], capturing_b)

    seen_waterfall_img = False

    async def _route_to_b(
        options: dict[str, Any], next_fn: Any
    ) -> AsyncIterator[dict[str, Any]]:
        nonlocal seen_waterfall_img
        # Before final projection, image block is still visible inside waterfall
        seen_waterfall_img = any(
            isinstance(b, dict) and b.get("type") == "image"
            for m in options.get("messages", [])
            for b in m.get("content", [])
            if isinstance(m.get("content"), list)
        )
        rerouted = dict(options)
        rerouted["provider"] = "b"
        rerouted["model"] = "text-only"
        return await next_fn(rerouted)

    dispose_route = ctx.on("llm/stream", _route_to_b, mode=EventMode.WATERFALL)

    frozen_req = deep_freeze(
        {
            "provider": "a1",
            "model": "vision-a",
            "messages": [
                create_user_message(
                    content=[
                        {
                            "type": "image",
                            "attachment": {
                                "attachmentId": "att-1",
                                "mediaType": "image/png",
                                "bytes": 100,
                                "width": 10,
                                "height": 10,
                            },
                        },
                        {
                            "type": "file",
                            "attachment": {
                                "attachmentId": "att-f",
                                "name": "data.txt",
                                "bytes": 12,
                            },
                        },
                    ]
                ),
                create_assistant_message(
                    content=[{"type": "text", "text": "prior turn"}],
                    source={
                        "provider": "a1",
                        "model": "vision-a",
                        "replayState": {"raw": "keep-or-strip"},
                    },
                ),
                create_assistant_message(
                    content=[{"type": "text", "text": "same adapter as b"}],
                    source={
                        "provider": "b",
                        "model": "text-only",
                        "replayState": {"raw": "must-keep-for-b"},
                    },
                ),
                create_developer_message(
                    content=[{"type": "tool-removal", "toolName": "old_tool"}]
                ),
            ],
        }
    )

    # Invoke via keyword-only overload: stream(options=...)
    chunks = [c async for c in llm.stream(options=frozen_req)]
    assert len(chunks) == 2
    assert seen_waterfall_img is True

    dispatched = capturing_b.last_options
    assert dispatched is not None
    assert dispatched["provider"] == "b"
    assert dispatched["reasoningEffort"] == "med"
    msgs = dispatched["messages"]
    # Image downgraded to text for text-only model AFTER waterfall; file downgraded to text handle
    user_blocks = msgs[0]["content"]
    assert all(b["type"] == "text" for b in user_blocks)
    # Assistant from a1 had replayState stripped because target is capturing_b; provider/model kept
    assert msgs[1]["source"]["provider"] == "a1"
    assert msgs[1]["source"]["model"] == "vision-a"
    assert "replayState" not in msgs[1]["source"]
    # Assistant from b kept replayState because target is same adapter instance
    assert msgs[2]["source"]["replayState"] == {"raw": "must-keep-for-b"}
    # Original frozen request remained untouched
    assert "replayState" in frozen_req["messages"][1]["source"]
    assert adapter_b is not capturing_b
    dispose_route()

    # Same-adapter multi-provider (a1 -> a2): replayState from a1 retained, b stripped.
    a2_req = deep_freeze(
        {
            "provider": "a2",
            "model": "vision-a",
            "messages": frozen_req["messages"],
        }
    )
    a2_chunks = [c async for c in llm.stream(a2_req)]
    assert len(a2_chunks) == 2
    assert adapter_a.last_options is not None
    a2_msgs = adapter_a.last_options["messages"]
    assert a2_msgs[1]["source"]["replayState"] == {"raw": "keep-or-strip"}
    assert "replayState" not in a2_msgs[2]["source"]
    assert a2_msgs[2]["source"]["provider"] == "b"


@pytest.mark.asyncio
async def test_stream_failure_normalization_abort_middleware_and_close() -> None:
    """Adapter failures normalize to finish chunks; middleware and cleanup errors propagate."""
    ctx, llm = _setup_runtime()

    # 1. Missing provider -> terminal error finish chunk
    chunks_missing = [
        c
        async for c in llm.stream({"provider": "absent", "model": "m", "messages": []})
    ]
    assert chunks_missing == [
        {
            "type": "finish",
            "reason": {
                "kind": "error",
                "failure": {
                    "message": 'no adapter registered for provider "absent"',
                    "code": "NO_ADAPTER",
                },
            },
        }
    ]

    # 2. Adapter raising synchronously -> terminal error / aborted finish chunk
    class _SyncThrowAdapter(LlmAdapter):
        def stream(self, options: Any) -> AsyncIterator[dict[str, Any]]:
            del options
            raise ValueError("adapter crashed")

    llm.register_adapter(["boom"], _SyncThrowAdapter())
    err_chunks = [
        c async for c in llm.stream({"provider": "boom", "model": "m", "messages": []})
    ]
    assert err_chunks[0]["type"] == "finish"
    assert err_chunks[0]["reason"]["kind"] == "error"
    assert "adapter crashed" in err_chunks[0]["reason"]["failure"]["message"]

    sig = AbortSignal()
    sig.abort("user cancelled")
    aborted_chunks = [
        c
        async for c in llm.stream(
            {"provider": "boom", "model": "m", "messages": [], "signal": sig}
        )
    ]
    assert aborted_chunks[0]["reason"]["kind"] == "aborted"

    # 3. Adapter yielding asynchronously: no aclose on normal exhaustion, closed & errors propagated on early close
    close_calls = 0
    remaining = 1
    close_error: RuntimeError | None = None

    class _CloseableIter:
        def __aiter__(self) -> _CloseableIter:
            return self

        async def __anext__(self) -> dict[str, Any]:
            nonlocal remaining
            if remaining <= 0:
                raise StopAsyncIteration
            remaining -= 1
            return {"type": "text-delta", "text": "one"}

        async def aclose(self) -> None:
            nonlocal close_calls
            close_calls += 1
            if close_error is not None:
                raise close_error

    class _CloseAdapter(LlmAdapter):
        def stream(self, options: Any) -> AsyncIterator[dict[str, Any]]:
            del options
            return _CloseableIter()  # type: ignore[return-value]

    llm.register_adapter(["closable"], _CloseAdapter())

    # Normal exhaustion: aclose() is NOT invoked.
    remaining = 1
    exhausted = [
        c
        async for c in llm.stream(
            {"provider": "closable", "model": "m", "messages": []}
        )
    ]
    assert len(exhausted) == 1
    assert close_calls == 0

    # Early exit: aclose() IS invoked once.
    remaining = 2
    stream = llm.stream({"provider": "closable", "model": "m", "messages": []})
    async for ch in stream:
        assert ch["text"] == "one"
        break
    await stream.aclose()  # type: ignore[attr-defined]
    assert close_calls == 1

    # Early exit when aclose() fails: cleanup error propagates.
    remaining = 2
    close_error = RuntimeError("close boom")
    err_stream = llm.stream({"provider": "closable", "model": "m", "messages": []})
    async for ch in err_stream:
        assert ch["text"] == "one"
        break
    with pytest.raises(RuntimeError, match="close boom"):
        await err_stream.aclose()  # type: ignore[attr-defined]
    close_error = None

    # 4. Middleware error (setup and mid-stream) propagates without extra aclose()
    async def _bad_mw(options: Any, next_fn: Any) -> Any:
        del options, next_fn
        raise RuntimeError("middleware boom")

    dispose_mw = ctx.on("llm/stream", _bad_mw, mode=EventMode.WATERFALL)
    with pytest.raises(RuntimeError, match="middleware boom"):
        async for _ in llm.stream(
            {"provider": "closable", "model": "m", "messages": []}
        ):
            pass
    dispose_mw()

    mw_close_calls = 0

    class _ThrowingMwIter:
        def __aiter__(self) -> _ThrowingMwIter:
            return self

        async def __anext__(self) -> dict[str, Any]:
            raise RuntimeError("mw iter boom")

        async def aclose(self) -> None:
            nonlocal mw_close_calls
            mw_close_calls += 1

    async def _throwing_mw(options: Any, next_fn: Any) -> Any:
        del options
        inner = await next_fn({"provider": "closable", "model": "m", "messages": []})
        await inner.aclose()
        return _ThrowingMwIter()

    dispose_throw_mw = ctx.on("llm/stream", _throwing_mw, mode=EventMode.WATERFALL)
    with pytest.raises(RuntimeError, match="mw iter boom"):
        async for _ in llm.stream(
            {"provider": "closable", "model": "m", "messages": []}
        ):
            pass
    dispose_throw_mw()
    assert mw_close_calls == 0


def test_misc_duck_typed_camel_adapter() -> None:
    """Plain duck-typed camelCase adapter methods work without inheriting LlmAdapter."""
    _, llm = _setup_runtime()
    camel_retry = resolve_retry_policy({"mode": "normal", "maxRetries": 2})
    camel = SimpleNamespace(
        providerInfo=lambda p: {"id": p, "name": "Camel Name"},
        providerRetryPolicy=lambda p: camel_retry,
        stream=lambda opts: _ScriptedAdapter(_sample_script()).stream(opts),
    )
    llm.register_adapter(["camel"], camel)
    assert llm.list_providers() == [{"id": "camel", "name": "Camel Name"}]
    assert llm.provider_retry_policy("camel") == camel_retry


@pytest.mark.asyncio
async def test_remaining_service_edge_branches_and_metadata() -> None:
    """Cover remaining adapter return styles, optional signals, metadata validation, and stream boundaries."""
    ctx, llm = _setup_runtime()

    # 1. Base fallback on duck-typed adapter omitting image_request_target
    duck = SimpleNamespace(
        providerInfo=lambda p: {"id": p, "name": p},
        listModels=lambda p: [
            {
                "provider": p,
                "id": "cat-desc",
                "name": "Catalog With Desc",
                "description": "Catalog description",
            }
        ],
        resolveModel=lambda p, m: {
            "provider": p,
            "id": m,
            "name": "Resolved",
            "description": "Resolved description",
            "reasoning": {
                "efforts": [{"id": "low", "name": "Low", "description": "Fast effort"}]
            },
        },
        stream=lambda opts: _ScriptedAdapter(_sample_script()).stream(opts),
    )
    llm.register_adapter(["duck"], duck)
    assert llm.image_request_target(
        "duck",
        "cat-desc",
        {
            "attachmentId": "att-img",
            "mediaType": "image/png",
            "bytes": 100,
            "width": 640,
            "height": 480,
        },
    ) == {"width": 640, "height": 480, "maxBytes": 100}

    # Sync listModels / resolveModel + description + no defaultEffort + resolve_call_config effective=None
    catalog = await llm.list_models("duck")
    assert catalog[0]["description"] == "Catalog description"
    resolved_no_def = await llm.resolve_model_info("duck", "cat-desc")
    assert resolved_no_def["description"] == "Resolved description"
    assert resolved_no_def["reasoning"] == {
        "efforts": [{"id": "low", "name": "Low", "description": "Fast effort"}]
    }
    cfg_no_effort = await llm.resolve_call_config(
        {"provider": "duck", "model": "cat-desc"}
    )
    assert "reasoningEffort" not in cfg_no_effort

    # 2. _invoke_with_optional_signal: 1-arg sync discover (no signal param) and invalid inspect.signature
    class _OpaqueCallable:
        @property
        def __signature__(self) -> Any:
            raise ValueError("opaque")

        def __call__(
            self, req: dict[str, Any], signal: AbortSignal | None = None
        ) -> list[dict[str, Any]]:
            del req, signal
            return [{"id": "opaque-id"}]

    llm.register_model_discovery("one-arg", lambda req: [{"id": "sync-1"}])
    llm.register_model_discovery("opaque", _OpaqueCallable())
    sig_ok = AbortSignal()
    assert await llm.discover_models("one-arg", {"provider": "k"}, signal=sig_ok) == [
        {"id": "sync-1"}
    ]
    assert await llm.discover_models("opaque", {"provider": "k"}, signal=sig_ok) == [
        {"id": "opaque-id"}
    ]

    # 3. Duck-typed signal without throw_if_aborted
    with pytest.raises(AbortError, match="custom halt"):
        await llm.prepare_call(
            {"provider": "duck", "model": "cat-desc"},
            signal=SimpleNamespace(aborted=True, reason="custom halt"),  # type: ignore[arg-type]
        )
    assert (
        await llm.prepare_call(
            {"provider": "duck", "model": "cat-desc"},
            signal=SimpleNamespace(aborted=False),  # type: ignore[arg-type]
        )
    ).config["provider"] == "duck"

    # 4. Attachments service present without file_host_path/fileHostPath callable
    ctx.provide("attachments", SimpleNamespace())
    file_ref = {"attachmentId": "att-f", "name": "empty.txt", "bytes": 1}
    assert "empty.txt" in llm.file_request_text(file_ref)

    # 5. list_models non-Mapping catalog item & resolve_model_info bad reasoning efforts
    bad_cat_item = _CatalogAdapter([], models=[123])  # type: ignore[list-item]
    llm.register_adapter(["bad-cat-item"], bad_cat_item)
    with pytest.raises(LlmError) as exc_non_map_cat:
        await llm.list_models("bad-cat-item")
    assert exc_non_map_cat.value.code == "INVALID_CATALOG"

    bad_reason = _CatalogAdapter(
        [],
        resolved={
            "non-map-effort": {
                "provider": "bad-r",
                "id": "non-map-effort",
                "name": "N",
                "reasoning": {"efforts": ["bad"]},
            },
            "empty-effort": {
                "provider": "bad-r",
                "id": "empty-effort",
                "name": "E",
                "reasoning": {"efforts": [{"id": "", "name": "Bad"}]},
            },
        },
    )
    llm.register_adapter(["bad-r"], bad_reason)
    for model_id in ("non-map-effort", "empty-effort"):
        with pytest.raises(LlmError) as exc_reason:
            await llm.resolve_model_info("bad-r", model_id)
        assert exc_reason.value.code == "INVALID_MODEL_REASONING"

    # 6. prepare_call returning dict / SimpleNamespace + configure=None + adapterDefaults reasoningEffort
    async def _direct_stream(opts: Any) -> AsyncIterator[dict[str, Any]]:
        del opts
        yield {"type": "text-delta", "text": "prepared-ok"}

    dict_adapter = SimpleNamespace(
        prepareCall=lambda p, m: {
            "model": {
                "provider": p,
                "id": m,
                "name": m,
                "defaultMaxTokens": 128,
                "reasoning": {
                    "efforts": [{"id": "low", "name": "Low"}],
                    "defaultEffort": "low",
                },
            },
            "stream": _direct_stream,
        }
    )
    ns_adapter = SimpleNamespace(
        prepare_call=lambda p, m: SimpleNamespace(
            model={"provider": p, "id": m, "name": m},
            stream=_direct_stream,
        )
    )
    llm.register_adapter(["dict-adp"], dict_adapter)
    llm.register_adapter(["ns-adp"], ns_adapter)

    prep_defaulted = await llm.prepare_call({"provider": "dict-adp", "model": "m"})
    assert prep_defaulted.adapterDefaults == {
        "maxTokens": True,
        "reasoningEffort": True,
    }
    prep_explicit_max = await llm.prepare_call(
        {"provider": "dict-adp", "model": "m", "maxTokens": 64}
    )
    assert prep_explicit_max.adapterDefaults == {"reasoningEffort": True}
    prep_ns = await llm.prepare_call({"provider": "ns-adp", "model": "m"})
    assert [
        c
        async for c in prep_ns.stream(
            {"provider": "ns-adp", "model": "m", "messages": []}
        )
    ] == [{"type": "text-delta", "text": "prepared-ok"}]

    # 7. _adapter_stream middleware mutating PreparedLlmCall config triggers INVALID_PREPARED_CALL finish chunk
    prep_drift = await llm.prepare_call({"provider": "dict-adp", "model": "m"})

    async def _drift_mw(
        options: dict[str, Any], next_fn: Any
    ) -> AsyncIterator[dict[str, Any]]:
        mutated = dict(options)
        mutated["maxTokens"] = 999999
        return await next_fn(mutated)

    dispose_mw = ctx.on("llm/stream", _drift_mw, mode=EventMode.WATERFALL)
    drift_chunks = [
        c
        async for c in prep_drift.stream(
            {
                "provider": "dict-adp",
                "model": "m",
                "maxTokens": 128,
                "reasoningEffort": "low",
                "messages": [],
            }
        )
    ]
    dispose_mw()
    assert drift_chunks[0]["reason"]["failure"]["code"] == "INVALID_PREPARED_CALL"

    # 8. Mutable options + assistant without replayState + awaitable stream() + mid-stream error + close styles
    async def _awaitable_stream_fn(opts: Any) -> AsyncIterator[dict[str, Any]]:
        del opts
        return _ScriptedAdapter(_sample_script()).stream({})

    llm.register_adapter(["await-adp"], SimpleNamespace(stream=_awaitable_stream_fn))
    mutable_opts: dict[str, Any] = {
        "provider": "await-adp",
        "model": "m",
        "tools": [
            {
                "name": "t1",
                "description": "d",
                "parameters": {},
                "deferLoading": True,
            }
        ],
        "messages": [
            create_user_message(content=[{"type": "file", "attachment": file_ref}]),
            create_assistant_message(
                content=[{"type": "text", "text": "plain"}],
                provider="await-adp",
                model="m",
            ),
        ],
    }
    await_chunks = [c async for c in llm.stream(mutable_opts)]
    assert len(await_chunks) == 2

    class _MidBoomAdapter(LlmAdapter):
        async def stream(
            self, options: Any, signal: AbortSignal | None = None
        ) -> AsyncIterator[dict[str, Any]]:
            del options, signal
            yield {"type": "text-delta", "text": "before"}
            raise RuntimeError("mid-stream boom")

    llm.register_adapter(["mid-boom"], _MidBoomAdapter())
    mid_chunks = [
        c
        async for c in llm.stream(
            {"provider": "mid-boom", "model": "m", "messages": []}
        )
    ]
    assert mid_chunks[0]["text"] == "before"
    assert "mid-stream boom" in mid_chunks[1]["reason"]["failure"]["message"]

    # Iterators with no aclose vs sync aclose
    sync_closed = False

    class _NoCloseIter:
        def __aiter__(self) -> _NoCloseIter:
            return self

        async def __anext__(self) -> dict[str, Any]:
            return {"type": "text-delta", "text": "x"}

    class _SyncCloseIter(_NoCloseIter):
        def aclose(self) -> None:
            nonlocal sync_closed
            sync_closed = True

    llm.register_adapter(["no-close"], SimpleNamespace(stream=lambda o: _NoCloseIter()))
    llm.register_adapter(
        ["sync-close"], SimpleNamespace(stream=lambda o: _SyncCloseIter())
    )
    s1 = llm.stream({"provider": "no-close", "model": "m", "messages": []})
    async for _ in s1:
        break
    await s1.aclose()  # type: ignore[attr-defined]

    s2 = llm.stream({"provider": "sync-close", "model": "m", "messages": []})
    async for _ in s2:
        break
    await s2.aclose()  # type: ignore[attr-defined]
    assert sync_closed is True


@pytest.mark.asyncio
async def test_llm_service_emit_models_updated_contains_failures() -> None:
    """Verifies llm/models-updated delivers provider arg and contains listener errors."""
    ctx = Context()
    llm = LLMService(ctx)
    seen: list[str] = []

    def _bad(_provider: str) -> None:
        raise RuntimeError("observer failed")

    async def _async_bad(_provider: str) -> None:
        raise RuntimeError("async observer failed")

    ctx.on("llm/models-updated", _bad)
    ctx.on("llm/models-updated", _async_bad)
    ctx.on("llm/models-updated", lambda p: seen.append(p))

    llm.emit_models_updated("route-alpha")
    llm.emitModelsUpdated("route-beta")
    await asyncio.sleep(0)
    assert seen == ["route-alpha", "route-beta"]
