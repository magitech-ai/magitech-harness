"""Unit tests for the pi-ai provider adapter."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.adapter import PiAiAdapter
from magitech_harness.llm.pi_ai.catalog import ModelSpec
from magitech_harness.llm.pi_ai.config import ProviderProfile, ResolvedProfile
from magitech_harness.llm.types import GenerateOptions


def _profile(
    provider: str = "groq",
    **profile_fields: Any,
) -> ResolvedProfile:
    profile = ProviderProfile.model_validate(
        {"apiKeyEnv": "PI_AI_TEST_KEY", **profile_fields}
    )
    spec = ModelSpec(
        provider=provider,
        id="llama-3.1-8b-instant",
        name="Llama 3.1 8B",
        api="openai-completions",
        base_url="https://api.groq.com/openai/v1",
        input=("text",),
        context_window=131072,
        max_tokens=131072,
        reasoning=False,
    )
    return ResolvedProfile(provider, profile, {spec.id: spec})


def _adapter(
    profiles: dict[str, ResolvedProfile],
    handler: Any = None,
) -> PiAiAdapter:
    def _default_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        )

    transport = httpx.MockTransport(handler or _default_handler)

    def factory(profile: ResolvedProfile) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport)

    return PiAiAdapter(profiles=lambda: profiles, client_factory=factory)


def test_adapter_describes_its_routes() -> None:
    """Verifies the adapter describes the routes it serves."""
    profiles = {"groq": _profile()}
    adapter = _adapter(profiles)
    assert adapter.provider_info("groq") == {"id": "groq", "name": "groq"}
    assert adapter.provider_retry_policy("groq") is not None
    assert adapter.provider_retry_policy("absent") is None


def test_adapter_describes_a_renamed_route() -> None:
    """Verifies a configured display name reaches the provider metadata."""
    adapter = _adapter({"groq": _profile(displayName="Groq Cloud")})
    assert adapter.provider_info("groq") == {"id": "groq", "name": "Groq Cloud"}


def test_adapter_creates_a_default_client_without_a_factory() -> None:
    """Verifies the adapter builds its own client when no factory is given."""
    adapter = PiAiAdapter(profiles=dict)
    profile = _profile()
    client = adapter._client(profile)
    assert isinstance(client, httpx.AsyncClient)


def test_adapter_contains_image_ignores_non_mapping_messages() -> None:
    """Verifies image detection skips messages that are not mappings."""
    adapter = PiAiAdapter(profiles=dict)
    options: GenerateOptions = {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "messages": ["not a mapping"],
    }
    assert adapter._contains_image(options) is False


def test_adapter_contains_image_ignores_non_list_content() -> None:
    """Verifies image detection skips content that is not a list."""
    adapter = PiAiAdapter(profiles=dict)
    options: GenerateOptions = {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "plain text"}],
    }
    assert adapter._contains_image(options) is False


@pytest.mark.asyncio
async def test_adapter_lists_models_once_the_credential_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the model listing follows the credential state."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-test")
    adapter = _adapter({"groq": _profile()})
    models = await adapter.list_models("groq")
    assert len(models) == 1
    assert models[0]["id"] == "llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_adapter_offers_no_model_without_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a route whose credential is missing offers no models."""
    monkeypatch.delenv("PI_AI_TEST_KEY", raising=False)
    adapter = _adapter({"groq": _profile()})
    assert await adapter.list_models("groq") == []


@pytest.mark.asyncio
async def test_adapter_refuses_an_unusable_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies an unusable credential fails loudly rather than hiding."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk\nsecret")
    adapter = _adapter({"groq": _profile()})
    with pytest.raises(LlmError) as excinfo:
        await adapter.list_models("groq")
    assert excinfo.value.code == "INVALID_CREDENTIAL"


@pytest.mark.asyncio
async def test_adapter_refuses_an_unknown_route() -> None:
    """Verifies a route the adapter does not own fails loudly."""
    adapter = _adapter({"groq": _profile()})
    with pytest.raises(LlmError) as excinfo:
        await adapter.list_models("absent")
    assert excinfo.value.code == "NO_ADAPTER"


@pytest.mark.asyncio
async def test_adapter_resolves_exact_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies exact-route metadata carries the model capabilities."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-test")
    adapter = _adapter({"groq": _profile()})
    info = await adapter.resolve_model("groq", "llama-3.1-8b-instant")
    assert info["provider"] == "groq"
    assert info["id"] == "llama-3.1-8b-instant"
    assert info["context"] == {"contextWindow": 131072}
    with pytest.raises(LlmError) as excinfo:
        await adapter.resolve_model("groq", "absent")
    assert excinfo.value.code == "UNKNOWN_MODEL"


@pytest.mark.asyncio
async def test_adapter_streams_a_prepared_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a prepared call streams through the configured protocol."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        )

    adapter = _adapter({"groq": _profile()}, handler)
    call = await adapter.prepare_call("groq", "llama-3.1-8b-instant")
    options = {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
    }
    chunks = [chunk async for chunk in call.stream(options)]
    assert next(chunk["type"] for chunk in chunks) == "block-start"
    assert captured["body"]["model"] == "llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_adapter_streams_through_the_stream_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the adapter stream entry dispatches the same protocol."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-test")
    adapter = _adapter({"groq": _profile()})
    chunks = [
        chunk
        async for chunk in adapter.stream(
            {
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "messages": [],
            }
        )
    ]
    assert chunks[-1]["type"] == "finish"


@pytest.mark.asyncio
async def test_adapter_refuses_image_input_for_a_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies an image request fails loudly for a text-only model."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-test")
    adapter = _adapter({"groq": _profile()})
    with pytest.raises(LlmError) as excinfo:
        [
            chunk
            async for chunk in adapter.stream(
                {
                    "provider": "groq",
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image", "attachment": {}}],
                        }
                    ],
                }
            )
        ]
    assert excinfo.value.code == "UNSUPPORTED_CONTENT"


@pytest.mark.asyncio
async def test_adapter_refuses_an_unserved_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a model on an unsupported protocol fails loudly."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-test")
    profile = _profile()
    spec = profile.models["llama-3.1-8b-instant"]
    profile.models["llama-3.1-8b-instant"] = ModelSpec(
        provider=spec.provider,
        id=spec.id,
        name=spec.name,
        api="bedrock-converse-stream",
        base_url=spec.base_url,
        input=spec.input,
        context_window=spec.context_window,
        max_tokens=spec.max_tokens,
        reasoning=spec.reasoning,
    )
    adapter = _adapter({"groq": profile})
    with pytest.raises(LlmError) as excinfo:
        [
            chunk
            async for chunk in adapter.stream(
                {
                    "provider": "groq",
                    "model": "llama-3.1-8b-instant",
                    "messages": [],
                }
            )
        ]
    assert excinfo.value.code == "INVALID_CONFIG"


@pytest.mark.asyncio
async def test_adapter_applies_the_profile_reasoning_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a profile reasoning level reaches the request."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=""
        )

    profile = _profile(reasoning="high")
    spec = profile.models["llama-3.1-8b-instant"]
    profile.models["llama-3.1-8b-instant"] = ModelSpec(
        provider=spec.provider,
        id=spec.id,
        name=spec.name,
        api=spec.api,
        base_url=spec.base_url,
        input=spec.input,
        context_window=spec.context_window,
        max_tokens=spec.max_tokens,
        reasoning=True,
        thinking_level_map={"high": "high"},
    )
    adapter = _adapter({"groq": profile}, handler)
    [
        chunk
        async for chunk in adapter.stream(
            {
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "messages": [],
            }
        )
    ]
    assert captured["body"]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_adapter_sends_the_profile_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies configured provider headers reach the request."""
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=""
        )

    adapter = _adapter({"groq": _profile(headers={"x-route": "blue"})}, handler)
    [
        chunk
        async for chunk in adapter.stream(
            {
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "messages": [],
            }
        )
    ]
    assert captured["headers"]["x-route"] == "blue"
