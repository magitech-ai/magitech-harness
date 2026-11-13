"""Unit tests for the first-prompt model title provider."""

import asyncio
from typing import Any

import pytest

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.types import ModelResponse
from magitech_harness.session.store import Session
from magitech_harness.session.title import TitleProviderRequest
from magitech_harness.session.title_first_prompt import (
    FirstPromptTitleProvider,
    frame_messages,
    resolve_first_prompt_config,
    system_prompt,
    title_from_output,
    title_from_response,
)

CONFIG: dict[str, Any] = {
    "target_words": 5,
    "target_cjk_characters": 10,
    "max_input_bytes": 4096,
    "max_output_tokens": 4096,
    "timeout_ms": 60000,
}


class _RecordingLlm:
    """Fake LLM service recording calls and returning scripted responses."""

    def __init__(self, response: ModelResponse | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response or ModelResponse(text="Recorded Title")
        self.gate: asyncio.Event | None = None

    async def call(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        if self.gate is not None:
            await self.gate.wait()
        return self.response


def _request(
    session: Session,
    messages: list[dict[str, Any]] | None = None,
    route: dict[str, str] | None = None,
) -> TitleProviderRequest:
    return TitleProviderRequest(
        session=session,
        messages=tuple(
            messages if messages is not None else [{"seq": 0, "text": "hi"}]
        ),
        route=route,
        signal=AbortSignal(),
    )


def _session() -> Session:
    return Session.create("s1")


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


def test_resolve_first_prompt_config_valid() -> None:
    config = resolve_first_prompt_config(CONFIG)
    assert config.target_words == 5
    assert config.target_cjk_characters == 10
    assert config.max_input_bytes == 4096
    assert config.max_output_tokens == 4096
    assert config.timeout_ms == 60000
    assert config.provider is None
    assert config.model is None


def test_resolve_first_prompt_config_with_explicit_pair() -> None:
    config = resolve_first_prompt_config({**CONFIG, "provider": "p", "model": "m"})
    assert config.provider == "p"
    assert config.model == "m"


@pytest.mark.parametrize("config", [None, {}, {"target_words": 5}])
def test_resolve_first_prompt_config_missing(config: Any) -> None:
    with pytest.raises(ValueError, match="configuration is required|positive integer"):
        resolve_first_prompt_config(config)


@pytest.mark.parametrize(
    "name,value",
    [
        ("target_words", 0),
        ("target_cjk_characters", -1),
        ("max_input_bytes", 1.5),
        ("max_output_tokens", True),
        ("timeout_ms", 0),
    ],
)
def test_resolve_first_prompt_config_non_positive(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        resolve_first_prompt_config({**CONFIG, name: value})


def test_resolve_first_prompt_config_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown config key 'surprise'"):
        resolve_first_prompt_config({**CONFIG, "surprise": 1})


@pytest.mark.parametrize(
    "pair",
    [
        {"provider": "p"},
        {"model": "m"},
        {"provider": "", "model": "m"},
        {"provider": "p", "model": ""},
    ],
)
def test_resolve_first_prompt_config_pairing(pair: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="provider and model"):
        resolve_first_prompt_config({**CONFIG, **pair})


def test_provider_requires_non_empty_id() -> None:
    llm = _RecordingLlm()
    with pytest.raises(ValueError, match="provider id must be a non-empty string"):
        FirstPromptTitleProvider(CONFIG, llm, "")


# ---------------------------------------------------------------------------
# Prompt, framing, output parsing.
# ---------------------------------------------------------------------------


def test_system_prompt_carries_targets() -> None:
    prompt = system_prompt(resolve_first_prompt_config(CONFIG))
    assert "about 5 words" in prompt
    assert "10 CJK characters" in prompt
    assert "Return only the title on one line" in prompt


def test_frame_messages_is_compact_json() -> None:
    framed = frame_messages([{"seq": 0, "text": "hello"}])
    assert framed.startswith(
        "Generate the session title from this JSON array of human messages:\n"
    )
    assert framed.endswith('[{"seq":0,"text":"hello"}]')


@pytest.mark.parametrize(
    "text,expected",
    [
        ("My Title\nSome commentary after", "My Title"),
        ("\n\n  Spaced Title  \nmore", "Spaced Title"),
        ("First\r\nSecond", "First"),
        ("**Bold Title**", "Bold Title"),
        ("*Star Title*", "Star Title"),
        ("***Triple***", "Triple"),
        ("*a*b* stays", "*a*b* stays"),
        ("**a **b**", "**a **b**"),
        ("", ""),
        ("   \n  ", ""),
    ],
)
def test_title_from_output(text: str, expected: str) -> None:
    assert title_from_output(text) == expected


@pytest.mark.parametrize(
    "response,expected",
    [
        (ModelResponse(text="Hello World", finish_reason="stop"), "Hello World"),
        (
            ModelResponse(text="**Hello World**\ncommentary", finish_reason="stop"),
            "Hello World",
        ),
    ],
)
def test_title_from_response_accepts_stop(
    response: ModelResponse, expected: str
) -> None:
    assert title_from_response(response) == expected


@pytest.mark.parametrize(
    "response,match",
    [
        (
            ModelResponse(text="Hello", finish_reason="tool_calls"),
            "must contain text only",
        ),
        (
            ModelResponse(text="Hello", finish_reason="max_tokens"),
            "reached maxOutputTokens",
        ),
        (
            ModelResponse(text="Hello", finish_reason="error"),
            "unsupported title finish",
        ),
        (
            ModelResponse(
                text="Hello",
                finish_reason="stop",
                tool_calls=[
                    {"call_id": "c1", "name": "tool", "arguments": {}},
                ],
            ),
            "must contain text only",
        ),
        (ModelResponse(text="", finish_reason="stop"), "produced no text"),
        (ModelResponse(text="   \n ", finish_reason="stop"), "produced no text"),
    ],
)
def test_title_from_response_rejects(response: ModelResponse, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        title_from_response(response)


def test_title_from_response_strips_controls() -> None:
    response = ModelResponse(text="\x1b[31mHello\x1b[0m World", finish_reason="stop")
    assert title_from_response(response) == "Hello World"


# ---------------------------------------------------------------------------
# generate.
# ---------------------------------------------------------------------------


async def test_generate_requires_a_message() -> None:
    llm = _RecordingLlm()
    provider = FirstPromptTitleProvider(CONFIG, llm, "first-prompt")
    with pytest.raises(ValueError, match="requires one human message"):
        await provider.generate(_request(_session(), messages=[]))


async def test_generate_sends_only_first_message() -> None:
    llm = _RecordingLlm()
    provider = FirstPromptTitleProvider(CONFIG, llm, "first-prompt")
    session = _session()
    result = await provider.generate(
        _request(
            session,
            messages=[
                {"seq": 0, "text": "first message"},
                {"seq": 1, "text": "second message"},
            ],
            route={"provider": "prov", "model": "mod"},
        )
    )
    assert result == {
        "title": "Recorded Title",
        "messageSeqs": [0],
        "model": {"provider": "prov", "model": "mod"},
    }
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == "prov/mod"
    assert call["max_tokens"] == 4096
    assert call["messages"][0]["content"][0]["text"].endswith(
        '[{"seq":0,"text":"first message"}]'
    )
    assert "second message" not in call["messages"][0]["content"][0]["text"]
    assert call["system_prompt"].startswith("Create a concise title")


async def test_generate_logs_request_event_before_dispatch() -> None:
    seen: list[bool] = []

    class _CheckingLlm(_RecordingLlm):
        async def call(self, **kwargs: Any) -> ModelResponse:
            events = [
                event
                for event in session.events
                if event.type == "session/title-llm-request"
            ]
            seen.append(len(events) == 1)
            return await super().call(**kwargs)

    session = _session()
    llm = _CheckingLlm()
    provider = FirstPromptTitleProvider(CONFIG, llm, "first-prompt")
    await provider.generate(
        _request(session, route={"provider": "prov", "model": "mod"})
    )
    assert seen == [True]
    event = next(
        event for event in session.events if event.type == "session/title-llm-request"
    )
    assert event.payload == {
        "titleProvider": "first-prompt",
        "messageSeqs": [0],
        "route": {"provider": "prov", "model": "mod"},
        "system": system_prompt(resolve_first_prompt_config(CONFIG)),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate the session title from this JSON array "
                        'of human messages:\n[{"seq":0,"text":"hi"}]',
                    }
                ],
            }
        ],
        "maxTokens": 4096,
    }


async def test_generate_uses_explicit_pair_over_session_route() -> None:
    llm = _RecordingLlm()
    provider = FirstPromptTitleProvider(
        {**CONFIG, "provider": "explicit", "model": "v1"}, llm, "first-prompt"
    )
    result = await provider.generate(
        _request(_session(), route={"provider": "prov", "model": "mod"})
    )
    assert result["model"] == {"provider": "explicit", "model": "v1"}
    assert llm.calls[0]["model"] == "explicit/v1"


async def test_generate_without_any_route_raises() -> None:
    llm = _RecordingLlm()
    provider = FirstPromptTitleProvider(CONFIG, llm, "first-prompt")
    with pytest.raises(ValueError, match="no logged request route is available"):
        await provider.generate(_request(_session(), route=None))
    assert llm.calls == []


async def test_generate_rejects_oversized_input() -> None:
    llm = _RecordingLlm()
    provider = FirstPromptTitleProvider(
        {**CONFIG, "max_input_bytes": 10}, llm, "first-prompt"
    )
    with pytest.raises(ValueError, match="exceeding maxInputBytes"):
        await provider.generate(
            _request(
                _session(),
                messages=[{"seq": 0, "text": "this message is far too long"}],
                route={"provider": "prov", "model": "mod"},
            )
        )
    assert llm.calls == []


async def test_generate_times_out() -> None:
    llm = _RecordingLlm()
    llm.gate = asyncio.Event()
    provider = FirstPromptTitleProvider(
        {**CONFIG, "timeout_ms": 50}, llm, "first-prompt"
    )
    with pytest.raises(TimeoutError, match="timed out after 50 ms"):
        await provider.generate(
            _request(_session(), route={"provider": "prov", "model": "mod"})
        )


async def test_generate_aborted_signal_raises_before_dispatch() -> None:
    llm = _RecordingLlm()
    provider = FirstPromptTitleProvider(CONFIG, llm, "first-prompt")
    signal = AbortSignal()
    signal.abort("cancelled")
    request = TitleProviderRequest(
        session=_session(),
        messages=({"seq": 0, "text": "hi"},),
        route={"provider": "prov", "model": "mod"},
        signal=signal,
    )
    with pytest.raises(AbortError):
        await provider.generate(request)
    assert llm.calls == []


async def test_generate_takes_first_non_empty_output_line() -> None:
    llm = _RecordingLlm(
        ModelResponse(text="\n\n**Quantum Basics**\nignored", finish_reason="stop")
    )
    provider = FirstPromptTitleProvider(CONFIG, llm, "first-prompt")
    result = await provider.generate(
        _request(_session(), route={"provider": "prov", "model": "mod"})
    )
    assert result["title"] == "Quantum Basics"


def test_provider_automatic_mode_is_first_prompt() -> None:
    llm = _RecordingLlm()
    provider = FirstPromptTitleProvider(CONFIG, llm, "first-prompt")
    assert provider.automatic == "first-prompt"
    assert provider.id == "first-prompt"
