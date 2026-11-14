"""First-prompt model title provider: titles from the first human message.

The provider sends only the first eligible message to the model, takes the
first non-empty output line as the title, and rejects tool-call and
max-token finishes. One ``session/title-llm-request`` event is logged for
each provider call, recording the exact prepared input and route before
dispatch, so the request is reconstructable from the log.

Execution controls are required and have no defaults here: the
composition layer supplies them. An explicit provider/model pair is
allowed only when supplied together; otherwise the session's current
logged route is used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from anima import Context, Disposer

from magitech_harness.session.title import (
    TitleProviderRequest,
    TitleProviderResult,
    clean_title_text,
)

if TYPE_CHECKING:
    from magitech_harness.llm.types import ModelResponse


class _TitleLlm(Protocol):
    """The narrow LLM seam one title call needs: a single model call."""

    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate one response for the prepared title request."""
        ...


logger = logging.getLogger("harness.session.title_first_prompt")


@dataclass(frozen=True)
class FirstPromptTitleConfig:
    """Required execution controls plus this provider's title-length targets."""

    target_words: int
    target_cjk_characters: int
    max_input_bytes: int
    max_output_tokens: int
    timeout_ms: int
    provider: str | None = None
    model: str | None = None


_CONFIG_KEYS = frozenset(
    {
        "target_words",
        "target_cjk_characters",
        "max_input_bytes",
        "max_output_tokens",
        "timeout_ms",
        "provider",
        "model",
    }
)


def _positive_first_prompt_limit(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"session-title-first-prompt: {name} must be a positive integer, "
            f"got {value!r}"
        )
    return value


def resolve_first_prompt_config(
    config: dict[str, Any] | None,
) -> FirstPromptTitleConfig:
    """Validate raw first-prompt config into immutable controls.

    Raises ``ValueError`` when the config is missing, carries an unknown
    key or a non-positive-integer limit, or supplies ``provider`` and
    ``model`` apart.
    """
    if config is None or not isinstance(config, dict):
        raise ValueError("session-title-first-prompt: configuration is required")
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "session-title-first-prompt: unknown config key "
            + ", ".join(repr(key) for key in unknown)
        )
    target_words = _positive_first_prompt_limit(config, "target_words")
    target_cjk_characters = _positive_first_prompt_limit(
        config, "target_cjk_characters"
    )
    max_input_bytes = _positive_first_prompt_limit(config, "max_input_bytes")
    max_output_tokens = _positive_first_prompt_limit(config, "max_output_tokens")
    timeout_ms = _positive_first_prompt_limit(config, "timeout_ms")
    provider = config.get("provider")
    model = config.get("model")
    if (provider is None) != (model is None):
        raise ValueError(
            "session-title-first-prompt: provider and model must be supplied together"
        )
    if provider is not None and (
        not isinstance(provider, str)
        or not provider
        or not isinstance(model, str)
        or not model
    ):
        raise ValueError(
            "session-title-first-prompt: provider and model overrides must "
            "be non-empty strings"
        )
    return FirstPromptTitleConfig(
        target_words=target_words,
        target_cjk_characters=target_cjk_characters,
        max_input_bytes=max_input_bytes,
        max_output_tokens=max_output_tokens,
        timeout_ms=timeout_ms,
        provider=provider,
        model=model,
    )


def system_prompt(config: FirstPromptTitleConfig) -> str:
    """Language-aware system instruction for a title from one message."""
    return "\n".join(
        [
            "Create a concise title for an AI coding-assistant session from the supplied human messages.",
            "Return only the title on one line, **in plain text of natural language**, with no quotes, prefix, explanation, Markdown, XML, or terminal control codes. No code is allowed.",
            "Use the language of the messages.",
            f"Aim for about {config.target_words} words in non-CJK languages or {config.target_cjk_characters} CJK characters.",
            "If the messages give little to name, still return a short best-effort title, such as Greeting, instead of explaining.",
        ]
    )


def frame_messages(messages: list[dict[str, Any]]) -> str:
    """Frame the selected messages as JSON so user text cannot break delimiters."""
    return (
        "Generate the session title from this JSON array of human messages:\n"
        + json.dumps(messages, separators=(",", ":"))
    )


_EMPHASIS_WRAPPER = re.compile(
    r"^(?P<marker>\*{1,3})(?P<inner>\S(?:.*\S)?)(?P=marker)$"
)


def title_from_output(text: str) -> str:
    """Take the title from model output: the first non-empty line.

    Whole-line asterisk emphasis is stripped; emphasis inside a longer
    line is kept. Models that disregard the one-line instruction put
    the title first and commentary after it.
    """
    line = ""
    for candidate in re.split(r"\r?\n", text):
        stripped = candidate.strip()
        if stripped:
            line = stripped
            break
    match = _EMPHASIS_WRAPPER.match(line)
    if match is not None:
        marker = match.group("marker")
        inner = match.group("inner")
        if marker not in inner:
            return inner
    return line


def title_from_response(response: ModelResponse) -> str:
    """Derive one non-empty title from a model response, or raise.

    Rejects tool-call and max-token finishes and any tool-call output;
    empty text raises.
    """
    if response.finish_reason == "tool_calls":
        raise ValueError("session-title-llm: title output must contain text only")
    if response.finish_reason == "max_tokens":
        raise ValueError("session-title-llm: title output reached maxOutputTokens")
    if response.finish_reason != "stop":
        raise ValueError(
            f"session-title-llm: unsupported title finish {response.finish_reason!r}"
        )
    if response.tool_calls:
        raise ValueError("session-title-llm: title output must contain text only")
    title = clean_title_text(title_from_output(response.text))
    if title == "":
        raise ValueError("session-title-llm: title model produced no text")
    return title


def _resolve_route(
    config: FirstPromptTitleConfig, request: TitleProviderRequest
) -> dict[str, str]:
    """Resolve the explicit pair or the exact route from the request."""
    if config.provider is not None and config.model is not None:
        return {"provider": config.provider, "model": config.model}
    if request.route is None:
        raise ValueError(
            "session-title-llm: no logged request route is available; "
            "configure provider and model together"
        )
    return dict(request.route)


class FirstPromptTitleProvider:
    """Model title provider that titles from the first human message only."""

    automatic: Literal["first-prompt", "all-prompts"] = "first-prompt"

    def __init__(
        self,
        config: dict[str, Any] | None,
        llm: _TitleLlm,
        provider_id: str,
    ) -> None:
        """Build the provider.

        ``provider_id`` is the stable id recorded with the title and the
        request event; the registering consumer supplies it. Raises
        ``ValueError`` when the config is missing or invalid, or the id
        is empty.
        """
        self._config = resolve_first_prompt_config(config)
        self._llm = llm
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError(
                "session-title-first-prompt: provider id must be a non-empty string"
            )
        self.id = provider_id

    async def generate(self, request: TitleProviderRequest) -> TitleProviderResult:
        """Title the session from its first eligible human message."""
        first = request.messages[0] if request.messages else None
        if first is None:
            raise ValueError("first-prompt title provider requires one human message")
        request.signal.throw_if_aborted()
        framed = frame_messages([first])
        input_bytes = len(framed.encode("utf-8"))
        if input_bytes > self._config.max_input_bytes:
            raise ValueError(
                "session-title-llm: input is "
                f"{input_bytes} bytes, exceeding maxInputBytes "
                f"{self._config.max_input_bytes}"
            )
        route = _resolve_route(self._config, request)
        system = system_prompt(self._config)
        messages = [{"role": "user", "content": [{"type": "text", "text": framed}]}]
        request.session.append(
            "session/title-llm-request",
            {
                "titleProvider": self.id,
                "messageSeqs": [first["seq"]],
                "route": dict(route),
                "system": system,
                "messages": messages,
                "maxTokens": self._config.max_output_tokens,
            },
        )
        request.signal.throw_if_aborted()
        try:
            response = await asyncio.wait_for(
                self._llm.call(
                    messages=messages,
                    system_prompt=system,
                    model=f"{route['provider']}/{route['model']}",
                    max_tokens=self._config.max_output_tokens,
                ),
                timeout=self._config.timeout_ms / 1000,
            )
        except TimeoutError as error:
            raise TimeoutError(
                "session-title-llm: title model request timed out after "
                f"{self._config.timeout_ms} ms"
            ) from error
        request.signal.throw_if_aborted()
        return {
            "title": title_from_response(response),
            "messageSeqs": [first["seq"]],
            "model": route,
        }


def apply(ctx: Context, config: dict[str, Any] | None = None) -> Disposer:
    """Register the first-prompt model title provider as a bundle row.

    Builds the provider over the context's model service and registers
    it with the title service under the row id, so automatic titles
    come from the model once a main request route is logged. The
    returned disposer unregisters the provider when the row unloads.
    Raises ``RuntimeError`` when the title or model service is absent,
    and ``ValueError`` when the config is missing or invalid.
    """
    title_service = ctx.get("sessionTitle", None, strict=False)
    if title_service is None:
        raise RuntimeError(
            "session-title-first-prompt requires the sessionTitle service"
        )
    llm = ctx.get("llm", None, strict=False)
    if llm is None:
        raise RuntimeError("session-title-first-prompt requires the llm service")
    provider = FirstPromptTitleProvider(config, llm, "session-title-first-prompt")
    disposer: Disposer = title_service.register(provider)
    return disposer


apply.inject = ["sessionTitle", "llm", "sessions"]  # type: ignore[attr-defined]
