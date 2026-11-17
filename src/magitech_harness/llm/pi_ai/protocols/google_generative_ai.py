"""Google Generative AI streaming protocol client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import httpx

from magitech_harness.agent.cancellation import AbortError
from magitech_harness.llm.pi_ai.catalog import ModelSpec, resolve_reasoning_effort
from magitech_harness.llm.pi_ai.protocols.common import (
    StreamEmitter,
    build_headers,
    contains_image,
    finish_chunk,
    flatten_text,
    iter_sse_events,
    map_finish_reason,
    parse_json_data,
    raise_for_status,
    token_usage,
    usage_chunk,
)


def _model_parts(content: Any) -> list[dict[str, Any]]:
    """Convert assistant history blocks into Google model parts."""
    parts: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text", ""))
            if text:
                parts.append({"text": text})
        elif block_type == "tool-call":
            arguments = block.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    parsed = {"raw": arguments}
            else:
                parsed = arguments
            parts.append(
                {
                    "functionCall": {
                        "name": str(block.get("name", "")),
                        "args": parsed,
                    }
                }
            )
    return parts or [{"text": ""}]


def _convert_contents(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert harness history into Google conversation contents."""
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        if role == "system":
            continue
        if role == "developer":
            text = flatten_text(content)
            if text:
                contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "user":
            if contains_image(content):
                raise RuntimeError("image input unsupported")
            contents.append(
                {"role": "user", "parts": [{"text": flatten_text(content)}]}
            )
        elif role == "assistant":
            contents.append({"role": "model", "parts": _model_parts(content)})
        elif role == "tool":
            tool_call_id = str(
                message.get("toolCallId") or message.get("tool_call_id") or ""
            )
            name = str(
                message.get("toolName") or message.get("tool_name") or tool_call_id
            )
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {
                                    "result": flatten_text(content) or "(no output)"
                                },
                            }
                        }
                    ],
                }
            )
    return contents


def _system_text(
    messages: Sequence[Mapping[str, Any]], system: str | None
) -> str | None:
    """Join the request-level and history system prompts."""
    parts: list[str] = []
    if system:
        parts.append(system)
    parts.extend(
        text
        for text in (
            flatten_text(message.get("content", []))
            for message in messages
            if message.get("role") == "system"
        )
        if text
    )
    joined = "\n\n".join(parts)
    return joined or None


def _convert_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert harness tool schemas into Google function declarations."""
    if not tools:
        return []
    declarations = [
        {
            "name": str(tool.get("name", "")),
            "description": str(tool.get("description", "")),
            "parameters": tool.get("parameters")
            or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]
    return [{"functionDeclarations": declarations}]


def _build_body(spec: ModelSpec, options: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Google Generative AI request body."""
    messages = cast(Sequence[Mapping[str, Any]], options.get("messages") or [])
    body: dict[str, Any] = {"contents": _convert_contents(messages)}
    system = _system_text(messages, cast(str | None, options.get("system")))
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    tools = _convert_tools(
        cast(Sequence[Mapping[str, Any]] | None, options.get("tools"))
    )
    if tools:
        body["tools"] = tools
    generation_config: dict[str, Any] = {}
    if options.get("temperature") is not None:
        generation_config["temperature"] = options["temperature"]
    if options.get("maxTokens") is not None:
        generation_config["maxOutputTokens"] = options["maxTokens"]
    reasoning = resolve_reasoning_effort(
        spec, cast(str | None, options.get("reasoningEffort"))
    )
    if reasoning and reasoning != "off":
        generation_config["thinkingConfig"] = {"thinkingBudget": 4096}
    if generation_config:
        body["generationConfig"] = generation_config
    return body


def _chunks_of(
    payload: Mapping[str, Any],
    emitter: StreamEmitter,
) -> tuple[list[dict[str, Any]], str | None]:
    """Translate one Google stream chunk into harness chunks."""
    chunks: list[dict[str, Any]] = []
    candidates = payload.get("candidates")
    finish_reason: str | None = None
    if isinstance(candidates, list) and candidates:
        candidate = cast(Mapping[str, Any], candidates[0] or {})
        content = cast(Mapping[str, Any], candidate.get("content") or {})
        parts = content.get("parts")
        if isinstance(parts, list):
            for index, part in enumerate(parts):
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    if part.get("thought") is True:
                        chunks.extend(emitter.reasoning(text))
                    else:
                        chunks.extend(emitter.text(text))
                elif "functionCall" in part:
                    function_call = cast(
                        Mapping[str, Any], part.get("functionCall") or {}
                    )
                    arguments = json.dumps(function_call.get("args", {}))
                    chunks.extend(
                        emitter.tool_call(
                            str(index),
                            "",
                            str(function_call.get("name") or ""),
                            arguments,
                        )
                    )
        raw_finish = candidate.get("finishReason")
        if raw_finish is not None:
            finish_reason = str(raw_finish)
    return chunks, finish_reason


async def stream_google_generative_ai(
    spec: ModelSpec,
    options: Mapping[str, Any],
    *,
    api_key: str | None,
    headers: Mapping[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream harness chunks from a Google Generative AI endpoint."""
    base = spec.base_url.rstrip("/")
    url = f"{base}/models/{spec.id}:streamGenerateContent?alt=sse"
    request_headers = build_headers(
        api_key=None, extra=headers, auth_header=None, auth_scheme=""
    )
    if api_key:
        url = f"{url}&key={api_key}"
        request_headers["x-goog-api-key"] = api_key
    body = _build_body(spec, options)
    signal = options.get("signal")
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=None)
    try:
        async with http.stream(
            "POST", url, headers=request_headers, json=body
        ) as response:
            await raise_for_status(response)
            emitter = StreamEmitter()
            finish_reason: str | None = None
            usage: dict[str, int] = {}
            async for event in iter_sse_events(response):
                if signal is not None and getattr(signal, "is_aborted", False):
                    raise AbortError("Operation was aborted.")
                payload = parse_json_data(event.get("data", ""))
                if not isinstance(payload, Mapping):
                    continue
                chunks, reason = _chunks_of(payload, emitter)
                for chunk in chunks:
                    yield chunk
                if reason is not None:
                    finish_reason = reason
                metadata = payload.get("usageMetadata")
                if isinstance(metadata, Mapping):
                    usage = {
                        "input": int(metadata.get("promptTokenCount") or 0),
                        "output": int(metadata.get("candidatesTokenCount") or 0),
                        "total": int(metadata.get("totalTokenCount") or 0),
                        "cacheRead": int(metadata.get("cachedContentTokenCount") or 0),
                        "reasoning": int(metadata.get("thoughtsTokenCount") or 0),
                    }
            for chunk in emitter.finalize():
                yield chunk
            yield usage_chunk(
                token_usage(
                    input_tokens=usage.get("input", 0),
                    output_tokens=usage.get("output", 0),
                    total_tokens=usage.get("total", 0),
                    cache_read_tokens=usage.get("cacheRead", 0),
                    reasoning_tokens=usage.get("reasoning", 0),
                )
            )
            yield finish_chunk(map_finish_reason(finish_reason))
    finally:
        if owns_client:
            await http.aclose()
