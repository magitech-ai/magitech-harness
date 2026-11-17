"""OpenAI Chat Completions streaming protocol client."""

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


def _tool_calls_of(content: Any) -> list[dict[str, Any]]:
    """Convert assistant tool-call blocks into Chat Completions tool calls."""
    calls: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, Mapping) or block.get("type") != "tool-call":
            continue
        arguments = block.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append(
            {
                "id": str(block.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(block.get("name", "")),
                    "arguments": arguments,
                },
            }
        )
    return calls


def _assistant_message(content: Any) -> dict[str, Any]:
    """Convert an assistant history message into a Chat Completions message."""
    text = flatten_text(content)
    calls = _tool_calls_of(content)
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return message


def _convert_messages(
    messages: Sequence[Mapping[str, Any]], system: str | None
) -> list[dict[str, Any]]:
    """Convert harness history into Chat Completions messages."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        if role in ("system", "developer"):
            text = flatten_text(content)
            if text:
                out.append({"role": "system", "content": text})
        elif role == "user":
            if contains_image(content):
                raise RuntimeError("image input unsupported")
            out.append({"role": "user", "content": flatten_text(content)})
        elif role == "assistant":
            out.append(_assistant_message(content))
        elif role == "tool":
            tool_call_id = str(
                message.get("toolCallId") or message.get("tool_call_id") or ""
            )
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": flatten_text(content) or "(no output)",
                }
            )
    return out


def _convert_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert harness tool schemas into Chat Completions tool definitions."""
    if not tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": str(tool.get("name", "")),
                "description": str(tool.get("description", "")),
                "parameters": tool.get("parameters")
                or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def _build_body(spec: ModelSpec, options: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Chat Completions request body."""
    messages = cast(Sequence[Mapping[str, Any]], options.get("messages") or [])
    body: dict[str, Any] = {
        "model": spec.id,
        "messages": _convert_messages(
            messages, cast(str | None, options.get("system"))
        ),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if options.get("temperature") is not None:
        body["temperature"] = options["temperature"]
    if options.get("maxTokens") is not None:
        body["max_tokens"] = options["maxTokens"]
    tools = _convert_tools(
        cast(Sequence[Mapping[str, Any]] | None, options.get("tools"))
    )
    if tools:
        body["tools"] = tools
    reasoning = resolve_reasoning_effort(
        spec, cast(str | None, options.get("reasoningEffort"))
    )
    if reasoning and reasoning != "off":
        mapped = (spec.thinking_level_map or {}).get(reasoning)
        body["reasoning_effort"] = mapped if isinstance(mapped, str) else reasoning
    return body


def _usage_of(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract the usage block of a Chat Completions payload."""
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("completion_tokens_details")
    return {
        "input": int(usage.get("prompt_tokens") or 0),
        "output": int(usage.get("completion_tokens") or 0),
        "total": int(usage.get("total_tokens") or 0),
        "reasoning": int(details.get("reasoning_tokens") or 0)
        if isinstance(details, Mapping)
        else 0,
    }


def _chunks_of(
    payload: Mapping[str, Any], emitter: StreamEmitter
) -> tuple[list[dict[str, Any]], str | None]:
    """Translate one Chat Completions payload into harness chunks."""
    chunks: list[dict[str, Any]] = []
    usage = _usage_of(payload)
    if usage is not None:
        chunks.append(
            usage_chunk(
                token_usage(
                    input_tokens=usage["input"],
                    output_tokens=usage["output"],
                    total_tokens=usage["total"],
                    reasoning_tokens=usage["reasoning"],
                )
            )
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return chunks, None
    choice = cast(Mapping[str, Any], choices[0] or {})
    delta = cast(Mapping[str, Any], choice.get("delta") or {})
    content = delta.get("content")
    if isinstance(content, str) and content:
        chunks.extend(emitter.text(content))
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        chunks.extend(emitter.reasoning(reasoning))
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            key = str(tool_call.get("index") or 0)
            function = cast(Mapping[str, Any], tool_call.get("function") or {})
            chunks.extend(
                emitter.tool_call(
                    key,
                    str(tool_call.get("id") or ""),
                    str(function.get("name") or ""),
                    str(function.get("arguments") or ""),
                )
            )
    finish_reason = choice.get("finish_reason")
    return chunks, str(finish_reason) if finish_reason else None


async def stream_openai_completions(
    spec: ModelSpec,
    options: Mapping[str, Any],
    *,
    api_key: str | None,
    headers: Mapping[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream harness chunks from an OpenAI Chat Completions endpoint."""
    url = f"{spec.base_url}/chat/completions"
    request_headers = build_headers(
        api_key=api_key,
        extra=headers,
        auth_header="authorization",
        auth_scheme="Bearer",
    )
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
            for chunk in emitter.finalize():
                yield chunk
            yield finish_chunk(map_finish_reason(finish_reason))
    finally:
        if owns_client:
            await http.aclose()
