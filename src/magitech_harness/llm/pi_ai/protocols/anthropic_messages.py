"""Anthropic Messages streaming protocol client."""

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

ANTHROPIC_VERSION = "2023-06-01"

THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}


def _tool_use_blocks(content: Any) -> list[dict[str, Any]]:
    """Convert assistant tool-call blocks into Anthropic tool-use blocks."""
    blocks: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, Mapping) or block.get("type") != "tool-call":
            continue
        arguments = block.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed = {"raw": arguments}
        else:
            parsed = arguments
        blocks.append(
            {
                "type": "tool_use",
                "id": str(block.get("id", "")),
                "name": str(block.get("name", "")),
                "input": parsed,
            }
        )
    return blocks


def _assistant_blocks(content: Any) -> list[dict[str, Any]]:
    """Convert assistant history blocks into Anthropic content blocks."""
    blocks: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text", ""))
            if text:
                blocks.append({"type": "text", "text": text})
        elif block_type == "reasoning":
            # Anthropic replays thinking only with its signature, which this
            # adapter does not carry; the reasoning text travels as plain text.
            text = str(block.get("text", ""))
            if text:
                blocks.append({"type": "text", "text": text})
    blocks.extend(_tool_use_blocks(content))
    return blocks or [{"type": "text", "text": ""}]


def _convert_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Convert harness history into Anthropic system text and messages."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        if role in ("system", "developer"):
            text = flatten_text(content)
            if text:
                system_parts.append(text)
        elif role == "user":
            if contains_image(content):
                raise RuntimeError("image input unsupported")
            out.append({"role": "user", "content": flatten_text(content)})
        elif role == "assistant":
            out.append({"role": "assistant", "content": _assistant_blocks(content)})
        elif role == "tool":
            tool_call_id = str(
                message.get("toolCallId") or message.get("tool_call_id") or ""
            )
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": flatten_text(content) or "(no output)",
            }
            if message.get("isError"):
                result["is_error"] = True
            out.append({"role": "user", "content": [result]})
    return system_parts, out


def _convert_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert harness tool schemas into Anthropic tool definitions."""
    if not tools:
        return []
    return [
        {
            "name": str(tool.get("name", "")),
            "description": str(tool.get("description", "")),
            "input_schema": tool.get("parameters")
            or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]


def _build_body(spec: ModelSpec, options: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Anthropic Messages request body."""
    messages = cast(Sequence[Mapping[str, Any]], options.get("messages") or [])
    system_parts, converted = _convert_messages(messages)
    extra_system = cast(str | None, options.get("system"))
    if extra_system:
        system_parts.insert(0, extra_system)
    body: dict[str, Any] = {
        "model": spec.id,
        "messages": converted,
        "max_tokens": int(options.get("maxTokens") or spec.max_tokens or 4096),
        "stream": True,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if options.get("temperature") is not None:
        body["temperature"] = options["temperature"]
    tools = _convert_tools(
        cast(Sequence[Mapping[str, Any]] | None, options.get("tools"))
    )
    if tools:
        body["tools"] = tools
    reasoning = resolve_reasoning_effort(
        spec, cast(str | None, options.get("reasoningEffort"))
    )
    if reasoning and reasoning != "off":
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": THINKING_BUDGETS.get(reasoning, 4096),
        }
    return body


def _chunks_of(
    payload: Mapping[str, Any],
    event: Mapping[str, str],
    emitter: StreamEmitter,
) -> tuple[list[dict[str, Any]], str | None]:
    """Translate one Anthropic event into harness chunks."""
    chunks: list[dict[str, Any]] = []
    event_type = str(payload.get("type") or event.get("event") or "")
    if event_type == "content_block_start":
        block = cast(Mapping[str, Any], payload.get("content_block") or {})
        block_type = str(block.get("type") or "")
        key = str(payload.get("index") or 0)
        if block_type == "text":
            chunks.extend(emitter.text(str(block.get("text") or "")))
        elif block_type == "thinking":
            chunks.extend(emitter.reasoning(str(block.get("thinking") or "")))
        elif block_type == "tool_use":
            chunks.extend(
                emitter.tool_call(
                    key, str(block.get("id") or ""), str(block.get("name") or ""), ""
                )
            )
    elif event_type == "content_block_delta":
        delta = cast(Mapping[str, Any], payload.get("delta") or {})
        delta_type = str(delta.get("type") or "")
        key = str(payload.get("index") or 0)
        if delta_type == "text_delta":
            chunks.extend(emitter.text(str(delta.get("text") or "")))
        elif delta_type == "thinking_delta":
            chunks.extend(emitter.reasoning(str(delta.get("thinking") or "")))
        elif delta_type == "input_json_delta":
            chunks.extend(
                emitter.tool_call(key, "", "", str(delta.get("partial_json") or ""))
            )
    elif event_type == "message_delta":
        delta = cast(Mapping[str, Any], payload.get("delta") or {})
        stop_reason = delta.get("stop_reason")
        return chunks, str(stop_reason) if stop_reason else None
    return chunks, None


async def stream_anthropic_messages(
    spec: ModelSpec,
    options: Mapping[str, Any],
    *,
    api_key: str | None,
    headers: Mapping[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream harness chunks from an Anthropic Messages endpoint."""
    base = spec.base_url.rstrip("/")
    url = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
    request_headers = build_headers(
        api_key=api_key, extra=headers, auth_header="x-api-key", auth_scheme=""
    )
    request_headers["anthropic-version"] = ANTHROPIC_VERSION
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
            input_tokens = 0
            output_tokens = 0
            cache_read = 0
            cache_write = 0
            async for event in iter_sse_events(response):
                if signal is not None and getattr(signal, "is_aborted", False):
                    raise AbortError("Operation was aborted.")
                payload = parse_json_data(event.get("data", ""))
                if not isinstance(payload, Mapping):
                    continue
                event_type = str(payload.get("type") or event.get("event") or "")
                if event_type == "message_start":
                    message = cast(Mapping[str, Any], payload.get("message") or {})
                    usage = cast(Mapping[str, Any], message.get("usage") or {})
                    input_tokens = int(usage.get("input_tokens") or 0)
                    cache_read = int(usage.get("cache_read_input_tokens") or 0)
                    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
                elif event_type == "message_delta":
                    usage = cast(Mapping[str, Any], payload.get("usage") or {})
                    if usage.get("output_tokens") is not None:
                        output_tokens = int(usage.get("output_tokens") or 0)
                chunks, reason = _chunks_of(payload, event, emitter)
                for chunk in chunks:
                    yield chunk
                if reason is not None:
                    finish_reason = reason
            for chunk in emitter.finalize():
                yield chunk
            yield usage_chunk(
                token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                )
            )
            yield finish_chunk(map_finish_reason(finish_reason))
    finally:
        if owns_client:
            await http.aclose()
