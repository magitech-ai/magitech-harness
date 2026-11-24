"""Token estimation, replay-aware metering projections, and route-aware pricing."""

from __future__ import annotations

import copy
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, cast
from weakref import WeakKeyDictionary

from animas import Context, Service

from magitech_harness.llm.assistant_stream import (
    assemble_assistant_stream,
    last_assistant_stream_chunk,
)
from magitech_harness.llm.values import (
    deep_freeze,
    is_non_negative_safe_integer,
    is_safe_integer,
)
from magitech_harness.session.events import SURFACE_EVENT_TYPES, SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    ProjectionHandle,
    SessionProjectionRegistry,
    WireView,
)
from magitech_harness.session.request_header import canonical_header, header_equals
from magitech_harness.session.store import Session
from magitech_harness.session.surface import derive_event_message

CHARS_PER_TOKEN = 4
BLOCK_OVERHEAD = 4
ROLE_OVERHEAD = 4

_MEASUREMENT_KEYS = frozenset({"headerSeq", "stepStart", "anchor", "invalidSeq"})
_STEP_START_KEYS = frozenset({"turn", "step"})
_ANCHOR_KEYS = frozenset({"headerSeq", "assistantTokens", "usage"})
_USAGE_BUCKET_KEYS = frozenset(
    {
        "uncachedInputTokens",
        "outputTokens",
        "cacheReadTokens",
        "cacheWriteTokens",
    }
)
_USAGE_STATE_KEYS = frozenset({"totals", "last"})
_LAST_SLOT_KEYS = frozenset({"turn", "step", "buckets"})
_PRESSURE_STATE_KEYS = frozenset(
    {
        "contextWindow",
        "pressureTokens",
        "surfaceTokens",
        "sampledSurfaceTokens",
        "claim",
    }
)
_PRESSURE_VIEW_KEYS = frozenset({"pressureTokens", "projectedTokens", "contextWindow"})
_CLAIM_KEYS = frozenset({"start", "end", "tokens"})
_BREAKDOWN_CLAIM_KEYS = frozenset({"start", "end", "tokens", "systemSeqs"})
_BREAKDOWN_KEYS = frozenset({"systemTokens", "toolsTokens", "messageTokens"})
_SYSTEM_NODE_KEYS = frozenset({"seq", "tokens"})


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _ceil_tokens(char_count: int) -> int:
    return (char_count + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_positive_safe_int(value: Any) -> bool:
    return is_non_negative_safe_integer(value) and value > 0


def _require_exact_mapping(
    value: Any,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str = "state",
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")  # noqa: TRY004
    keys = set(value.keys())
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise ValueError(f"{label} has invalid keys")
    return value


def _is_valid_usage_record(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_finite_number(value.get("inputTokens"))
        and _is_finite_number(value.get("outputTokens"))
        and all(
            _is_finite_number(value[k])
            for k in (
                "totalTokens",
                "cacheReadTokens",
                "cacheWriteTokens",
                "reasoningTokens",
            )
            if k in value
        )
    )


def _validate_measurement_state(value: Any) -> dict[str, Any]:
    mapping = _require_exact_mapping(
        value, _MEASUREMENT_KEYS, label="tokenMeasurement state"
    )
    for seq_key in ("headerSeq", "invalidSeq"):
        seq_val = mapping[seq_key]
        if seq_val is not None and not is_non_negative_safe_integer(seq_val):
            raise ValueError(f"tokenMeasurement.{seq_key} must be a seq or None")
    step_start = mapping["stepStart"]
    if step_start is not None:
        ss = _require_exact_mapping(
            step_start, _STEP_START_KEYS, label="tokenMeasurement.stepStart"
        )
        if not _is_finite_number(ss["turn"]) or not _is_finite_number(ss["step"]):
            raise ValueError("tokenMeasurement.stepStart turn and step must be numbers")
    anchor = mapping["anchor"]
    if anchor is not None:
        anch = _require_exact_mapping(
            anchor, _ANCHOR_KEYS, label="tokenMeasurement.anchor"
        )
        if (
            (
                anch["headerSeq"] is not None
                and not is_non_negative_safe_integer(anch["headerSeq"])
            )
            or not _is_finite_number(anch["assistantTokens"])
            or (anch["usage"] is not None and not _is_valid_usage_record(anch["usage"]))
        ):
            raise ValueError("tokenMeasurement.anchor is invalid")
    return dict(mapping)


def _validate_usage_buckets(value: Any, label: str = "tokenUsage") -> dict[str, int]:
    mapping = _require_exact_mapping(value, _USAGE_BUCKET_KEYS, label=label)
    if not all(is_non_negative_safe_integer(mapping[k]) for k in _USAGE_BUCKET_KEYS):
        raise ValueError(f"{label} buckets must be non-negative integers")
    return {k: int(mapping[k]) for k in _USAGE_BUCKET_KEYS}


def _validate_token_usage_state(value: Any) -> dict[str, Any]:
    mapping = _require_exact_mapping(value, _USAGE_STATE_KEYS, label="tokenUsage state")
    totals = _validate_usage_buckets(mapping["totals"], "tokenUsage.totals")
    last = mapping["last"]
    if last is not None:
        last_map = _require_exact_mapping(
            last, _LAST_SLOT_KEYS, label="tokenUsage.last"
        )
        if not is_non_negative_safe_integer(
            last_map["turn"]
        ) or not is_non_negative_safe_integer(last_map["step"]):
            raise ValueError("tokenUsage.last turn/step must be non-negative int")
        buckets = _validate_usage_buckets(
            last_map["buckets"], "tokenUsage.last.buckets"
        )
        last_validated: dict[str, Any] | None = {
            "turn": last_map["turn"],
            "step": last_map["step"],
            "buckets": buckets,
        }
    else:
        last_validated = None
    return {"totals": totals, "last": last_validated}


def _validate_pressure_state(value: Any) -> dict[str, Any]:
    mapping = _require_exact_mapping(
        value,
        frozenset({"surfaceTokens"}),
        _PRESSURE_STATE_KEYS - frozenset({"surfaceTokens"}),
        label="contextPressure state",
    )
    claim = mapping.get("claim")
    valid_claim = claim is None or (
        isinstance(claim, Mapping)
        and set(claim.keys()) == _CLAIM_KEYS
        and all(is_non_negative_safe_integer(claim[k]) for k in _CLAIM_KEYS)
    )
    if (
        not is_non_negative_safe_integer(mapping["surfaceTokens"])
        or (
            "contextWindow" in mapping
            and not _is_positive_safe_int(mapping["contextWindow"])
        )
        or any(
            k in mapping and not is_non_negative_safe_integer(mapping[k])
            for k in ("pressureTokens", "sampledSurfaceTokens")
        )
        or not valid_claim
    ):
        raise ValueError("contextPressure state is invalid")
    return dict(mapping)


def _validate_pressure_view(value: Any) -> dict[str, int]:
    mapping = _require_exact_mapping(
        value, frozenset(), _PRESSURE_VIEW_KEYS, label="contextPressure view"
    )
    if (
        "contextWindow" in mapping
        and not _is_positive_safe_int(mapping["contextWindow"])
    ) or any(
        k in mapping and not is_non_negative_safe_integer(mapping[k])
        for k in ("pressureTokens", "projectedTokens")
    ):
        raise ValueError("contextPressure view is invalid")
    return dict(mapping)


def _validate_breakdown_view(value: Any) -> dict[str, int]:
    mapping = _require_exact_mapping(
        value, _BREAKDOWN_KEYS, label="contextBreakdown view"
    )
    if not all(is_non_negative_safe_integer(mapping[k]) for k in _BREAKDOWN_KEYS):
        raise ValueError("contextBreakdown view is invalid")
    return dict(mapping)


def _validate_breakdown_state(value: Any) -> dict[str, Any]:
    mapping = _require_exact_mapping(
        value,
        frozenset({"systems", "surfaceTokens", "breakdown"}),
        frozenset({"claim"}),
        label="contextBreakdown state",
    )
    _validate_breakdown_view(mapping["breakdown"])
    systems = mapping["systems"]
    claim = mapping.get("claim")
    valid_systems = (
        isinstance(systems, Sequence)
        and not isinstance(systems, (str, bytes))
        and all(
            isinstance(s, Mapping)
            and set(s.keys()) == _SYSTEM_NODE_KEYS
            and is_non_negative_safe_integer(s["seq"])
            and is_non_negative_safe_integer(s["tokens"])
            for s in systems
        )
    )
    valid_claim = claim is None or (
        isinstance(claim, Mapping)
        and set(claim.keys()) == _BREAKDOWN_CLAIM_KEYS
        and all(is_non_negative_safe_integer(claim[k]) for k in _CLAIM_KEYS)
        and isinstance(claim["systemSeqs"], Sequence)
        and not isinstance(claim["systemSeqs"], (str, bytes))
        and all(is_non_negative_safe_integer(sq) for sq in claim["systemSeqs"])
    )
    if (
        not is_non_negative_safe_integer(mapping["surfaceTokens"])
        or not valid_systems
        or not valid_claim
    ):
        raise ValueError("contextBreakdown state is invalid")
    return dict(mapping)


def estimate_structural_block(block: Mapping[str, Any]) -> int:
    """Estimate token count for a structural content block."""
    payload = (
        {k: v for k, v in block.items() if k != "offloaded"}
        if block.get("type") == "image" and "offloaded" in block
        else block
    )
    return BLOCK_OVERHEAD + _ceil_tokens(_utf16_len(_compact_json(payload)))


estimateStructuralBlock = estimate_structural_block


def estimate_content(blocks: Sequence[Mapping[str, Any]]) -> int:
    """Estimate token count for a sequence of content blocks."""
    total = 0
    for block in blocks:
        block_type = block.get("type")
        if block_type in ("text", "reasoning"):
            total += _ceil_tokens(_utf16_len(block["text"])) + BLOCK_OVERHEAD
        elif block_type == "tool-call":
            total += (
                _ceil_tokens(_utf16_len(block["name"]))
                + _ceil_tokens(_utf16_len(block["arguments"]))
                + BLOCK_OVERHEAD
            )
        else:
            total += estimate_structural_block(block)
    return total


estimateContent = estimate_content


def estimate_system_message(message: Mapping[str, Any]) -> int:
    """Estimate token count for a system message."""
    content = message["content"]
    if len(content) == 0:
        return 0
    char_count = 0
    for block in content:
        if block.get("type") == "text":
            char_count += _utf16_len(block["text"])
        else:
            char_count += _utf16_len(_compact_json(block))
    return _ceil_tokens(char_count) + ROLE_OVERHEAD


estimateSystemMessage = estimate_system_message


def estimate_message(message: Mapping[str, Any]) -> int:
    """Estimate token count for a single message."""
    if message.get("role") == "system":
        return estimate_system_message(message)
    return estimate_content(message["content"]) + ROLE_OVERHEAD


estimateMessage = estimate_message


def estimate_tools_tokens(header: Mapping[str, Any] | None) -> int:
    """Estimate token count for serialized tool definitions on a request header."""
    if header is None:
        return 0
    tools = header.get("tools")
    if not tools:
        return 0
    return _ceil_tokens(_utf16_len(_compact_json(tools))) + BLOCK_OVERHEAD


estimateToolsTokens = estimate_tools_tokens


def _collect_projected_attachments(
    blocks: Sequence[Mapping[str, Any]],
    images: list[dict[str, Any]],
    files: list[dict[str, Any]],
) -> tuple[int, int]:
    image_tokens = 0
    file_tokens = 0
    for block in blocks:
        block_type = block.get("type")
        if block_type == "image":
            images.append(dict(block))
            image_tokens += estimate_structural_block(block)
        elif block_type == "file":
            files.append(dict(block["attachment"]))
            file_tokens += estimate_structural_block(block)
    return image_tokens, file_tokens


def _analyze_node(seq: int, message: Mapping[str, Any] | None) -> dict[str, Any]:
    if message is None:
        return {
            "seq": seq,
            "heuristicTokens": 0,
            "imageStructuralTokens": 0,
            "fileStructuralTokens": 0,
            "images": [],
            "files": [],
        }
    heuristic_tokens = estimate_message(message)
    images: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    image_tokens, file_tokens = _collect_projected_attachments(
        message["content"], images, files
    )
    return {
        "seq": seq,
        "heuristicTokens": heuristic_tokens,
        "imageStructuralTokens": image_tokens,
        "fileStructuralTokens": file_tokens,
        "images": images,
        "files": files,
    }


def plan_surface_tokens(
    nodes: Sequence[Mapping[str, Any]],
    event: SessionEvent,
) -> dict[str, Any]:
    """Validate and price one surface event without mutating ``nodes``."""
    node = _analyze_node(event.seq, derive_event_message(event))
    tokens = node["heuristicTokens"]
    op = event.surface_op
    if op == "append":
        return {
            "tokens": tokens,
            "deltaTokens": tokens,
            "node": node,
            "target": "append",
        }
    assert isinstance(op, Mapping)
    start_seq = op["startSeq"]
    end_seq = op["endSeq"]
    start_idx = next(
        (i for i, candidate in enumerate(nodes) if candidate["seq"] == start_seq), -1
    )
    end_idx = next(
        (i for i, candidate in enumerate(nodes) if candidate["seq"] == end_seq), -1
    )
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        raise ValueError(
            f"token surface: replace at seq {event.seq} has invalid current range "
            f"{start_seq}-{end_seq}"
        )
    removed = sum(
        int(candidate["heuristicTokens"])
        for candidate in nodes[start_idx : end_idx + 1]
    )
    return {
        "tokens": tokens,
        "deltaTokens": tokens - removed,
        "node": node,
        "target": {"startIdx": start_idx, "endIdx": end_idx},
    }


planSurfaceTokens = plan_surface_tokens


def commit_surface_tokens(
    nodes: list[dict[str, Any]],
    plan: Mapping[str, Any],
) -> None:
    """Apply one validated surface token plan in place."""
    target = plan["target"]
    if target == "append":
        nodes.append(plan["node"])
        return
    start_idx = target["startIdx"]
    end_idx = target["endIdx"]
    nodes[start_idx : end_idx + 1] = [plan["node"]]


commitSurfaceTokens = commit_surface_tokens


def _resolve_price_images(
    pricing: Any,
) -> Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None:
    if pricing is None:
        return None
    fn = (
        pricing.get("priceImages") or pricing.get("price_images")
        if isinstance(pricing, Mapping)
        else getattr(pricing, "priceImages", None)
        or getattr(pricing, "price_images", None)
    )
    return fn if callable(fn) else None


def price_surface(
    nodes: Sequence[Mapping[str, Any]],
    pricing: Any = None,
    file_text: Callable[[Mapping[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Price one ordered surface under its model-request attachment projection."""
    price_images = _resolve_price_images(pricing)
    images: list[Mapping[str, Any]] = (
        []
        if price_images is None
        else [img for node in nodes for img in node["images"]]
    )
    has_files = file_text is not None and any(len(node["files"]) > 0 for node in nodes)
    if (price_images is None or len(images) == 0) and not has_files:
        surface_tokens = 0
        public_nodes: list[dict[str, int]] = []
        for node in nodes:
            heuristic = int(node["heuristicTokens"])
            surface_tokens += heuristic
            public_nodes.append(
                {
                    "seq": int(node["seq"]),
                    "tokens": heuristic,
                    "heuristicTokens": heuristic,
                }
            )
        return {"nodes": public_nodes, "surfaceTokens": surface_tokens}

    prices = [] if price_images is None else list(price_images(images))
    if price_images is not None and len(prices) != len(images):
        raise ValueError(
            f"token meter: route image pricing answered {len(prices)} prices for "
            f"{len(images)} occurrences"
        )

    cursor = 0
    surface_tokens = 0
    public_nodes = []
    for node in nodes:
        tokens = int(node["heuristicTokens"])
        if file_text is not None and len(node["files"]) > 0:
            tokens -= int(node["fileStructuralTokens"])
            for file_ref in node["files"]:
                tokens += estimate_content(
                    [{"type": "text", "text": file_text(file_ref)}]
                )
        if price_images is not None and len(node["images"]) > 0:
            tokens -= int(node["imageStructuralTokens"])
            for _ in node["images"]:
                price = prices[cursor]
                cursor += 1
                tokens += int(price["visualTokens"]) + estimate_content(
                    [{"type": "text", "text": price["text"]}]
                )
        surface_tokens += tokens
        public_nodes.append(
            {
                "seq": int(node["seq"]),
                "tokens": tokens,
                "heuristicTokens": int(node["heuristicTokens"]),
            }
        )
    return {"nodes": public_nodes, "surfaceTokens": surface_tokens}


priceSurface = price_surface


def fold_surface_projection(
    claim: Mapping[str, Any] | None,
    event: SessionEvent,
) -> dict[str, Any]:
    """Fold one committed event onto a running surface-token total."""
    if event.type in ("compaction/summary", "compaction/prune"):
        shadowed_range = event.payload["shadowedRange"]
        shadowed_tokens = int(event.payload["shadowedTokenCount"])
        return {
            "deltaTokens": 0,
            "claim": {
                "start": int(shadowed_range["start"]),
                "end": int(shadowed_range["end"]),
                "tokens": shadowed_tokens,
            },
        }
    if event.type not in SURFACE_EVENT_TYPES:
        return {"deltaTokens": 0, "claim": None}
    message = derive_event_message(event)
    tokens = 0 if message is None else estimate_message(message)
    op = event.surface_op
    if op == "append":
        return {"deltaTokens": tokens, "claim": None}
    if claim is None:
        return {"deltaTokens": 0, "claim": None}
    assert isinstance(op, Mapping)
    if claim["start"] != op["startSeq"] or claim["end"] != op["endSeq"]:
        raise ValueError(
            f"token surface: replace at seq {event.seq} over range "
            f"{op['startSeq']}-{op['endSeq']} has no adjacent shadow price "
            f"(armed claim covers {claim['start']}-{claim['end']})"
        )
    return {"deltaTokens": tokens - int(claim["tokens"]), "claim": None}


foldSurfaceProjection = fold_surface_projection


def _zero_buckets() -> dict[str, int]:
    return {
        "uncachedInputTokens": 0,
        "outputTokens": 0,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
    }


def _buckets_from(usage: Mapping[str, Any]) -> dict[str, int]:
    return {
        "uncachedInputTokens": int(usage["inputTokens"]),
        "outputTokens": int(usage["outputTokens"]),
        "cacheReadTokens": int(usage.get("cacheReadTokens") or 0),
        "cacheWriteTokens": int(usage.get("cacheWriteTokens") or 0),
    }


def _add_replacing(
    totals: Mapping[str, int],
    previous: Mapping[str, int] | None,
    nxt: Mapping[str, int],
) -> dict[str, int]:
    prev = previous or _zero_buckets()
    return {
        "uncachedInputTokens": totals["uncachedInputTokens"]
        - prev["uncachedInputTokens"]
        + nxt["uncachedInputTokens"],
        "outputTokens": totals["outputTokens"]
        - prev["outputTokens"]
        + nxt["outputTokens"],
        "cacheReadTokens": totals["cacheReadTokens"]
        - prev["cacheReadTokens"]
        + nxt["cacheReadTokens"],
        "cacheWriteTokens": totals["cacheWriteTokens"]
        - prev["cacheWriteTokens"]
        + nxt["cacheWriteTokens"],
    }


def _pressure_from(usage: Mapping[str, Any]) -> int:
    return (
        int(usage["inputTokens"])
        + int(usage.get("cacheReadTokens") or 0)
        + int(usage.get("cacheWriteTokens") or 0)
    )


def _usage_of(event: SessionEvent) -> Mapping[str, Any] | None:
    if event.type == "assistant/message" and event.payload.get("usage") is not None:
        return cast(Mapping[str, Any], event.payload["usage"])
    if event.type not in ("assistant/message", "assistant/attempt"):
        return None
    chunk = last_assistant_stream_chunk(event.payload["stream"], "usage")
    return None if chunk is None else cast(Mapping[str, Any], chunk["usage"])


def advance_measurement(
    state: Mapping[str, Any],
    event: SessionEvent,
) -> dict[str, Any]:
    """Advance one measurement prefix without mutating its previous state."""
    if event.type == "request/header":
        canonical_header(event.payload["header"])
        return {**state, "headerSeq": event.seq}
    if event.type == "step/start":
        step_start = state["stepStart"]
        if step_start is not None:
            raise ValueError(
                f"token meter: step/start at seq {event.seq} arrived before "
                f"turn {step_start['turn']}/step {step_start['step']} ended"
            )
        return {
            **state,
            "stepStart": {
                "turn": event.payload["turn"],
                "step": event.payload["step"],
            },
        }
    if event.type == "step/end":
        step_start = state["stepStart"]
        if (
            step_start is None
            or step_start["turn"] != event.payload["turn"]
            or step_start["step"] != event.payload["step"]
        ):
            raise ValueError(
                f"token meter: step/end at seq {event.seq} has no matching "
                "step/start event"
            )
        return {**state, "stepStart": None}
    if event.type == "assistant/message":
        step_start = state["stepStart"]
        if (
            step_start is None
            or step_start["turn"] != event.payload["turn"]
            or step_start["step"] != event.payload["step"]
        ):
            raise ValueError(
                f"token meter: assistant/message at seq {event.seq} has no matching "
                "step/start event"
            )
        raw_usage = event.payload.get("usage")
        if raw_usage is not None and state["headerSeq"] is not None:
            content = assemble_assistant_stream(event.payload["stream"]).blocks()
            assistant_tokens = (
                0 if len(content) == 0 else estimate_content(content) + ROLE_OVERHEAD
            )
            usage: Any = raw_usage
        else:
            message = derive_event_message(event)
            assistant_tokens = 0 if message is None else estimate_message(message)
            usage = None
        return {
            **state,
            "anchor": {
                "headerSeq": state["headerSeq"],
                "assistantTokens": assistant_tokens,
                "usage": usage,
            },
        }
    return dict(state)


advanceMeasurement = advance_measurement


def _init_measurement(
    _header: Mapping[str, Any] | None = None,
    _inherited: int = 0,
) -> dict[str, Any]:
    return {
        "headerSeq": None,
        "stepStart": None,
        "anchor": None,
        "invalidSeq": None,
    }


def _apply_measurement(state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
    if state["invalidSeq"] is not None:
        return state
    if event.type not in (
        "request/header",
        "step/start",
        "step/end",
        "assistant/message",
    ):
        return state
    try:
        return advance_measurement(state, event)
    except Exception:  # noqa: BLE001
        return {**state, "invalidSeq": event.seq}


measurement_projection_definition: ProjectionDefinition[dict[str, Any]] = (
    ProjectionDefinition(
        key="tokenMeasurement",
        state_version=2,
        state_schema=_validate_measurement_state,
        init=_init_measurement,
        apply=_apply_measurement,
        wire=None,
    )
)

measurementProjectionDefinition = measurement_projection_definition


def _init_token_usage(
    _header: Mapping[str, Any] | None = None,
    _inherited: int = 0,
) -> dict[str, Any]:
    return {"totals": _zero_buckets(), "last": None}


def _apply_token_usage(state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
    if event.type == "llm/retry-started":
        last = state["last"]
        if (
            last is not None
            and last["turn"] == event.payload["turn"]
            and last["step"] == event.payload["step"]
        ):
            return {**state, "last": None}
        return state
    if event.type not in ("assistant/message", "assistant/attempt"):
        return state
    sample = _usage_of(event)
    if sample is None:
        return state
    turn = event.payload["turn"]
    step = event.payload["step"]
    buckets = _buckets_from(sample)
    last = state["last"]
    previous = (
        last["buckets"]
        if last is not None and last["turn"] == turn and last["step"] == step
        else None
    )
    if previous is not None and previous == buckets:
        return state
    return {
        "totals": _add_replacing(state["totals"], previous, buckets),
        "last": {"turn": turn, "step": step, "buckets": buckets},
    }


token_usage_projection_definition: ProjectionDefinition[dict[str, Any]] = (
    ProjectionDefinition(
        key="tokenUsage",
        state_version=2,
        state_schema=_validate_token_usage_state,
        init=_init_token_usage,
        apply=_apply_token_usage,
        wire=WireView(
            view_schema=lambda v: _validate_usage_buckets(v, "tokenUsage view"),
            view=lambda s: s["totals"],
            summary=True,
        ),
    )
)

tokenUsageProjectionDefinition = token_usage_projection_definition


def _init_context_pressure(
    _header: Mapping[str, Any] | None = None,
    _inherited: int = 0,
) -> dict[str, Any]:
    return {"surfaceTokens": 0}


def _apply_context_pressure(
    state: dict[str, Any], event: SessionEvent
) -> dict[str, Any]:
    fold = fold_surface_projection(state.get("claim"), event)
    nxt = state
    if event.type == "request/context":
        context_window = event.payload.get("contextWindow")
        if context_window != state.get("contextWindow"):
            if context_window is not None:
                nxt = {**nxt, "contextWindow": context_window}
            else:
                nxt = {k: v for k, v in nxt.items() if k != "contextWindow"}
    usage = _usage_of(event)
    if usage is not None:
        pressure_tokens = _pressure_from(usage)
        if (
            pressure_tokens != nxt.get("pressureTokens")
            or nxt.get("sampledSurfaceTokens") != nxt["surfaceTokens"]
        ):
            nxt = {
                **nxt,
                "pressureTokens": pressure_tokens,
                "sampledSurfaceTokens": nxt["surfaceTokens"],
            }
    if fold["deltaTokens"] != 0:
        nxt = {**nxt, "surfaceTokens": nxt["surfaceTokens"] + fold["deltaTokens"]}
    if "claim" not in state and fold["claim"] is None:
        return nxt
    without_claim = {k: v for k, v in nxt.items() if k != "claim"}
    return (
        without_claim
        if fold["claim"] is None
        else {**without_claim, "claim": fold["claim"]}
    )


def _wire_context_pressure(state: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    if "contextWindow" in state:
        out["contextWindow"] = state["contextWindow"]
    if "pressureTokens" in state:
        out["pressureTokens"] = state["pressureTokens"]
        if "sampledSurfaceTokens" in state:
            out["projectedTokens"] = max(
                0,
                state["pressureTokens"]
                + state["surfaceTokens"]
                - state["sampledSurfaceTokens"],
            )
    return out


context_pressure_projection_definition: ProjectionDefinition[dict[str, Any]] = (
    ProjectionDefinition(
        key="contextPressure",
        state_version=5,
        state_schema=_validate_pressure_state,
        init=_init_context_pressure,
        apply=_apply_context_pressure,
        wire=WireView(
            view_schema=_validate_pressure_view,
            view=_wire_context_pressure,
            summary=False,
        ),
    )
)

contextPressureProjectionDefinition = context_pressure_projection_definition


def _init_context_breakdown(
    _header: Mapping[str, Any] | None = None,
    _inherited: int = 0,
) -> dict[str, Any]:
    return {
        "systems": [],
        "surfaceTokens": 0,
        "breakdown": {"systemTokens": 0, "toolsTokens": 0, "messageTokens": 0},
    }


def _place_surface_event(
    state: dict[str, Any],
    claim: Mapping[str, Any] | None,
    event: SessionEvent,
    tokens: int,
) -> tuple[list[dict[str, int]], int]:
    op = event.surface_op
    node = {"seq": event.seq, "tokens": tokens}
    is_system = event.type == "system/message"
    systems: list[dict[str, int]] = state["systems"]
    surface_tokens: int = state["surfaceTokens"]
    if op == "append":
        return (
            [*systems, node] if is_system else systems,
            surface_tokens + tokens,
        )
    assert isinstance(op, Mapping)
    start_seq = op["startSeq"]
    end_seq = op["endSeq"]
    if claim is not None:
        if claim["start"] != start_seq or claim["end"] != end_seq:
            raise ValueError(
                f"context breakdown: replace at seq {event.seq} over range "
                f"{start_seq}-{end_seq} has no adjacent shadow price "
                f"(armed claim covers {claim['start']}-{claim['end']})"
            )
        if is_system:
            raise ValueError(
                f"context breakdown: replace at seq {event.seq} puts a system message "
                "in place of a metered range; its order among the surviving system "
                "prompts is unknown"
            )
        shadowed = set(claim["systemSeqs"])
        return (
            [sys_node for sys_node in systems if sys_node["seq"] not in shadowed],
            surface_tokens + tokens - int(claim["tokens"]),
        )
    index = (
        next((i for i, s in enumerate(systems) if s["seq"] == start_seq), -1)
        if start_seq == end_seq
        else -1
    )
    if index == -1:
        return systems, surface_tokens
    replaced = systems[index]
    if is_system:
        updated = [*systems[:index], node, *systems[index + 1 :]]
    else:
        updated = [*systems[:index], *systems[index + 1 :]]
    return updated, surface_tokens + tokens - int(replaced["tokens"])


def _with_breakdown_totals(
    state: dict[str, Any],
    systems: list[dict[str, int]],
    raw_surface_tokens: int,
) -> dict[str, Any]:
    surface_tokens = max(0, raw_surface_tokens)
    system_tokens = next((s["tokens"] for s in reversed(systems) if s["tokens"] > 0), 0)
    message_tokens = max(0, surface_tokens - system_tokens)
    cur_bd = state["breakdown"]
    breakdown = (
        cur_bd
        if system_tokens == cur_bd["systemTokens"]
        and message_tokens == cur_bd["messageTokens"]
        else {
            "systemTokens": system_tokens,
            "toolsTokens": cur_bd["toolsTokens"],
            "messageTokens": message_tokens,
        }
    )
    return {
        **state,
        "systems": systems,
        "surfaceTokens": surface_tokens,
        "breakdown": breakdown,
    }


def _apply_context_breakdown(
    state: dict[str, Any], event: SessionEvent
) -> dict[str, Any]:
    if event.type in ("compaction/summary", "compaction/prune"):
        shadowed = set(event.payload["shadowedSeqs"])
        sr = event.payload["shadowedRange"]
        return {
            **state,
            "claim": {
                "start": int(sr["start"]),
                "end": int(sr["end"]),
                "tokens": int(event.payload["shadowedTokenCount"]),
                "systemSeqs": [
                    s["seq"] for s in state["systems"] if s["seq"] in shadowed
                ],
            },
        }
    unclaimed = (
        state
        if "claim" not in state
        else {k: v for k, v in state.items() if k != "claim"}
    )
    if event.type == "request/header":
        header = canonical_header(event.payload["header"])
        tools_tokens = estimate_tools_tokens(header)
        if tools_tokens == unclaimed["breakdown"]["toolsTokens"]:
            return unclaimed
        return {
            **unclaimed,
            "breakdown": {**unclaimed["breakdown"], "toolsTokens": tools_tokens},
        }
    if event.type not in SURFACE_EVENT_TYPES:
        return unclaimed
    message = derive_event_message(event)
    tokens = 0 if message is None else estimate_message(message)
    systems, raw_surface = _place_surface_event(
        unclaimed, state.get("claim"), event, tokens
    )
    return _with_breakdown_totals(unclaimed, systems, raw_surface)


context_breakdown_projection_definition: ProjectionDefinition[dict[str, Any]] = (
    ProjectionDefinition(
        key="contextBreakdown",
        state_version=6,
        state_schema=_validate_breakdown_state,
        init=_init_context_breakdown,
        apply=_apply_context_breakdown,
        wire=WireView(
            view_schema=_validate_breakdown_view,
            view=lambda s: s["breakdown"],
            summary=False,
        ),
    )
)

contextBreakdownProjectionDefinition = context_breakdown_projection_definition


def _safe_sum(values: Sequence[int]) -> int | None:
    total = 0
    for val in values:
        total += val
        if not is_safe_integer(total):
            return None
    return total


def _message_route(message: Mapping[str, Any]) -> dict[str, str] | None:
    source = message["source"]
    provider = source.get("provider", "")
    model = source.get("model", "")
    return (
        {"provider": provider, "model": model}
        if isinstance(provider, str)
        and len(provider) > 0
        and isinstance(model, str)
        and len(model) > 0
        else None
    )


def _stream_usage(stream: Sequence[dict[str, Any]]) -> Mapping[str, Any] | None:
    chunk = last_assistant_stream_chunk(stream, "usage")
    return None if chunk is None else cast(Mapping[str, Any], chunk["usage"])


def _normalize_usage(
    usage: Mapping[str, Any],
    route: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    inp = usage.get("inputTokens")
    out = usage.get("outputTokens")
    if not is_non_negative_safe_integer(inp) or not is_non_negative_safe_integer(out):
        return None
    cache_read = usage.get("cacheReadTokens")
    if "cacheReadTokens" in usage and not is_non_negative_safe_integer(cache_read):
        return None
    cache_write = usage.get("cacheWriteTokens")
    if "cacheWriteTokens" in usage and not is_non_negative_safe_integer(cache_write):
        return None
    reasoning = usage.get("reasoningTokens")
    if "reasoningTokens" in usage and (
        not is_non_negative_safe_integer(reasoning) or reasoning > out
    ):
        return None

    prompt_parts = [inp]
    if cache_read is not None:
        prompt_parts.append(cache_read)
    if cache_write is not None:
        prompt_parts.append(cache_write)
    known_prompt = _safe_sum(prompt_parts)
    if known_prompt is None:
        return None

    if "totalTokens" in usage:
        total = usage["totalTokens"]
        if not is_non_negative_safe_integer(total):
            return None
        exact_prompt = total - out
        if (
            not is_non_negative_safe_integer(exact_prompt)
            or exact_prompt < known_prompt
        ):
            return None
        if (
            cache_read is not None
            and cache_write is not None
            and exact_prompt != known_prompt
        ):
            return None
        exact_total = total
    else:
        if cache_read is None or cache_write is None:
            return None
        derived_total = _safe_sum([known_prompt, out])
        if derived_total is None:
            return None
        exact_total = derived_total

    normalized: dict[str, Any] = {
        "inputTokens": inp,
        "outputTokens": out,
        "totalTokens": exact_total,
    }
    if cache_read is not None:
        normalized["cacheReadTokens"] = cache_read
    if cache_write is not None:
        normalized["cacheWriteTokens"] = cache_write
    if reasoning is not None:
        normalized["reasoningTokens"] = reasoning
    if route is not None:
        normalized["route"] = route
    return normalized


def _aggregate_attempts(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if len(attempts) == 0:
        return None
    input_tokens = _safe_sum([a["inputTokens"] for a in attempts])
    output_tokens = _safe_sum([a["outputTokens"] for a in attempts])
    total_tokens = _safe_sum([a["totalTokens"] for a in attempts])
    if input_tokens is None or output_tokens is None or total_tokens is None:
        return None

    out: dict[str, Any] = {
        "uncachedInputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": total_tokens,
    }
    if all("cacheReadTokens" in a for a in attempts):
        out["cacheReadTokens"] = sum(a["cacheReadTokens"] for a in attempts)
    if all("cacheWriteTokens" in a for a in attempts):
        out["cacheWriteTokens"] = sum(a["cacheWriteTokens"] for a in attempts)
    if all("reasoningTokens" in a for a in attempts):
        out["reasoningTokens"] = sum(a["reasoningTokens"] for a in attempts)
    if all("route" in a for a in attempts):
        unique: dict[tuple[str, str], dict[str, str]] = {}
        for attempt in attempts:
            route = attempt["route"]
            unique[(route["provider"], route["model"])] = route
        out["routes"] = list(unique.values())
    return out


def derive_turn_token_usage(events: Sequence[SessionEvent]) -> dict[str, Any] | None:
    """Fold one complete turn's durable attempt lifecycle into exact token accounting."""
    state: dict[str, Any] = {"kind": "idle"}
    attempts: list[dict[str, Any]] = []
    turn: int | None = None
    saw_end = False
    invalid = False

    def close_open(route: dict[str, str] | None = None) -> bool:
        if state["kind"] != "open" or "sample" not in state:
            return False
        normalized = _normalize_usage(state["sample"], route)
        if normalized is None:
            return False
        attempts.append(normalized)
        return True

    def same_attempt(ev_turn: Any, ev_step: Any) -> bool:
        return bool(state["turn"] == ev_turn and state["step"] == ev_step)

    for event in events:
        if invalid:
            break
        etype = event.type
        data = event.payload
        if etype == "turn/start":
            if turn is not None or state["kind"] != "idle":
                invalid = True
            else:
                turn = data["turn"]
            continue
        if turn is None:
            invalid = True
            break
        if etype == "turn/end":
            if data["turn"] != turn or state["kind"] != "idle" or saw_end:
                invalid = True
            else:
                saw_end = True
            continue
        if saw_end:
            invalid = True
            break
        if etype == "step/start":
            if data["turn"] != turn or state["kind"] != "idle":
                invalid = True
            else:
                state = {"kind": "open", "turn": turn, "step": data["step"]}
            continue
        if etype == "llm/retry-started":
            if (
                data["turn"] != turn
                or state["kind"] != "settled"
                or state["by"] != "retry"
                or not same_attempt(data["turn"], data["step"])
            ):
                invalid = True
            else:
                state = {"kind": "open", "turn": turn, "step": data["step"]}
            continue
        if etype == "assistant/attempt":
            if (
                data["turn"] != turn
                or state["kind"] != "open"
                or not same_attempt(data["turn"], data["step"])
            ):
                invalid = True
                continue
            sample = _stream_usage(data["stream"]) or state.get("sample")
            state = {
                "kind": "open",
                "turn": turn,
                "step": data["step"],
                **({} if sample is None else {"sample": sample}),
            }
            if not close_open():
                invalid = True
            else:
                state = {"kind": "finishClosed", "turn": turn, "step": data["step"]}
            continue
        if etype == "assistant/message":
            if (
                data["turn"] != turn
                or state["kind"] != "open"
                or not same_attempt(data["turn"], data["step"])
            ):
                invalid = True
                continue
            sample = data.get("usage")
            if sample is None:
                sample = _stream_usage(data["stream"])
            if sample is not None:
                state = {**state, "sample": sample}
            if not close_open(_message_route(data["message"])):
                invalid = True
            else:
                state = {
                    "kind": "settled",
                    "turn": turn,
                    "step": data["step"],
                    "by": "message",
                }
            continue
        if etype == "llm/retry":
            if (
                data["turn"] != turn
                or state["kind"] == "idle"
                or not same_attempt(data["turn"], data["step"])
            ):
                invalid = True
                continue
            if state["kind"] == "settled" or (
                state["kind"] == "open" and not close_open()
            ):
                invalid = True
            if not invalid:
                state = {
                    "kind": "settled",
                    "turn": turn,
                    "step": data["step"],
                    "by": "retry",
                }
            continue
        if etype == "step/end":
            if (
                data["turn"] != turn
                or state["kind"] == "idle"
                or not same_attempt(data["turn"], data["step"])
            ):
                invalid = True
                continue
            if state["kind"] == "open" and not close_open():
                invalid = True
            if not invalid:
                state = {"kind": "idle"}

    if invalid or not saw_end or state["kind"] != "idle":
        return None
    return _aggregate_attempts(attempts)


deriveTurnTokenUsage = derive_turn_token_usage


def _usage_tokens(usage: Mapping[str, Any]) -> int:
    return (
        int(usage["inputTokens"])
        + int(usage.get("cacheReadTokens") or 0)
        + int(usage.get("cacheWriteTokens") or 0)
        + int(usage["outputTokens"])
    )


def _optional_header_equals(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return header_equals(left, right)


def _validate_config_keys(config: Any) -> None:
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise TypeError("TokenMeterConfig: expected a mapping")
    for key in config:
        raise ValueError(
            f'TokenMeterConfig: unknown key "{key}" (no settings are supported)'
        )


class TokenMeter(Service):
    """Single replay-aware token-meter service for request and surface pressure."""

    provide = "tokenMeter"
    inject: ClassVar[list[str]] = ["sessionProjections"]

    def __init__(self, ctx: Context, config: Any = None) -> None:
        registry = ctx.get("sessionProjections", None, strict=False)
        if not isinstance(registry, SessionProjectionRegistry):
            raise RuntimeError(  # noqa: TRY004
                "token-meter requires the sessionProjections service"
            )
        _validate_config_keys(config)
        super().__init__(ctx, "tokenMeter")
        self._registry = registry
        self._states: WeakKeyDictionary[Session, dict[str, Any]] = WeakKeyDictionary()
        self._handles: list[ProjectionHandle[Any]] = []
        self._measurement_handle: ProjectionHandle[dict[str, Any]] | None = None
        self._pressure_handle: ProjectionHandle[dict[str, Any]] | None = None
        self._dispose_event: Callable[[], Any] | None = None
        self._disposed = False

    async def init(self) -> Callable[[], None]:
        """Register metering projections and observe session events."""
        definitions: tuple[ProjectionDefinition[Any], ...] = (
            measurement_projection_definition,
            token_usage_projection_definition,
            context_pressure_projection_definition,
            context_breakdown_projection_definition,
        )
        try:
            for definition in definitions:
                result = self._registry.register(definition)
                handle = await result if inspect.isawaitable(result) else result
                self._handles.append(handle)
                if definition.key == "tokenMeasurement":
                    self._measurement_handle = handle
                elif definition.key == "contextPressure":
                    self._pressure_handle = handle

            if self._disposed:
                self.dispose()
                return self.dispose

            self._dispose_event = self.ctx.on("session/event", self._on_session_event)
        except BaseException:
            self.dispose()
            raise
        return self.dispose

    def _on_session_event(self, session: Session, _event: SessionEvent) -> None:
        if session in self._states:
            self._sync(session)

    def context_pressure(self, session: Session) -> dict[str, int]:
        """Read the latest logged provider prompt size and route capacity."""
        if self._pressure_handle is None:
            raise RuntimeError(
                "token-meter: contextPressure projection is not initialized"
            )
        state = self._pressure_handle.state_of(session)
        out: dict[str, int] = {}
        if "pressureTokens" in state:
            out["pressureTokens"] = state["pressureTokens"]
        if "contextWindow" in state:
            out["contextWindow"] = state["contextWindow"]
        return out

    contextPressure = context_pressure

    def estimate_message(self, message: Mapping[str, Any]) -> int:
        """Heuristically price one model-visible message."""
        return estimate_message(message)

    estimateMessage = estimate_message

    def _route_image_pricing(self, header: dict[str, Any] | None) -> Any:
        if header is None:
            return None
        config = header["config"]
        llm = self.ctx.get("llm", None, strict=False)
        if llm is None:
            return None
        return llm.image_request_pricing(config["provider"], config["model"])

    def _file_request_text(self) -> Callable[[Mapping[str, Any]], str] | None:
        llm = self.ctx.get("llm", None, strict=False)
        if llm is None:
            return None
        return lambda ref: llm.file_request_text(ref)

    @staticmethod
    def _cached_header(
        state: Mapping[str, Any], seq: int | None
    ) -> dict[str, Any] | None:
        if seq is None:
            return None
        header = state["header"]
        if header is not None and header["seq"] == seq:
            return cast(dict[str, Any], header["value"])
        anchor_header = state["anchorHeader"]
        return (
            None
            if anchor_header is None
            else cast(dict[str, Any], anchor_header["value"])
        )

    def _fold_event(self, state: dict[str, Any], event: SessionEvent) -> None:
        if event.type == "request/header":
            state["header"] = {
                "seq": event.seq,
                "value": canonical_header(event.payload["header"]),
            }
        if event.type == "image/offload":
            offloaded = {
                int(target["seq"]): set(target["imageIndexes"])
                for target in event.payload["targets"]
            }
            new_surface: list[dict[str, Any]] = []
            for node in state["surface"]:
                indexes = offloaded.get(int(node["seq"]))
                if indexes is None:
                    new_surface.append(node)
                else:
                    new_images = [
                        {**img, "offloaded": True} if idx in indexes else img
                        for idx, img in enumerate(node["images"])
                    ]
                    new_surface.append({**node, "images": new_images})
            state["surface"] = new_surface
        plan = (
            plan_surface_tokens(state["surface"], event)
            if event.type in SURFACE_EVENT_TYPES
            else None
        )
        if event.type == "assistant/message":
            state["anchorNodes"] = list(state["surface"])
            state["anchorHeader"] = state["header"]
        if plan is not None:
            commit_surface_tokens(state["surface"], plan)

    def _sync(
        self,
        session: Session,
        use_projection: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self._states.get(session)
        if state is None:
            state = {
                "consumedEvents": 0,
                "surface": [],
                "header": None,
                "anchorHeader": None,
                "anchorNodes": [],
                "pendingMeasurement": _init_measurement(),
            }
            self._states[session] = state

        projected = (
            self._measurement_handle.state_of(session)
            if use_projection and self._measurement_handle is not None
            else None
        )
        while state["consumedEvents"] < session.seq:
            event = session.event_at(state["consumedEvents"])
            assert event is not None
            nxt = (
                None
                if projected is not None
                else _apply_measurement(state["pendingMeasurement"], event)
            )
            measurement = projected if nxt is None else nxt
            assert measurement is not None
            if measurement["invalidSeq"] == event.seq:
                if event.type == "assistant/message":
                    plan_surface_tokens(state["surface"], event)
                advance_measurement(measurement, event)
            self._fold_event(state, event)
            if nxt is not None:
                state["pendingMeasurement"] = nxt
            state["consumedEvents"] += 1

        if projected is not None:
            header_entry = state["header"]
            header_seq = None if header_entry is None else header_entry["seq"]
            anchor_obj = projected["anchor"]
            anchor_header = state["anchorHeader"]
            anchor_header_seq = None if anchor_header is None else anchor_header["seq"]
            if projected["headerSeq"] != header_seq or (
                anchor_obj is not None and anchor_obj["headerSeq"] != anchor_header_seq
            ):
                del self._states[session]
                return self._sync(session, False)

        return state, (
            projected if projected is not None else state["pendingMeasurement"]
        )

    def measure(
        self,
        session: Session,
        request_header: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Measure current request pressure and surface through the durable tail."""
        state, measurement = self._sync(session)
        header = (
            self._cached_header(state, measurement["headerSeq"])
            if request_header is None
            else canonical_header(request_header)
        )
        pricing = self._route_image_pricing(header)
        file_text = self._file_request_text()
        surface = price_surface(state["surface"], pricing, file_text)
        anchor = measurement["anchor"]

        if anchor is not None and _optional_header_equals(
            self._cached_header(state, anchor["headerSeq"]), header
        ):
            anchor_surface = price_surface(state["anchorNodes"], pricing, file_text)[
                "surfaceTokens"
            ] + int(anchor["assistantTokens"])
            estimated_anchor = estimate_tools_tokens(header) + anchor_surface
            usage = anchor["usage"]
            if usage is not None and _usage_tokens(usage) >= estimated_anchor:
                baseline: dict[str, Any] = {
                    "kind": "usage",
                    "tokens": _usage_tokens(usage),
                    "usage": usage,
                }
            else:
                baseline = {"kind": "estimated", "tokens": estimated_anchor}
            surface_delta_tokens = surface["surfaceTokens"] - anchor_surface
        elif header is None and surface["surfaceTokens"] == 0:
            baseline = {"kind": "none", "tokens": 0}
            surface_delta_tokens = 0
        else:
            baseline = {
                "kind": "estimated",
                "tokens": estimate_tools_tokens(header) + surface["surfaceTokens"],
            }
            surface_delta_tokens = 0

        payload = copy.deepcopy(
            {
                "logRevision": state["consumedEvents"],
                "baseline": baseline,
                "surfaceDeltaTokens": surface_delta_tokens,
                "totalTokens": max(0, baseline["tokens"] + surface_delta_tokens),
                "surfaceTokens": surface["surfaceTokens"],
                "nodes": surface["nodes"],
            }
        )
        return deep_freeze(payload)  # type: ignore[no-any-return]

    def dispose(self) -> None:
        """Tear down subscriptions and registered projections."""
        self._disposed = True
        if self._dispose_event is not None:
            self._dispose_event()
            self._dispose_event = None
        for handle in reversed(self._handles):
            handle.dispose()
        self._handles.clear()
        self._measurement_handle = None
        self._pressure_handle = None
        self._states.clear()


__all__ = [
    "BLOCK_OVERHEAD",
    "CHARS_PER_TOKEN",
    "ROLE_OVERHEAD",
    "TokenMeter",
    "advanceMeasurement",
    "advance_measurement",
    "commitSurfaceTokens",
    "commit_surface_tokens",
    "contextBreakdownProjectionDefinition",
    "contextPressureProjectionDefinition",
    "context_breakdown_projection_definition",
    "context_pressure_projection_definition",
    "deriveTurnTokenUsage",
    "derive_turn_token_usage",
    "estimateContent",
    "estimateMessage",
    "estimateStructuralBlock",
    "estimateSystemMessage",
    "estimateToolsTokens",
    "estimate_content",
    "estimate_message",
    "estimate_structural_block",
    "estimate_system_message",
    "estimate_tools_tokens",
    "foldSurfaceProjection",
    "fold_surface_projection",
    "measurementProjectionDefinition",
    "measurement_projection_definition",
    "planSurfaceTokens",
    "plan_surface_tokens",
    "priceSurface",
    "price_surface",
    "tokenUsageProjectionDefinition",
    "token_usage_projection_definition",
]
