"""Tests for replay-aware TokenMeter service, estimators, folds, and projections."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from animas import Context, FiberState
from animas.jsexpr import dump_yaml
from animas.loader import load_profile

from magitech_harness import create_harness
from magitech_harness.llm.assistant_stream import AssistantStreamAccumulator
from magitech_harness.llm.service import LLMService
from magitech_harness.llm.token_meter import (
    BLOCK_OVERHEAD,
    CHARS_PER_TOKEN,
    ROLE_OVERHEAD,
    TokenMeter,
    advance_measurement,
    advanceMeasurement,
    commit_surface_tokens,
    commitSurfaceTokens,
    context_breakdown_projection_definition,
    context_pressure_projection_definition,
    contextBreakdownProjectionDefinition,
    contextPressureProjectionDefinition,
    derive_turn_token_usage,
    deriveTurnTokenUsage,
    estimate_content,
    estimate_message,
    estimate_structural_block,
    estimate_system_message,
    estimate_tools_tokens,
    estimateContent,
    estimateMessage,
    estimateStructuralBlock,
    estimateSystemMessage,
    estimateToolsTokens,
    fold_surface_projection,
    foldSurfaceProjection,
    measurement_projection_definition,
    measurementProjectionDefinition,
    plan_surface_tokens,
    planSurfaceTokens,
    price_surface,
    priceSurface,
    token_usage_projection_definition,
    tokenUsageProjectionDefinition,
)
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    ProjectionHandle,
    SessionProjectionRegistry,
)
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session


def _timed_text_stream(
    text: str, *, usage: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    acc = AssistantStreamAccumulator()
    if text:
        acc.push(
            {
                "time": 1,
                "chunk": {"type": "block-start", "index": 0, "blockType": "text"},
            }
        )
        acc.push({"time": 2, "chunk": {"type": "text-delta", "index": 0, "text": text}})
        acc.push(
            {
                "time": 3,
                "chunk": {
                    "type": "block-end",
                    "index": 0,
                    "block": {"type": "text", "text": text},
                },
            }
        )
    if usage is not None:
        acc.push({"time": 4, "chunk": {"type": "usage", "usage": usage}})
    acc.push({"time": 5, "chunk": {"type": "finish", "reason": {"kind": "stop"}}})
    return acc.snapshot()


def _header(
    *,
    provider: str = "mock",
    model: str = "fast",
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hdr: dict[str, Any] = {
        "config": {
            "provider": provider,
            "model": model,
            "reasoningEffort": "medium",
            "sampling": {},
        },
    }
    if tools:
        hdr["tools"] = tools
    return hdr


def _user_event_payload(text: str, *, msg_id: str = "u1") -> dict[str, Any]:
    return {
        "role": "user",
        "id": msg_id,
        "content": [{"type": "text", "text": text}],
        "source": {"kind": "user"},
    }


def _system_event_payload(text: str, *, msg_id: str = "s1") -> dict[str, Any]:
    content = [{"type": "text", "text": text}] if text else []
    return {
        "turn": 1,
        "step": 1,
        "message": {
            "id": msg_id,
            "role": "system",
            "content": content,
            "source": {"kind": "@deepseek-ai/dsh-system-prompt"},
        },
    }


def _assistant_event_payload(
    *,
    turn: int = 1,
    step: int = 1,
    text: str = "ok",
    msg_id: str = "a1",
    provider: str = "mock",
    model: str = "fast",
    usage: dict[str, Any] | None = None,
    stream: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    content = [{"type": "text", "text": text}] if text else []
    payload: dict[str, Any] = {
        "turn": turn,
        "step": step,
        "message": {
            "id": msg_id,
            "role": "assistant",
            "content": content,
            "source": {"kind": "model", "provider": provider, "model": model},
        },
        "stream": stream
        if stream is not None
        else _timed_text_stream(text, usage=usage),
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _attempt_event_payload(
    *,
    turn: int = 1,
    step: int = 1,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "turn": turn,
        "step": step,
        "stream": _timed_text_stream("partial", usage=usage),
    }


def test_config_and_missing_dependency_errors() -> None:
    """TokenMeter rejects unknown config keys, non-mappings, and missing registry."""
    bare_ctx = Context()
    with pytest.raises(
        RuntimeError, match="token-meter requires the sessionProjections service"
    ):
        TokenMeter(bare_ctx)

    ctx = Context()
    reg = SessionProjectionRegistry(ctx)
    try:
        m_default = TokenMeter(ctx, None)
        m_default.dispose()
        with pytest.raises(TypeError, match="TokenMeterConfig: expected a mapping"):
            TokenMeter(ctx, "invalid")  # type: ignore[arg-type]
        with pytest.raises(
            ValueError,
            match='TokenMeterConfig: unknown key "extra" \\(no settings are supported\\)',
        ):
            TokenMeter(ctx, {"extra": 1})
    finally:
        reg.dispose()


def test_heuristic_estimators_and_aliases() -> None:
    """Token estimation primitives match JS UTF-16 density math and aliases."""
    assert CHARS_PER_TOKEN == 4
    assert BLOCK_OVERHEAD == 4
    assert ROLE_OVERHEAD == 4
    assert estimateStructuralBlock is estimate_structural_block
    assert estimateContent is estimate_content
    assert estimateSystemMessage is estimate_system_message
    assert estimateMessage is estimate_message
    assert estimateToolsTokens is estimate_tools_tokens

    # Image block strips offloaded flag when estimating structural heuristic
    img_plain = {
        "type": "image",
        "attachment": {"id": "att-1", "mimeType": "image/png"},
    }
    img_offloaded = {**img_plain, "offloaded": True}
    assert estimate_structural_block(img_offloaded) == estimate_structural_block(
        img_plain
    )

    # Other structural block keeps full JSON
    custom_block = {"type": "custom", "payload": "abcd"}
    expected_custom = BLOCK_OVERHEAD + (
        (len(json.dumps(custom_block, separators=(",", ":"), ensure_ascii=False)) + 3)
        // CHARS_PER_TOKEN
    )
    assert estimate_structural_block(custom_block) == expected_custom

    # Text, reasoning, tool-call, surrogate pair UTF-16 length (😀 is 2 UTF-16 units)
    assert estimate_content([{"type": "text", "text": "a😀"}]) == 1 + BLOCK_OVERHEAD
    assert (
        estimate_content([{"type": "reasoning", "text": "12345"}]) == 2 + BLOCK_OVERHEAD
    )
    assert (
        estimate_content(
            [{"type": "tool-call", "id": "c1", "name": "read", "arguments": "{}"}]
        )
        == 1 + 1 + BLOCK_OVERHEAD
    )
    assert estimate_content([img_plain]) == estimate_structural_block(img_plain)

    # System message empty vs mixed blocks
    assert estimate_system_message({"role": "system", "content": []}) == 0
    mixed_sys = {
        "role": "system",
        "content": [
            {"type": "text", "text": "abcd"},
            {"type": "custom", "k": 1},
        ],
    }
    chars = 4 + len(json.dumps({"type": "custom", "k": 1}, separators=(",", ":")))
    assert estimate_system_message(mixed_sys) == ((chars + 3) // 4) + ROLE_OVERHEAD
    assert estimate_message(mixed_sys) == estimate_system_message(mixed_sys)
    assert estimate_message(
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ) == (1 + BLOCK_OVERHEAD + ROLE_OVERHEAD)

    # Tools header
    assert estimate_tools_tokens(None) == 0
    assert estimate_tools_tokens(_header(tools=[])) == 0
    tools = [{"type": "function", "function": {"name": "ping", "description": "x"}}]
    assert estimate_tools_tokens(_header(tools=tools)) == (
        ((len(json.dumps(tools, separators=(",", ":"))) + 3) // 4) + BLOCK_OVERHEAD
    )


def test_surface_fold_and_route_pricing() -> None:
    """Plan/commit surface fold and route-owned attachment pricing work accurately."""
    assert priceSurface is price_surface
    assert planSurfaceTokens is plan_surface_tokens
    assert commitSurfaceTokens is commit_surface_tokens

    nodes: list[dict[str, Any]] = []
    session = Session.create("surface-fold")
    file_block = {
        "type": "file",
        "attachment": {"id": "f1", "mimeType": "text/plain", "filename": "a.txt"},
    }
    img_block = {
        "type": "image",
        "attachment": {"id": "i1", "mimeType": "image/png"},
    }
    e0 = session.append(
        "user/message",
        {
            "role": "user",
            "id": "u1",
            "content": [{"type": "text", "text": "hello"}, file_block, img_block],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )
    plan0 = plan_surface_tokens(nodes, e0)
    assert plan0["target"] == "append"
    commit_surface_tokens(nodes, plan0)
    assert len(nodes) == 1

    # Heuristic surface pricing when no route pricing/fileText supplied
    plain_priced = price_surface(nodes, None)
    assert plain_priced["surfaceTokens"] == nodes[0]["heuristicTokens"]
    assert plain_priced["nodes"][0] == {
        "seq": 0,
        "tokens": nodes[0]["heuristicTokens"],
        "heuristicTokens": nodes[0]["heuristicTokens"],
    }

    # Route image pricing + file request text projection (mapping and object pricing)
    pricing_ok = {
        "priceImages": lambda imgs: [{"visualTokens": 50, "text": ""} for _ in imgs]
    }

    class _ObjPricing:
        @staticmethod
        def price_images(imgs: Any) -> list[dict[str, Any]]:
            return [{"visualTokens": 50, "text": ""} for _ in imgs]

    file_text = lambda ref: f"<file:{ref['filename']}>"
    routed = price_surface(nodes, pricing_ok, file_text)
    routed_obj = price_surface(nodes, _ObjPricing(), file_text)
    expected_tokens = (
        nodes[0]["heuristicTokens"]
        - nodes[0]["fileStructuralTokens"]
        + estimate_content([{"type": "text", "text": "<file:a.txt>"}])
        - nodes[0]["imageStructuralTokens"]
        + 50
        + estimate_content([{"type": "text", "text": ""}])
    )
    assert routed["surfaceTokens"] == expected_tokens
    assert routed_obj["surfaceTokens"] == expected_tokens

    # Empty image pricing without file_text on image-only surface
    pricing_bad = {"priceImages": lambda _imgs: []}
    with pytest.raises(ValueError, match="answered 0 prices for 1 occurrences"):
        price_surface(nodes, pricing_bad)

    # Empty-content assistant event derives null message (0 tokens)
    e1 = session.append(
        "assistant/message",
        _assistant_event_payload(text="", msg_id="a-empty"),
        surface_op="append",
    )
    plan1 = plan_surface_tokens(nodes, e1)
    assert plan1["tokens"] == 0
    commit_surface_tokens(nodes, plan1)
    assert (
        price_surface(nodes, pricing_ok, file_text)["surfaceTokens"] == expected_tokens
    )

    # Valid replace plan
    e2 = session.append(
        "user/message",
        _user_event_payload("summary", msg_id="u-sum"),
        surface_op={"op": "replace", "startSeq": 0, "endSeq": 1},
        source_event_seqs=[0, 1],
    )
    plan2 = plan_surface_tokens(nodes, e2)
    commit_surface_tokens(nodes, plan2)
    assert [n["seq"] for n in nodes] == [2]

    # Invalid replace range fails loud without mutating nodes
    bad_replace = SessionEvent(
        id="bad",
        session_id="surface-fold",
        type="user/message",
        timestamp=1.0,
        payload=_user_event_payload("x", msg_id="ux"),
        seq=99,
        surface_op={"op": "replace", "startSeq": 0, "endSeq": 0},
    )
    snapshot_before = list(nodes)
    with pytest.raises(ValueError, match="has invalid current range 0-0"):
        plan_surface_tokens(nodes, bad_replace)
    assert nodes == snapshot_before


def test_surface_projection_shadow_claim_protocol() -> None:
    """fold_surface_projection handles compaction claims, expiry, and range errors."""
    assert foldSurfaceProjection is fold_surface_projection

    s = Session.create("shadow")
    ev_u = s.append("user/message", _user_event_payload("alpha"), surface_op="append")
    r_u = fold_surface_projection(None, ev_u)
    assert r_u["claim"] is None
    assert r_u["deltaTokens"] == estimate_message(ev_u.payload)

    # Non-surface unrelated event resets claim
    ev_step = s.append("step/start", {"turn": 1, "step": 1})
    r_step = fold_surface_projection({"start": 0, "end": 0, "tokens": 5}, ev_step)
    assert r_step == {"deltaTokens": 0, "claim": None}

    # Unmetered replace degrades neutrally (0 delta)
    ev_unmetered = s.append(
        "user/message",
        _user_event_payload("unmetered"),
        surface_op={"op": "replace", "startSeq": 0, "endSeq": 0},
        source_event_seqs=[0],
    )
    assert fold_surface_projection(None, ev_unmetered) == {
        "deltaTokens": 0,
        "claim": None,
    }

    # Arm claim via compaction/prune and consume with matching replace
    ev_prune = s.append(
        "compaction/prune",
        {
            "shadowedRange": {"start": 2, "end": 2},
            "shadowedSeqs": [2],
            "shadowedTokenCount": 42,
        },
    )
    armed = fold_surface_projection(None, ev_prune)
    assert armed == {
        "deltaTokens": 0,
        "claim": {"start": 2, "end": 2, "tokens": 42},
    }
    ev_repl = s.append(
        "user/message",
        _user_event_payload("kept", msg_id="u-kept"),
        surface_op={"op": "replace", "startSeq": 2, "endSeq": 2},
        source_event_seqs=[2],
    )
    consumed = fold_surface_projection(armed["claim"], ev_repl)
    assert consumed == {
        "deltaTokens": estimate_message(ev_repl.payload) - 42,
        "claim": None,
    }

    # Mismatched replace range against armed claim fails loud
    bad_ev = SessionEvent(
        id="bad-claim",
        session_id="shadow",
        type="user/message",
        timestamp=1.0,
        payload=_user_event_payload("mismatch", msg_id="um"),
        seq=50,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 2},
    )
    with pytest.raises(ValueError, match="has no adjacent shadow price"):
        fold_surface_projection(armed["claim"], bad_ev)


def test_measurement_projection_and_advance_validation() -> None:
    """measurementProjectionDefinition tracks anchors and sticky invalidSeq."""
    assert measurementProjectionDefinition is measurement_projection_definition
    assert advanceMeasurement is advance_measurement
    assert measurement_projection_definition.key == "tokenMeasurement"
    assert measurement_projection_definition.state_version == 2
    assert measurement_projection_definition.wire is None

    init_state = measurement_projection_definition.init()
    assert init_state == {
        "headerSeq": None,
        "stepStart": None,
        "anchor": None,
        "invalidSeq": None,
    }
    assert measurement_projection_definition.state_schema(init_state) == init_state
    valid_full = {
        "headerSeq": 0,
        "stepStart": {"turn": 1, "step": 1},
        "anchor": {
            "headerSeq": 0,
            "assistantTokens": 12,
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        },
        "invalidSeq": None,
    }
    assert measurement_projection_definition.state_schema(valid_full) == valid_full
    for bad in (
        "nope",
        {**init_state, "extra": 1},
        {**init_state, "headerSeq": -1},
        {**init_state, "stepStart": {"turn": "1", "step": 1}},
        {
            **init_state,
            "anchor": {"headerSeq": 0, "assistantTokens": 1, "usage": "bad"},
        },
    ):
        with pytest.raises(ValueError):
            measurement_projection_definition.state_schema(bad)

    s = Session.create("meas")
    e_step1 = s.append("step/start", {"turn": 1, "step": 1})
    st1 = measurement_projection_definition.apply(init_state, e_step1)
    assert st1["stepStart"] == {"turn": 1, "step": 1}

    # Unrelated event leaves state untouched (and advance_measurement returns shallow copy)
    e_turn = s.append("turn/start", {"turn": 1})
    assert measurement_projection_definition.apply(st1, e_turn) is st1
    assert advance_measurement(st1, e_turn) == st1

    # assistant/message without usage or prior headerSeq sets heuristic anchor with usage=None
    e_msg_no_hdr = s.append(
        "assistant/message",
        _assistant_event_payload(turn=1, step=1, text="no header", usage=None),
        surface_op="append",
    )
    st_anchored = advance_measurement(st1, e_msg_no_hdr)
    assert st_anchored["anchor"]["usage"] is None
    assert st_anchored["anchor"]["headerSeq"] is None
    assert st_anchored["anchor"]["assistantTokens"] > 0

    # Duplicate step/start sets invalidSeq sticky without breaking replay
    dup_step = SessionEvent(
        id="dup",
        session_id="meas",
        type="step/start",
        timestamp=1.0,
        payload={"turn": 1, "step": 2},
        seq=99,
    )
    with pytest.raises(ValueError, match="arrived before turn 1/step 1 ended"):
        advance_measurement(st1, dup_step)
    failed_st = measurement_projection_definition.apply(st1, dup_step)
    assert failed_st["invalidSeq"] == 99
    assert measurement_projection_definition.apply(failed_st, e_turn) is failed_st

    # step/end & assistant/message without matching step/start fail loud
    end_no_start = SessionEvent(
        id="end-bad",
        session_id="meas",
        type="step/end",
        timestamp=1.0,
        payload={"turn": 1, "step": 1},
        seq=5,
    )
    with pytest.raises(ValueError, match="has no matching step/start event"):
        advance_measurement(init_state, end_no_start)
    msg_no_start = SessionEvent(
        id="msg-bad",
        session_id="meas",
        type="assistant/message",
        timestamp=1.0,
        payload=_assistant_event_payload(text="hi"),
        seq=6,
        surface_op="append",
    )
    with pytest.raises(ValueError, match="has no matching step/start event"):
        advance_measurement(init_state, msg_no_start)


def test_token_usage_and_context_pressure_projections() -> None:
    """tokenUsage and contextPressure fold usage, retries, capacity, and compactions."""
    assert tokenUsageProjectionDefinition is token_usage_projection_definition
    assert contextPressureProjectionDefinition is context_pressure_projection_definition

    assert token_usage_projection_definition.key == "tokenUsage"
    assert token_usage_projection_definition.state_version == 2
    assert token_usage_projection_definition.wire is not None
    assert token_usage_projection_definition.wire.summary is True

    tu_state = token_usage_projection_definition.init()
    s = Session.create("usage-press")
    ev_other = s.append("step/start", {"turn": 1, "step": 1})
    assert token_usage_projection_definition.apply(tu_state, ev_other) is tu_state

    # assistant/attempt with usage sets initial last slot for (1, 1)
    ev_att1 = s.append(
        "assistant/attempt",
        _attempt_event_payload(
            turn=1,
            step=1,
            usage={"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 5},
        ),
    )
    tu_state = token_usage_projection_definition.apply(tu_state, ev_att1)
    assert token_usage_projection_definition.wire.view(tu_state) == {
        "uncachedInputTokens": 100,
        "outputTokens": 20,
        "cacheReadTokens": 5,
        "cacheWriteTokens": 0,
    }
    # Idempotent identical attempt leaves state unchanged
    assert token_usage_projection_definition.apply(tu_state, ev_att1) is tu_state

    # Same (turn, step) settlement before retry-started replaces previous buckets
    ev_msg1 = s.append(
        "assistant/message",
        _assistant_event_payload(
            turn=1,
            step=1,
            usage={
                "inputTokens": 120,
                "outputTokens": 30,
                "cacheReadTokens": 5,
                "cacheWriteTokens": 2,
            },
        ),
        surface_op="append",
    )
    tu_state = token_usage_projection_definition.apply(tu_state, ev_msg1)
    assert token_usage_projection_definition.wire.view(tu_state) == {
        "uncachedInputTokens": 120,
        "outputTokens": 30,
        "cacheReadTokens": 5,
        "cacheWriteTokens": 2,
    }

    # llm/retry-started closes slot so retried attempt adds instead of replacing
    ev_rs = s.append(
        "llm/retry-started",
        {"retryId": "r1", "turn": 1, "step": 1, "retry": 1},
    )
    tu_state = token_usage_projection_definition.apply(tu_state, ev_rs)
    assert tu_state["last"] is None
    # Non-matching retry-started is ignored
    ev_rs_other = s.append(
        "llm/retry-started",
        {"retryId": "r2", "turn": 9, "step": 1, "retry": 1},
    )
    assert token_usage_projection_definition.apply(tu_state, ev_rs_other) is tu_state

    ev_att2 = s.append(
        "assistant/attempt",
        _attempt_event_payload(
            turn=1, step=1, usage={"inputTokens": 10, "outputTokens": 4}
        ),
    )
    tu_state = token_usage_projection_definition.apply(tu_state, ev_att2)
    assert token_usage_projection_definition.wire.view(tu_state) == {
        "uncachedInputTokens": 130,
        "outputTokens": 34,
        "cacheReadTokens": 5,
        "cacheWriteTokens": 2,
    }

    # Stream without usage is ignored
    ev_no_usage = s.append(
        "assistant/message",
        _assistant_event_payload(
            turn=1, step=2, usage=None, stream=_timed_text_stream("hi")
        ),
        surface_op="append",
    )
    assert token_usage_projection_definition.apply(tu_state, ev_no_usage) is tu_state

    # Validation check on tokenUsage state_schema & view_schema
    assert token_usage_projection_definition.state_schema(tu_state) == tu_state
    tu_none_last = token_usage_projection_definition.init()
    assert token_usage_projection_definition.state_schema(tu_none_last) == tu_none_last
    assert (
        token_usage_projection_definition.wire.view_schema(tu_state["totals"])
        == tu_state["totals"]
    )
    for bad_tu in (
        {"totals": {}, "last": None},
        {
            "totals": {
                "uncachedInputTokens": -1,
                "outputTokens": 0,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
            },
            "last": None,
        },
        {
            "totals": tu_state["totals"],
            "last": {"turn": -1, "step": 1, "buckets": tu_state["totals"]},
        },
    ):
        with pytest.raises(ValueError):
            token_usage_projection_definition.state_schema(bad_tu)

    # contextPressure projection checks
    assert context_pressure_projection_definition.key == "contextPressure"
    assert context_pressure_projection_definition.state_version == 5
    assert context_pressure_projection_definition.wire is not None
    assert context_pressure_projection_definition.wire.summary is False

    cp = context_pressure_projection_definition.init()
    assert context_pressure_projection_definition.wire.view(cp) == {}
    assert context_pressure_projection_definition.wire.view(
        {"surfaceTokens": 10, "pressureTokens": 50}
    ) == {"pressureTokens": 50}
    ev_ctx = s.append(
        "request/context", {"provider": "mock", "model": "m", "contextWindow": 8192}
    )
    cp = context_pressure_projection_definition.apply(cp, ev_ctx)
    assert context_pressure_projection_definition.wire.view(cp) == {
        "contextWindow": 8192
    }
    # Removing contextWindow omits key cleanly
    ev_ctx_none = s.append("request/context", {"provider": "mock", "model": "m"})
    cp = context_pressure_projection_definition.apply(cp, ev_ctx_none)
    assert "contextWindow" not in cp
    # Same None omits no change
    cp_same = context_pressure_projection_definition.apply(cp, ev_ctx_none)
    assert cp_same is cp

    cp = context_pressure_projection_definition.apply(cp, ev_ctx)
    cp = context_pressure_projection_definition.apply(cp, ev_msg1)
    # Repeat with same usage and no surface delta
    ev_same_att = s.append(
        "assistant/attempt",
        _attempt_event_payload(
            turn=1,
            step=1,
            usage={
                "inputTokens": 120,
                "outputTokens": 30,
                "cacheReadTokens": 5,
                "cacheWriteTokens": 2,
            },
        ),
    )
    cp_idemp = context_pressure_projection_definition.apply(cp, ev_same_att)
    cp_idemp2 = context_pressure_projection_definition.apply(cp_idemp, ev_same_att)
    assert cp_idemp2 is cp_idemp
    assert cp_idemp["pressureTokens"] == 127
    view_cp = context_pressure_projection_definition.wire.view(cp)
    assert view_cp["pressureTokens"] == 127
    assert view_cp["projectedTokens"] >= 127
    assert view_cp["contextWindow"] == 8192

    # Compaction claim through contextPressure
    ev_cp_prune = s.append(
        "compaction/prune",
        {
            "shadowedRange": {"start": ev_msg1.seq, "end": ev_msg1.seq},
            "shadowedSeqs": [ev_msg1.seq],
            "shadowedTokenCount": 5,
        },
    )
    cp_claimed = context_pressure_projection_definition.apply(cp, ev_cp_prune)
    assert "claim" in cp_claimed
    assert context_pressure_projection_definition.state_schema(cp_claimed) == cp_claimed
    assert context_pressure_projection_definition.wire.view_schema(view_cp) == view_cp
    for bad_cp in (
        {"surfaceTokens": -1},
        {"surfaceTokens": 0, "contextWindow": 0},
        {"surfaceTokens": 0, "pressureTokens": -1},
        {"surfaceTokens": 0, "claim": {"start": 0, "end": 0, "tokens": -1}},
    ):
        with pytest.raises(ValueError):
            context_pressure_projection_definition.state_schema(bad_cp)
    for bad_cp_view in ({"contextWindow": 0}, {"pressureTokens": -1}):
        with pytest.raises(ValueError):
            context_pressure_projection_definition.wire.view_schema(bad_cp_view)
    cp_cleared = context_pressure_projection_definition.apply(cp_claimed, ev_other)
    assert "claim" not in cp_cleared


def test_context_breakdown_projection() -> None:
    """contextBreakdown tracks surviving systems, tools header, and compaction claims."""
    assert (
        contextBreakdownProjectionDefinition is context_breakdown_projection_definition
    )
    assert context_breakdown_projection_definition.key == "contextBreakdown"
    assert context_breakdown_projection_definition.state_version == 6
    assert context_breakdown_projection_definition.wire is not None
    assert context_breakdown_projection_definition.wire.summary is False

    s = Session.create("breakdown")
    cb = context_breakdown_projection_definition.init()
    assert context_breakdown_projection_definition.state_schema(cb) == cb
    assert (
        context_breakdown_projection_definition.wire.view_schema(cb["breakdown"])
        == cb["breakdown"]
    )
    e_sys1 = s.append(
        "system/message",
        _system_event_payload("sys one", msg_id="s1"),
        surface_op="append",
    )
    cb = context_breakdown_projection_definition.apply(cb, e_sys1)
    assert cb["breakdown"]["systemTokens"] > 0

    # Non-surface event leaves breakdown unchanged (and clears any stale claim)
    e_step = s.append("step/start", {"turn": 1, "step": 1})
    assert context_breakdown_projection_definition.apply(cb, e_step) is cb

    e_hdr = s.append(
        "request/header",
        {
            "header": _header(
                tools=[
                    {"type": "function", "function": {"name": "t1", "description": "d"}}
                ]
            ),
            "reason": "initial",
        },
    )
    cb = context_breakdown_projection_definition.apply(cb, e_hdr)
    assert cb["breakdown"]["toolsTokens"] > 0
    # Repeated identical header is idempotent
    assert context_breakdown_projection_definition.apply(cb, e_hdr) is cb

    # User message adds to messageTokens
    e_u1 = s.append(
        "user/message", _user_event_payload("hello user"), surface_op="append"
    )
    cb = context_breakdown_projection_definition.apply(cb, e_u1)
    assert cb["breakdown"]["messageTokens"] == estimate_message(e_u1.payload)

    # Unmetered replace of a non-system message leaves breakdown neutral
    e_u1_rew = s.append(
        "user/message",
        _user_event_payload("rewritten user", msg_id="u1r"),
        surface_op={"op": "replace", "startSeq": e_u1.seq, "endSeq": e_u1.seq},
        source_event_seqs=[e_u1.seq],
    )
    cb = context_breakdown_projection_definition.apply(cb, e_u1_rew)

    # Second system message supersedes first system message into messageTokens
    e_sys2 = s.append(
        "system/message",
        _system_event_payload("sys two", msg_id="s2"),
        surface_op="append",
    )
    cb = context_breakdown_projection_definition.apply(cb, e_sys2)

    # Single-seq unmetered rewrite on surviving system node updates it cleanly
    e_sys_rew = s.append(
        "system/message",
        _system_event_payload("sys updated", msg_id="s2r"),
        surface_op={"op": "replace", "startSeq": e_sys2.seq, "endSeq": e_sys2.seq},
        source_event_seqs=[e_sys2.seq],
    )
    cb = context_breakdown_projection_definition.apply(cb, e_sys_rew)
    assert cb["systems"][-1]["seq"] == e_sys_rew.seq

    # Compaction claim removing system 0 and user 2, replaced by summary user message
    e_sum = s.append(
        "compaction/summary",
        {
            "compactionId": "c1",
            "shadowedRange": {"start": e_sys1.seq, "end": e_u1_rew.seq},
            "shadowedSeqs": [e_sys1.seq, e_u1_rew.seq],
            "shadowedTokenCount": estimate_message(e_sys1.payload["message"])
            + estimate_message(e_u1.payload),
            "summary": "summary",
            "provider": "mock",
            "model": "fast",
        },
    )
    cb = context_breakdown_projection_definition.apply(cb, e_sum)
    assert "claim" in cb
    assert context_breakdown_projection_definition.state_schema(cb) == cb
    for bad_cb in (
        {**cb, "surfaceTokens": -1},
        {**cb, "systems": "bad"},
        {**cb, "systems": [{"seq": -1, "tokens": 1}]},
        {**cb, "claim": {"start": 0, "end": 0, "tokens": 1, "systemSeqs": "bad"}},
    ):
        with pytest.raises(ValueError):
            context_breakdown_projection_definition.state_schema(bad_cb)
    with pytest.raises(ValueError):
        context_breakdown_projection_definition.wire.view_schema(
            {"systemTokens": -1, "toolsTokens": 0, "messageTokens": 0}
        )
    e_repl = SessionEvent(
        id="cmp-repl",
        session_id="breakdown",
        type="user/message",
        timestamp=1.0,
        payload=_user_event_payload("compacted summary", msg_id="u-cmp"),
        seq=e_sum.seq + 1,
        surface_op={"op": "replace", "startSeq": e_sys1.seq, "endSeq": e_u1_rew.seq},
        source_event_seqs=[e_sys1.seq, e_u1_rew.seq],
    )
    cb = context_breakdown_projection_definition.apply(cb, e_repl)
    assert "claim" not in cb
    assert context_breakdown_projection_definition.wire.view(cb) == cb["breakdown"]
    assert [n["seq"] for n in cb["systems"]] == [e_sys_rew.seq]

    # Error when metered range receives a system/message
    cb_armed = context_breakdown_projection_definition.apply(cb, e_sum)
    bad_sys = SessionEvent(
        id="bad-sys",
        session_id="breakdown",
        type="system/message",
        timestamp=1.0,
        payload=_system_event_payload("bad", msg_id="s-bad"),
        seq=80,
        surface_op={"op": "replace", "startSeq": e_sys1.seq, "endSeq": e_u1_rew.seq},
    )
    with pytest.raises(
        ValueError, match="puts a system message in place of a metered range"
    ):
        context_breakdown_projection_definition.apply(cb_armed, bad_sys)

    # Error when claim range disagrees
    bad_range = SessionEvent(
        id="bad-range",
        session_id="breakdown",
        type="user/message",
        timestamp=1.0,
        payload=_user_event_payload("bad", msg_id="ub"),
        seq=81,
        surface_op={"op": "replace", "startSeq": 999, "endSeq": 999},
    )
    with pytest.raises(ValueError, match="has no adjacent shadow price"):
        context_breakdown_projection_definition.apply(cb_armed, bad_range)

    # Replacing a system node with non-system unmetered removes it cleanly
    e_rm_sys = s.append(
        "user/message",
        _user_event_payload("replaced sys", msg_id="urs"),
        surface_op={
            "op": "replace",
            "startSeq": e_sys_rew.seq,
            "endSeq": e_sys_rew.seq,
        },
        source_event_seqs=[e_sys_rew.seq],
    )
    cb = context_breakdown_projection_definition.apply(cb, e_rm_sys)
    assert cb["systems"] == []


def test_derive_turn_token_usage() -> None:
    """derive_turn_token_usage aggregates proven attempts and rejects incomplete turns."""
    assert deriveTurnTokenUsage is derive_turn_token_usage
    s = Session.create("turn-usage")
    e_ts = s.append("turn/start", {"turn": 1})
    e_ss1 = s.append("step/start", {"turn": 1, "step": 1})
    e_att1 = s.append(
        "assistant/attempt",
        _attempt_event_payload(
            turn=1,
            step=1,
            usage={
                "inputTokens": 100,
                "outputTokens": 10,
                "cacheReadTokens": 20,
                "cacheWriteTokens": 5,
                "reasoningTokens": 4,
            },
        ),
    )
    e_r1 = s.append(
        "llm/retry",
        {
            "retryId": "rid-1",
            "turn": 1,
            "step": 1,
            "provider": "mock",
            "mode": "normal",
            "policyKey": "pk",
            "retry": 1,
            "maxRetries": 2,
            "delayMs": 1,
            "failure": {"message": "temp", "code": "SERVER"},
        },
    )
    e_rs1 = s.append(
        "llm/retry-started", {"retryId": "rid-1", "turn": 1, "step": 1, "retry": 1}
    )
    e_msg1 = s.append(
        "assistant/message",
        _assistant_event_payload(
            turn=1,
            step=1,
            provider="mock",
            model="fast",
            usage={
                "inputTokens": 110,
                "outputTokens": 25,
                "totalTokens": 165,
                "cacheReadTokens": 30,
                "cacheWriteTokens": 0,
                "reasoningTokens": 10,
            },
        ),
        surface_op="append",
    )
    e_se1 = s.append("step/end", {"turn": 1, "step": 1})
    e_te = s.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    summary = derive_turn_token_usage(
        [e_ts, e_ss1, e_att1, e_r1, e_rs1, e_msg1, e_se1, e_te]
    )
    assert summary == {
        "uncachedInputTokens": 210,
        "outputTokens": 35,
        "totalTokens": 300,
        "cacheReadTokens": 50,
        "cacheWriteTokens": 5,
        "reasoningTokens": 14,
    }

    # Route attribution present when all attempts come from assistant/message
    summary_routes = derive_turn_token_usage([e_ts, e_ss1, e_msg1, e_se1, e_te])
    assert summary_routes == {
        "uncachedInputTokens": 110,
        "outputTokens": 25,
        "totalTokens": 165,
        "cacheReadTokens": 30,
        "cacheWriteTokens": 0,
        "reasoningTokens": 10,
        "routes": [{"provider": "mock", "model": "fast"}],
    }

    # Missing turn/end, malformed total, or missing start returns None
    assert derive_turn_token_usage([e_ts, e_ss1, e_msg1, e_se1]) is None
    assert derive_turn_token_usage([e_ss1, e_te]) is None
    bad_total_msg = s.append(
        "assistant/message",
        _assistant_event_payload(
            turn=1,
            step=2,
            msg_id="a-bad-tot",
            usage={
                "inputTokens": 10,
                "outputTokens": 5,
                "totalTokens": 999,
                "cacheReadTokens": 1,
                "cacheWriteTokens": 1,
            },
        ),
        surface_op="append",
    )
    e_ss2 = s.append("step/start", {"turn": 1, "step": 2})
    e_se2 = s.append("step/end", {"turn": 1, "step": 2})
    assert derive_turn_token_usage([e_ts, e_ss2, bad_total_msg, e_se2, e_te]) is None

    def _ev(seq: int, etype: str, payload: dict[str, Any]) -> SessionEvent:
        return SessionEvent(
            id=f"e-{seq}",
            session_id="raw",
            type=etype,
            timestamp=float(seq),
            payload=payload,
            seq=seq,
        )

    def _msg_ev(
        seq: int,
        usage_val: dict[str, Any] | None = None,
        *,
        step: int = 1,
        provider: str = "mock",
        model: str = "fast",
        stream_usage: dict[str, Any] | None = None,
    ) -> SessionEvent:
        stream = (
            []
            if stream_usage is None
            else [
                {
                    "type": "chunk",
                    "time": seq,
                    "chunk": {"type": "usage", "usage": stream_usage},
                }
            ]
        )
        data: dict[str, Any] = {
            "turn": 1,
            "step": step,
            "stream": stream,
            "message": {
                "id": f"m-{seq}",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "source": {"kind": "model", "provider": provider, "model": model},
            },
        }
        if usage_val is not None:
            data["usage"] = usage_val
        return _ev(seq, "assistant/message", data)

    def _wrap(*middle: SessionEvent) -> list[SessionEvent]:
        return [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "step/start", {"turn": 1, "step": 1}),
            *middle,
            _ev(89, "tool/result", {"turn": 1, "step": 1}),
            _ev(90, "step/end", {"turn": 1, "step": 1}),
            _ev(91, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        ]

    # Stream fallback when final message omits usage
    assert derive_turn_token_usage(
        _wrap(
            _msg_ev(
                3,
                None,
                stream_usage={
                    "inputTokens": 10,
                    "outputTokens": 2,
                    "cacheReadTokens": 1,
                    "cacheWriteTokens": 0,
                },
            )
        )
    ) == {
        "uncachedInputTokens": 10,
        "outputTokens": 2,
        "totalTokens": 13,
        "cacheReadTokens": 1,
        "cacheWriteTokens": 0,
        "routes": [{"provider": "mock", "model": "fast"}],
    }

    # Unattributed route & omitted optional buckets
    assert derive_turn_token_usage(
        _wrap(
            _msg_ev(
                3,
                {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
                provider="",
                model="",
            )
        )
    ) == {"uncachedInputTokens": 10, "outputTokens": 2, "totalTokens": 12}

    # Scheduled retry that never started (open attempt closed on llm/retry)
    assert derive_turn_token_usage(
        _wrap(
            _ev(
                3,
                "assistant/attempt",
                {
                    "turn": 1,
                    "step": 1,
                    "stream": [
                        {
                            "type": "chunk",
                            "time": 3,
                            "chunk": {
                                "type": "usage",
                                "usage": {
                                    "inputTokens": 8,
                                    "outputTokens": 2,
                                    "totalTokens": 10,
                                },
                            },
                        }
                    ],
                },
            ),
            _ev(4, "llm/retry", {"turn": 1, "step": 1}),
        )
    ) == {"uncachedInputTokens": 8, "outputTokens": 2, "totalTokens": 10}

    max_safe = (1 << 53) - 1
    half = (max_safe // 2) + 1
    for bad_usage in (
        {"inputTokens": -1, "outputTokens": 1, "totalTokens": 1},
        {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2, "cacheReadTokens": -1},
        {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2, "cacheWriteTokens": -1},
        {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2, "reasoningTokens": 5},
        {
            "inputTokens": max_safe,
            "outputTokens": 0,
            "totalTokens": max_safe,
            "cacheReadTokens": 1,
        },
        {"inputTokens": 1, "outputTokens": 1, "totalTokens": -1},
        {"inputTokens": 10, "outputTokens": 5, "totalTokens": 12},
        {"inputTokens": 10, "outputTokens": 5, "cacheReadTokens": 1},
        {
            "inputTokens": max_safe,
            "outputTokens": 1,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
        },
    ):
        assert derive_turn_token_usage(_wrap(_msg_ev(3, bad_usage))) is None

    # Aggregate safe-integer overflow
    overflow_events = [
        _ev(1, "turn/start", {"turn": 1}),
        _ev(2, "step/start", {"turn": 1, "step": 1}),
        _msg_ev(3, {"inputTokens": half, "outputTokens": 0, "totalTokens": half}),
        _ev(4, "step/end", {"turn": 1, "step": 1}),
        _ev(5, "step/start", {"turn": 1, "step": 2}),
        _msg_ev(
            6, {"inputTokens": half, "outputTokens": 0, "totalTokens": half}, step=2
        ),
        _ev(7, "step/end", {"turn": 1, "step": 2}),
        _ev(8, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    ]
    assert derive_turn_token_usage(overflow_events) is None

    # Invalid lifecycle transitions fail closed
    for invalid_seq in (
        [_ev(1, "turn/start", {"turn": 1}), _ev(2, "turn/end", {"turn": 1})],
        [_ev(1, "turn/start", {"turn": 1}), _ev(2, "turn/start", {"turn": 1})],
        [_ev(1, "turn/start", {"turn": 1}), _ev(2, "turn/end", {"turn": 2})],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "turn/end", {"turn": 1}),
            _ev(3, "step/start", {"turn": 1, "step": 1}),
        ],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "step/start", {"turn": 2, "step": 1}),
        ],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "llm/retry-started", {"turn": 1, "step": 1}),
        ],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "step/start", {"turn": 1, "step": 1}),
            _ev(3, "assistant/attempt", {"turn": 1, "step": 2, "stream": []}),
        ],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "step/start", {"turn": 1, "step": 1}),
            _ev(3, "assistant/attempt", {"turn": 1, "step": 1, "stream": []}),
        ],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "step/start", {"turn": 1, "step": 1}),
            _msg_ev(3, None),
        ],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "step/start", {"turn": 1, "step": 1}),
            _msg_ev(3, {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, step=2),
        ],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "llm/retry", {"turn": 1, "step": 1}),
        ],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "step/start", {"turn": 1, "step": 1}),
            _ev(3, "llm/retry", {"turn": 1, "step": 1}),
        ],
        [_ev(1, "turn/start", {"turn": 1}), _ev(2, "step/end", {"turn": 1, "step": 1})],
        [
            _ev(1, "turn/start", {"turn": 1}),
            _ev(2, "step/start", {"turn": 1, "step": 1}),
            _ev(3, "step/end", {"turn": 1, "step": 1}),
        ],
    ):
        assert derive_turn_token_usage(invalid_seq) is None


async def test_token_meter_service_measure_pressure_offload_and_errors() -> None:
    """TokenMeter measures usage/estimated/none baselines, reprices routes, and isolates errors."""
    ctx = Context()
    reg = SessionProjectionRegistry(ctx)
    sessions_svc = SessionService(ctx)
    llm = LLMService(ctx)

    class _PricingAdapter:
        def image_request_pricing(self, _provider: str, _model: str) -> Any:
            return {
                "priceImages": lambda imgs: [
                    {
                        "visualTokens": 0 if img.get("offloaded") else 85,
                        "text": "[img]" if img.get("offloaded") else "",
                    }
                    for img in imgs
                ]
            }

    llm.register_adapter(["mock"], _PricingAdapter())
    meter_fiber = ctx.use(TokenMeter)
    await ctx.flush()
    meter: TokenMeter = ctx.tokenMeter

    assert meter.estimateMessage(
        {"role": "user", "content": [{"type": "text", "text": "x"}]}
    ) == (1 + BLOCK_OVERHEAD + ROLE_OVERHEAD)

    class _OffloadProjection:
        type = "image/offload"

        def project(self, _event: Any, _context: Any) -> dict[int, dict[str, Any]]:
            return {}

    sessions_svc.store.register_message_projection(_OffloadProjection())  # type: ignore[arg-type]
    s_unmeasured = sessions_svc.create("unmeasured")
    s_unmeasured.append("turn/start", {"turn": 1})
    s = sessions_svc.create("meter-live")
    assert meter.context_pressure(s) == {}
    empty_m = meter.measure(s)
    assert empty_m["baseline"] == {"kind": "none", "tokens": 0}
    assert empty_m["totalTokens"] == 0
    assert empty_m["logRevision"] == 0
    with pytest.raises(TypeError):
        empty_m["nodes"].append("nope")

    # Add header, step, user message with image + file, assistant message with usage
    hdr = _header(provider="mock", model="fast")
    s.append("request/header", {"header": hdr, "reason": "initial"})
    s.append(
        "request/context", {"provider": "mock", "model": "fast", "contextWindow": 16000}
    )
    s.append("step/start", {"turn": 1, "step": 1})
    u_img = s.append(
        "user/message",
        {
            "role": "user",
            "id": "u-img",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image",
                    "attachment": {
                        "attachmentId": "sha256:abcdef0123456789",
                        "mediaType": "image/png",
                        "bytes": 100,
                        "width": 10,
                        "height": 10,
                    },
                },
                {
                    "type": "file",
                    "attachment": {
                        "attachmentId": "sha256:0123456789abcdef",
                        "name": "notes.txt",
                        "bytes": 12,
                        "mediaType": "text/plain",
                    },
                },
            ],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )
    a_msg = s.append(
        "assistant/message",
        _assistant_event_payload(
            turn=1,
            step=1,
            text="saw it",
            usage={
                "inputTokens": 500,
                "outputTokens": 20,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
            },
        ),
        surface_op="append",
    )
    s.append("step/end", {"turn": 1, "step": 1})

    m1 = meter.measure(s)
    assert m1["baseline"]["kind"] == "usage"
    assert m1["baseline"]["tokens"] == 520
    assert m1["surfaceDeltaTokens"] == 0
    assert meter.context_pressure(s) == {"pressureTokens": 500, "contextWindow": 16000}

    # Image offload updates route price (visualTokens drops to 0) & re-measures cleanly
    s.append(
        "image/offload",
        {"targets": [{"seq": u_img.seq, "imageIndexes": [0]}]},
    )
    m2 = meter.measure(s)
    assert m2["surfaceTokens"] < m1["surfaceTokens"]
    assert m2["nodes"][0]["heuristicTokens"] == m1["nodes"][0]["heuristicTokens"]

    # Explicit requestHeader override changes baseline from usage to estimated
    other_hdr = _header(provider="other", model="alt", tools=[{"name": "x"}])
    m_override = meter.measure(s, other_hdr)
    assert m_override["baseline"]["kind"] == "estimated"
    assert a_msg.seq < m_override["logRevision"]

    # Appending a new request/header exercises _cached_header fallback to anchorHeader
    s.append("request/header", {"header": other_hdr, "reason": "change"})
    m_switched = meter.measure(s)
    assert m_switched["baseline"]["kind"] == "estimated"

    # Low usage anchor falls back to estimated baseline when usage < estimated_anchor
    s_low = Session.create("meter-low")
    s_low.append("step/start", {"turn": 1, "step": 1})
    s_low.append(
        "assistant/message",
        _assistant_event_payload(
            turn=1,
            step=1,
            text="abcd" * 200,
            usage={"inputTokens": 1, "outputTokens": 1},
        ),
        surface_op="append",
    )
    s_low.append("step/end", {"turn": 1, "step": 1})
    m_low = meter.measure(s_low)
    assert m_low["baseline"]["kind"] == "estimated"
    assert meter.measure(s_low, other_hdr)["baseline"]["kind"] == "estimated"

    # Assistant message without matching step/start validates surface before throwing
    s_bad_ass = Session.create("bad-ass")
    s_bad_ass.append(
        "assistant/message",
        _assistant_event_payload(turn=1, step=1, text="orphan"),
        surface_op="append",
    )
    with pytest.raises(ValueError, match="has no matching step/start event"):
        meter.measure(s_bad_ass)

    # Desynchronized headerSeq in hydrated checkpoint triggers full replay resync
    s_desync = Session.create("desync")
    s_desync.append("request/header", {"header": hdr, "reason": "initial"})
    s_desync.append("request/header", {"header": other_hdr, "reason": "change"})
    expected_desync = meter.measure(s_desync)
    desync_checkpoint = reg.checkpoint(s_desync)
    tm_entry = dict(desync_checkpoint["tokenMeasurement"])
    tm_val = dict(tm_entry["val"])
    tm_val["headerSeq"] = 0
    tm_entry["val"] = tm_val
    tampered_checkpoint = {**desync_checkpoint, "tokenMeasurement": tm_entry}

    restored_ctx = Context()
    restored_reg = SessionProjectionRegistry(restored_ctx)
    restored_fiber = restored_ctx.use(TokenMeter)
    await restored_ctx.flush()
    restored_reg.hydrate(s_desync, tampered_checkpoint, [], s_desync.seq)
    restored_meter: TokenMeter = restored_ctx.tokenMeter
    assert restored_meter.measure(s_desync) == expected_desync
    assert restored_meter.measure(s_desync) == expected_desync
    await restored_fiber.dispose()
    restored_reg.dispose()
    await restored_ctx.fiber.dispose()

    # Invalid step/start failure does not corrupt consumedEvents on retry
    s_dup = Session.create("dup-start")
    bad_start = s_dup.append("step/start", {"turn": 2, "step": 1})
    dup_start = s_dup.append("step/start", {"turn": 2, "step": 2})
    assert bad_start.seq < dup_start.seq
    with pytest.raises(ValueError, match="arrived before turn 2/step 1 ended"):
        meter.measure(s_dup)
    with pytest.raises(ValueError, match="arrived before turn 2/step 1 ended"):
        meter.measure(s_dup)

    await meter_fiber.dispose()
    reg.dispose()
    await ctx.fiber.dispose()


async def test_pre_init_synchronous_measure_and_swapped_row_probe(
    tmp_path: Path,
) -> None:
    """TokenMeter works synchronously before deferred registration settles and in swapped bundles."""
    ctx = Context()
    reg = SessionProjectionRegistry(ctx)
    original_register = reg.register
    reg_started = asyncio.Event()
    release_reg: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def slow_register(
        definition: ProjectionDefinition[Any],
    ) -> ProjectionHandle[Any]:
        handle = original_register(definition)
        if definition.key == "tokenMeasurement":
            reg_started.set()
            await release_reg
        return handle

    reg.register = slow_register  # type: ignore[method-assign]
    fiber = ctx.use(TokenMeter)
    await reg_started.wait()
    meter: TokenMeter = ctx.tokenMeter

    s = Session.create("pre-init-meter")
    s.append("request/header", {"header": _header(), "reason": "initial"})
    s.append("user/message", _user_event_payload("early"), surface_op="append")
    pre_m = meter.measure(s)
    assert pre_m["baseline"]["kind"] == "estimated"
    assert pre_m["totalTokens"] > 0

    with pytest.raises(RuntimeError, match="not initialized"):
        meter.context_pressure(s)

    release_reg.set_result(None)
    await fiber.load()
    post_m = meter.measure(s)
    assert post_m["totalTokens"] == pre_m["totalTokens"]
    await fiber.dispose()

    # Disposing before deferred init settles cleans up on completion
    reg_started2 = asyncio.Event()
    release_reg2: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def slow_register2(
        definition: ProjectionDefinition[Any],
    ) -> ProjectionHandle[Any]:
        handle = original_register(definition)
        if definition.key == "tokenMeasurement":
            reg_started2.set()
            await release_reg2
        return handle

    reg.register = slow_register2  # type: ignore[method-assign]
    fiber2 = ctx.use(TokenMeter)
    await reg_started2.wait()
    ctx.tokenMeter.dispose()
    release_reg2.set_result(None)
    await fiber2.load()
    await fiber2.dispose()

    reg.dispose()
    await ctx.fiber.dispose()

    # Swapped row probe: token-meter BEFORE session-projections via load_profile
    config_file = tmp_path / "swapped_meter.yaml"
    rows = [
        {
            "id": "token-meter",
            "name": "magitech_harness.llm.token_meter:TokenMeter",
            "config": {},
        },
        {
            "id": "session-projections",
            "name": "magitech_harness.session.projection_registry:SessionProjectionRegistry",
            "config": {},
        },
    ]
    with config_file.open("w", encoding="utf-8") as handle:
        dump_yaml(rows, handle)

    standalone = Context()
    pending_fiber = standalone.use(TokenMeter)
    await standalone.flush()
    assert pending_fiber.state != FiberState.ACTIVE
    assert not standalone.has("tokenMeter")
    await standalone.fiber.dispose()

    boot_ctx = Context()
    load_profile(boot_ctx, config_file)
    await boot_ctx.flush()
    assert boot_ctx.has("tokenMeter")
    assert boot_ctx.sessionProjections.summary_keys() == frozenset({"tokenUsage"})
    await boot_ctx.fiber.dispose()


async def test_init_failure_disposes_partial_handles() -> None:
    """A failure mid-init disposes any already-registered projection handles."""
    ctx = Context()
    reg = SessionProjectionRegistry(ctx)
    original_register = reg.register
    disposed_keys: list[str] = []

    def failing_register(
        definition: ProjectionDefinition[Any],
    ) -> ProjectionHandle[Any]:
        if definition.key == "tokenUsage":
            raise RuntimeError("registration failed")
        handle = original_register(definition)
        orig_dispose = handle.dispose

        def _track_dispose() -> None:
            disposed_keys.append(definition.key)
            orig_dispose()

        handle.dispose = _track_dispose  # type: ignore[method-assign]
        return handle

    reg.register = failing_register  # type: ignore[method-assign]
    meter = TokenMeter(ctx)
    with pytest.raises(RuntimeError, match="registration failed"):
        await meter.init()
    assert disposed_keys == ["tokenMeasurement"]
    reg.dispose()
    await ctx.fiber.dispose()


async def test_bundle_route_pricing_and_file_text_via_loader(tmp_path: Path) -> None:
    """TokenMeter resolves optional llm across isolated bundle rows regardless of order."""

    class _LoadedAdapter:
        def __init__(self, visual_tokens: int) -> None:
            self._visual_tokens = visual_tokens

        def image_request_pricing(self, _provider: str, _model: str) -> Any:
            visual_tokens = self._visual_tokens
            return {
                "priceImages": lambda imgs: [
                    {
                        "visualTokens": 0 if img.get("offloaded") else visual_tokens,
                        "text": "[off]" if img.get("offloaded") else "",
                    }
                    for img in imgs
                ]
            }

    def _build_session(provider: str = "mock", *, with_image: bool = True) -> Session:
        s = Session.create(f"bundle-{provider}-{with_image}")
        s.append(
            "request/header",
            {"header": _header(provider=provider, model="fast"), "reason": "initial"},
        )
        s.append("step/start", {"turn": 1, "step": 1})
        content: list[dict[str, Any]] = []
        if with_image:
            content.append(
                {
                    "type": "image",
                    "attachment": {
                        "attachmentId": "sha256:0011223344556677",
                        "mediaType": "image/png",
                        "bytes": 50,
                        "width": 10,
                        "height": 10,
                    },
                }
            )
        content.append(
            {
                "type": "file",
                "attachment": {
                    "attachmentId": "sha256:7766554433221100",
                    "name": "spec.md",
                    "bytes": 9,
                    "mediaType": "text/markdown",
                },
            }
        )
        s.append(
            "user/message",
            {
                "role": "user",
                "id": "u-loader",
                "content": content,
                "source": {"kind": "user"},
            },
            surface_op="append",
        )
        return s

    # 1. Full create_harness() bundle isolates each row in its own child context
    harness_ctx = create_harness()
    await harness_ctx.flush()
    harness_ctx.llm.register_adapter(["mock"], _LoadedAdapter(123))
    harness_ctx.llm.register_adapter(["mock2"], _LoadedAdapter(223))
    harness_session = _build_session("mock")
    measured_harness = harness_ctx.tokenMeter.measure(harness_session)
    measured_delta = harness_ctx.tokenMeter.measure(_build_session("mock2"))
    assert measured_delta["surfaceTokens"] - measured_harness["surfaceTokens"] == 100
    assert (
        measured_harness["nodes"][0]["tokens"]
        != (measured_harness["nodes"][0]["heuristicTokens"])
    )
    measured_file_only = harness_ctx.tokenMeter.measure(
        _build_session("mock", with_image=False)
    )
    await harness_ctx.fiber.dispose()

    # Baseline without LLMService falls back to raw heuristicTokens
    no_llm_ctx = Context()
    no_llm_reg = SessionProjectionRegistry(no_llm_ctx)
    no_llm_fiber = no_llm_ctx.use(TokenMeter)
    await no_llm_ctx.flush()
    measured_file_no_llm = no_llm_ctx.tokenMeter.measure(
        _build_session("mock", with_image=False)
    )
    assert (
        measured_file_no_llm["nodes"][0]["tokens"]
        == (measured_file_no_llm["nodes"][0]["heuristicTokens"])
    )
    assert (
        measured_file_only["nodes"][0]["tokens"]
        != (measured_file_no_llm["nodes"][0]["tokens"])
    )
    await no_llm_fiber.dispose()
    no_llm_reg.dispose()
    await no_llm_ctx.fiber.dispose()

    # 2. Swapped-row load_profile(): token-meter BEFORE session-projections AND llm
    swapped_file = tmp_path / "swapped_meter_before_llm.yaml"
    rows = [
        {
            "id": "token-meter",
            "name": "magitech_harness.llm.token_meter:TokenMeter",
            "config": {},
        },
        {
            "id": "session-projections",
            "name": "magitech_harness.session.projection_registry:SessionProjectionRegistry",
            "config": {},
        },
        {
            "id": "llm",
            "name": "magitech_harness.llm.service:LLMService",
            "config": {},
        },
    ]
    with swapped_file.open("w", encoding="utf-8") as handle:
        dump_yaml(rows, handle)

    boot_ctx = Context()
    load_profile(boot_ctx, swapped_file)
    await boot_ctx.flush()
    boot_ctx.llm.register_adapter(["mock"], _LoadedAdapter(123))
    measured_swapped = boot_ctx.tokenMeter.measure(_build_session("mock"))
    assert measured_swapped == measured_harness
    await boot_ctx.fiber.dispose()
