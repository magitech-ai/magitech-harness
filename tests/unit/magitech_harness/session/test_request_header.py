"""Unit tests for request-header canonical form, equality, and fold."""

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.request_header import (
    REQUEST_HEADER_REASONS,
    call_config_equals,
    canonical_header,
    fold_request_header,
    header_equals,
    request_proposal,
    same_schema,
)


def _header(
    *,
    config: dict | None = None,
    tools: list | None = None,
    adapter_defaults: dict | None = None,
    extra: dict | None = None,
) -> dict:
    header: dict = {
        "config": config if config is not None else {"provider": "p", "model": "m"}
    }
    if tools is not None:
        header["tools"] = tools
    if adapter_defaults is not None:
        header["adapterDefaults"] = adapter_defaults
    if extra:
        header.update(extra)
    return header


def _event(seq: int, type_: str, payload: dict) -> SessionEvent:
    return SessionEvent(
        id=f"e:{seq}",
        session_id="s",
        type=type_,
        timestamp=float(seq),
        payload=payload,
        seq=seq,
    )


def test_canonical_header_drops_empty_tools_and_keeps_nonempty() -> None:
    assert "tools" not in canonical_header(_header(tools=[]))
    tools = [{"type": "function", "function": {"name": "t"}}]
    canonical = canonical_header(_header(tools=tools))
    assert canonical["tools"] == tools
    assert canonical["tools"] is not tools
    tools[0]["function"]["name"] = "changed"
    assert canonical["tools"][0]["function"]["name"] == "t"


def test_canonical_header_adapter_defaults_keep_only_true_markers() -> None:
    assert canonical_header(
        _header(adapter_defaults={"reasoningEffort": True, "maxTokens": False})
    )["adapterDefaults"] == {"reasoningEffort": True}
    assert canonical_header(
        _header(adapter_defaults={"reasoningEffort": True, "maxTokens": True})
    )["adapterDefaults"] == {"reasoningEffort": True, "maxTokens": True}
    assert "adapterDefaults" not in canonical_header(
        _header(adapter_defaults={"reasoningEffort": False, "maxTokens": False})
    )
    assert "adapterDefaults" not in canonical_header(
        _header(adapter_defaults={"reasoningEffort": None, "other": True})
    )


def test_canonical_header_config_detached_and_system_carried() -> None:
    header = _header(extra={"system": "old", "unknown": 1})
    canonical = canonical_header(header)
    assert canonical["config"] == {"provider": "p", "model": "m"}
    assert canonical["config"] is not header["config"]
    assert canonical["system"] == "old"
    assert "unknown" not in canonical
    canonical["config"]["model"] = "changed"
    assert header["config"]["model"] == "m"


def test_call_config_equals_field_wise_and_stop_order() -> None:
    base = {"provider": "p", "model": "m", "temperature": 0.5}
    assert call_config_equals(base, dict(base))
    assert call_config_equals(
        {"provider": "p", "model": "m"},
        {"provider": "p", "model": "m", "temperature": None},
    )
    assert not call_config_equals(base, {**base, "temperature": 0.7})
    assert not call_config_equals(base, {**base, "model": "n"})
    assert call_config_equals(
        {**base, "stop": ["a", "b"]}, {**base, "stop": ["a", "b"]}
    )
    assert not call_config_equals(
        {**base, "stop": ["a", "b"]}, {**base, "stop": ["b", "a"]}
    )
    assert not call_config_equals({**base, "stop": ["a"]}, base)
    assert call_config_equals(base, base)


def test_same_schema_is_insertion_order_sensitive() -> None:
    assert same_schema({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert not same_schema({"a": 1, "b": 2}, {"b": 2, "a": 1})


def test_header_equals_config_defaults_and_tool_order() -> None:
    tools = [
        {"type": "function", "function": {"name": "a"}},
        {"type": "function", "function": {"name": "b"}},
    ]
    assert header_equals(_header(tools=tools), _header(tools=list(tools)))
    assert not header_equals(_header(tools=tools), _header(tools=list(reversed(tools))))
    assert not header_equals(
        _header(config={"provider": "p", "model": "m"}),
        _header(config={"provider": "p", "model": "n"}),
    )
    # Tri-state adapter defaults: True differs from absent; a dropped
    # False marker compares equal to one that was never present.
    assert not header_equals(
        _header(adapter_defaults={"reasoningEffort": True}),
        _header(),
    )
    assert header_equals(
        _header(adapter_defaults={"reasoningEffort": False}),
        _header(),
    )
    assert header_equals(
        _header(adapter_defaults={"maxTokens": True}),
        _header(adapter_defaults={"maxTokens": True}),
    )


def test_fold_request_header_latest_wins_and_skips_non_headers() -> None:
    first = _header(config={"provider": "p", "model": "one"})
    second = _header(config={"provider": "p", "model": "two"}, tools=[])
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _event(1, "request/header", {"header": first, "reason": "initial"}),
        _event(2, "request/context", {"provider": "p", "model": "one"}),
        _event(3, "request/header", {"header": second, "reason": "change"}),
    ]
    folded = fold_request_header(events)
    assert folded == {"config": {"provider": "p", "model": "two"}}
    assert fold_request_header(events[:1]) is None
    assert fold_request_header([]) is None


def test_fold_request_header_continues_from_previous_state() -> None:
    first = _header(config={"provider": "p", "model": "one"})
    second = _header(config={"provider": "p", "model": "two"})
    events = [
        _event(0, "request/header", {"header": first, "reason": "initial"}),
        _event(1, "request/header", {"header": second, "reason": "change"}),
    ]
    state = fold_request_header(events[:1])
    assert fold_request_header(events[1:], from_=state) == {
        "config": {"provider": "p", "model": "two"}
    }
    assert fold_request_header([], from_=state) == {
        "config": {"provider": "p", "model": "one"}
    }


def test_request_proposal_strips_marked_defaults_only() -> None:
    header = _header(
        config={
            "provider": "p",
            "model": "m",
            "reasoningEffort": "high",
            "maxTokens": 100,
            "temperature": 0.5,
        },
        adapter_defaults={"reasoningEffort": True},
    )
    assert request_proposal(header) == {
        "provider": "p",
        "model": "m",
        "maxTokens": 100,
        "temperature": 0.5,
    }
    assert request_proposal(_header()) == {"provider": "p", "model": "m"}
    max_tokens_header = _header(
        config={"provider": "p", "model": "m", "maxTokens": 100},
        adapter_defaults={"maxTokens": True},
    )
    assert request_proposal(max_tokens_header) == {"provider": "p", "model": "m"}


def test_request_header_reasons_are_the_four_taxonomy_values() -> None:
    assert REQUEST_HEADER_REASONS == {"initial", "resume", "change", "series"}
