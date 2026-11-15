"""Unit tests for call config structural comparison and request identity markers."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from magitech_harness.llm import deep_freeze
from magitech_harness.llm.call_config import (
    _AGENT_LOOP_REQUESTS,
    call_config_equals,
    is_agent_loop_request,
    mark_agent_loop_request,
)


def test_call_config_equals_handles_controls_bool_int_and_stop_sequences() -> None:
    base = {
        "provider": "openrouter",
        "model": "m1",
        "reasoningEffort": "high",
        "temperature": 0.5,
        "maxTokens": 100,
        "stop": ["END", "STOP"],
    }
    assert call_config_equals(base, {**base, "stop": ["END", "STOP"]})
    assert call_config_equals(
        {"provider": "p", "model": "m"}, {"provider": "p", "model": "m"}
    )
    assert not call_config_equals(base, {**base, "provider": "other"})
    assert not call_config_equals(base, {**base, "maxTokens": 200})
    assert not call_config_equals(
        {"provider": "p", "model": "m", "temperature": True},
        {"provider": "p", "model": "m", "temperature": 1},
    )
    assert not call_config_equals(
        {"provider": "p", "model": "m", "maxTokens": 1},
        {"provider": "p", "model": "m", "maxTokens": True},
    )
    assert not call_config_equals(base, {**base, "stop": None})
    assert not call_config_equals(base, {**base, "stop": ["END"]})
    assert not call_config_equals(base, {**base, "stop": ["END", "OTHER"]})


def test_agent_loop_request_identity_tracking_weakref_eviction_and_plain_dict_rejection() -> (
    None
):
    class _WeakRefMapping(Mapping[str, Any]):
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    # WeakRef-capable FrozenDict and custom Mapping are tracked without strong refs.
    frozen_req = deep_freeze({"provider": "p", "model": "m", "messages": []})
    frozen_copy = deep_freeze(dict(frozen_req))
    assert not is_agent_loop_request(frozen_req)
    assert mark_agent_loop_request(frozen_req) is frozen_req
    assert is_agent_loop_request(frozen_req)
    assert not is_agent_loop_request(frozen_copy)

    req_wr = _WeakRefMapping({"provider": "p", "model": "m", "messages": []})
    wr_ref = weakref.ref(req_wr)
    assert not is_agent_loop_request(req_wr)
    assert mark_agent_loop_request(req_wr) is req_wr
    assert is_agent_loop_request(req_wr)

    # Dropping strong references evicts weak entries automatically (zero leak).
    del frozen_req
    del req_wr
    gc.collect()
    assert wr_ref() is None
    assert len(_AGENT_LOOP_REQUESTS) == 0

    # Plain un-weakrefable dict cannot be registered (TypeError) and checks return False.
    plain_dict = {"provider": "p", "model": "m"}
    assert not is_agent_loop_request(plain_dict)
    with pytest.raises(TypeError):
        mark_agent_loop_request(plain_dict)
    assert not is_agent_loop_request(plain_dict)
