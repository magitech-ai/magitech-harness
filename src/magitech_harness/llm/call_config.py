"""Call configuration equality and loop-owned request identity markers."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NotRequired, TypedDict

from magitech_harness.llm.types import LlmResolvedModelInfo

_AGENT_LOOP_REQUESTS: weakref.WeakValueDictionary[int, Mapping[str, Any]] = (
    weakref.WeakValueDictionary()
)


class LlmCallControls(TypedDict, total=False):
    """Concrete generation controls; omitted fields take route defaults."""

    reasoningEffort: str
    temperature: float
    maxTokens: int
    stop: Sequence[str]


class LlmCallConfig(TypedDict):
    """Provider, model, reasoning effort, and sampling settings for a call."""

    provider: str
    model: str
    reasoningEffort: NotRequired[str]
    temperature: NotRequired[float]
    maxTokens: NotRequired[int]
    stop: NotRequired[Sequence[str]]


class LlmCallConfigAdapterDefaults(TypedDict, total=False):
    """Config fields materialized from adapter model metadata rather than caller input."""

    reasoningEffort: bool
    maxTokens: bool


ConfigureCall = Callable[
    [Mapping[str, Any], LlmResolvedModelInfo],
    Mapping[str, Any],
]


def call_config_equals(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare two call configuration mappings field-by-field including stop lists."""
    if (
        left.get("provider") != right.get("provider")
        or left.get("model") != right.get("model")
        or left.get("reasoningEffort") != right.get("reasoningEffort")
        or left.get("temperature") != right.get("temperature")
        or left.get("maxTokens") != right.get("maxTokens")
    ):
        return False
    # bool != int distinction for temperature/maxTokens
    if (
        type(left.get("temperature")) is bool or type(right.get("temperature")) is bool
    ) and type(left.get("temperature")) is not type(right.get("temperature")):
        return False
    if (
        type(left.get("maxTokens")) is bool or type(right.get("maxTokens")) is bool
    ) and type(left.get("maxTokens")) is not type(right.get("maxTokens")):
        return False
    stop_a = left.get("stop")
    stop_b = right.get("stop")
    if stop_a is None or stop_b is None:
        return stop_a is stop_b
    return len(stop_a) == len(stop_b) and list(stop_a) == list(stop_b)


def mark_agent_loop_request[T: Mapping[str, Any]](request: T) -> T:
    """Mark one exact request mapping as assembled by the agent loop."""
    _AGENT_LOOP_REQUESTS[id(request)] = request
    return request


def is_agent_loop_request(request: Mapping[str, Any]) -> bool:
    """Return True only for the exact marked request object."""
    return _AGENT_LOOP_REQUESTS.get(id(request)) is request


callConfigEquals = call_config_equals
markAgentLoopRequest = mark_agent_loop_request
isAgentLoopRequest = is_agent_loop_request
