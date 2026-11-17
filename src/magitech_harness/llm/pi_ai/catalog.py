"""Model catalog and route resolution for the pi-ai provider."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, cast

from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.pi_ai.provider_data import LANES, SUPPORTED_APIS
from magitech_harness.llm.types import (
    LlmModelContext,
    LlmModelInfo,
    LlmModelReasoningInfo,
    LlmProviderInfo,
    LlmReasoningEffortInfo,
    LlmResolvedModelInfo,
    ModelModality,
)

THINKING_LEVELS: tuple[str, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

DEFAULT_CONTEXT_WINDOW = 1_000_000
DEFAULT_MAX_TOKENS = 128_000
DEFAULT_INPUT: tuple[ModelModality, ...] = ("text",)


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model descriptor used for request dispatch and metadata."""

    provider: str
    id: str
    name: str
    api: str
    base_url: str
    input: tuple[ModelModality, ...]
    context_window: int
    max_tokens: int
    reasoning: bool
    thinking_level_map: dict[str, Any] | None = None
    compat: dict[str, Any] = field(default_factory=dict)


def _lane_models(lane: str) -> Mapping[str, Mapping[str, Any]]:
    path = resources.files("magitech_harness.llm.pi_ai").joinpath(
        "data", f"{lane}.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return cast(Mapping[str, Mapping[str, Any]], raw.get("models", {}))


def load_lane_catalog(lane: str) -> dict[str, ModelSpec]:
    """Load vendored model specs for one supported lane."""
    if lane not in LANES:
        return {}
    specs: dict[str, ModelSpec] = {}
    for model_id, item in _lane_models(lane).items():
        api = str(item.get("api", ""))
        base_url = str(item.get("baseUrl", ""))
        if api not in SUPPORTED_APIS or not base_url:
            continue
        modalities = (
            tuple(m for m in (item.get("input") or ["text"]) if m in ("text", "image"))
            or DEFAULT_INPUT
        )
        thinking_map = item.get("thinkingLevelMap")
        specs[model_id] = ModelSpec(
            provider=lane,
            id=str(item.get("id", model_id)),
            name=str(item.get("name", model_id)),
            api=api,
            base_url=base_url,
            input=cast(tuple[ModelModality, ...], modalities),
            context_window=int(item.get("contextWindow") or DEFAULT_CONTEXT_WINDOW),
            max_tokens=int(item.get("maxTokens") or DEFAULT_MAX_TOKENS),
            reasoning=bool(item.get("reasoning", False)),
            thinking_level_map=(
                dict(thinking_map) if isinstance(thinking_map, Mapping) else None
            ),
            compat=dict(item.get("compat") or {}),
        )
    return specs


_LANE_CATALOGS: dict[str, dict[str, ModelSpec]] = {}


def lane_catalog(lane: str) -> dict[str, ModelSpec]:
    """Return the cached vendored catalog for a lane."""
    cached = _LANE_CATALOGS.get(lane)
    if cached is None:
        cached = load_lane_catalog(lane)
        _LANE_CATALOGS[lane] = cached
    return cached


def provider_info(provider: str, display_name: str | None = None) -> LlmProviderInfo:
    """Build display metadata for a provider route."""
    return {"id": provider, "name": display_name or provider}


def model_info(provider: str, spec: ModelSpec) -> LlmModelInfo:
    """Build catalog listing metadata for one model."""
    info: LlmModelInfo = {
        "provider": provider,
        "id": spec.id,
        "name": spec.name,
    }
    if spec.input:
        info["inputModalities"] = list(spec.input)
    return info


def supported_thinking_levels(spec: ModelSpec) -> list[str]:
    """Resolve the selectable reasoning levels of one model."""
    if not spec.reasoning:
        return ["off"]
    level_map = spec.thinking_level_map
    levels: list[str] = []
    for level in THINKING_LEVELS:
        mapped = level_map.get(level) if level_map is not None else None
        if mapped is None and level_map is not None and level in level_map:
            continue
        if level in ("xhigh", "max") and level not in (level_map or {}):
            continue
        levels.append(level)
    return levels or ["off"]


def _reasoning_info(
    spec: ModelSpec, default_effort: str | None
) -> LlmModelReasoningInfo | None:
    if not spec.reasoning:
        return None
    levels = supported_thinking_levels(spec)
    efforts: list[LlmReasoningEffortInfo] = [
        {"id": level, "name": level.capitalize()} for level in levels
    ]
    info: LlmModelReasoningInfo = {"efforts": efforts}
    if default_effort is not None and default_effort in levels:
        info["defaultEffort"] = default_effort
    return info


def resolve_reasoning_effort(spec: ModelSpec, effort: str | None) -> str | None:
    """Validate a requested reasoning effort against a model's capabilities."""
    if effort is None:
        return None
    if effort not in supported_thinking_levels(spec):
        raise LlmError(
            f'pi-ai provider "{spec.provider}" model "{spec.id}" does not support '
            f'reasoning effort "{effort}"',
            "UNSUPPORTED_REASONING_EFFORT",
        )
    return effort


def resolved_model_info(
    provider: str,
    spec: ModelSpec,
    default_effort: str | None = None,
) -> LlmResolvedModelInfo:
    """Build exact-route metadata for one resolved model."""
    info: LlmResolvedModelInfo = {
        "provider": provider,
        "id": spec.id,
        "name": spec.name,
        "inputModalities": list(spec.input),
        "context": cast(LlmModelContext, {"contextWindow": spec.context_window}),
    }
    reasoning = _reasoning_info(spec, default_effort)
    if reasoning is not None:
        info["reasoning"] = reasoning
    return info


def _merge_model(
    provider: str,
    base: ModelSpec | None,
    configured: Mapping[str, Any],
    *,
    route_api: str | None,
    route_base_url: str | None,
    default_input: Sequence[ModelModality],
) -> ModelSpec:
    model_id = str(configured.get("id") or (base.id if base else ""))
    if not model_id:
        raise LlmError(
            f'pi-ai provider "{provider}" has a model with an empty id',
            "INVALID_CONFIG",
        )
    api = str(configured.get("api") or route_api or (base.api if base else ""))
    if not api:
        raise LlmError(
            f'pi-ai provider "{provider}" model "{model_id}" needs an api; set the'
            " route's api to the wire protocol its endpoint speaks",
            "INVALID_CONFIG",
        )
    if api not in SUPPORTED_APIS:
        raise LlmError(
            f'pi-ai provider "{provider}" model "{model_id}" uses unsupported api'
            f' "{api}"; supported protocols are'
            f" {', '.join(sorted(SUPPORTED_APIS))}",
            "INVALID_CONFIG",
        )
    base_url = str(
        configured.get("baseURL")
        or configured.get("baseUrl")
        or route_base_url
        or (base.base_url if base else "")
    )
    if not base_url:
        raise LlmError(
            f'pi-ai provider "{provider}" model "{model_id}" needs a baseURL',
            "INVALID_CONFIG",
        )
    name = str(configured.get("name") or (base.name if base else model_id))
    context_window = int(
        configured.get("contextWindow")
        or (base.context_window if base else DEFAULT_CONTEXT_WINDOW)
    )
    max_tokens = int(
        configured.get("maxTokens") or (base.max_tokens if base else DEFAULT_MAX_TOKENS)
    )
    input_raw = configured.get("input")
    if input_raw is None:
        modalities = base.input if base else tuple(default_input)
    else:
        modalities = tuple(m for m in input_raw if m in ("text", "image")) or tuple(
            default_input
        )
    reasoning = base.reasoning if base else False
    thinking_map = (
        dict(base.thinking_level_map) if base and base.thinking_level_map else None
    )
    reasoning_efforts = configured.get("reasoningEfforts")
    if reasoning_efforts is False:
        reasoning = False
        thinking_map = None
    elif isinstance(reasoning_efforts, Mapping):
        reasoning = True
        thinking_map = {
            k: v for k, v in reasoning_efforts.items() if k in THINKING_LEVELS
        }
    compat = dict(base.compat) if base else {}
    if isinstance(configured.get("compat"), Mapping):
        compat.update(cast(Mapping[str, Any], configured["compat"]))
    return ModelSpec(
        provider=provider,
        id=model_id,
        name=name,
        api=api,
        base_url=base_url.rstrip("/"),
        input=cast(tuple[ModelModality, ...], modalities),
        context_window=context_window,
        max_tokens=max_tokens,
        reasoning=reasoning,
        thinking_level_map=thinking_map,
        compat=compat,
    )


def _configured_models(
    provider: str,
    configured: Sequence[Any],
    vendored: Mapping[str, ModelSpec],
    *,
    route_api: str | None,
    route_base_url: str | None,
    default_input: Sequence[ModelModality],
) -> dict[str, ModelSpec]:
    """Resolve the models a route declares outright."""
    resolved: dict[str, ModelSpec] = {}
    for entry in configured:
        if not isinstance(entry, Mapping):
            raise LlmError(
                f'pi-ai provider "{provider}" has a malformed model entry',
                "INVALID_CONFIG",
            )
        model_id = str(entry.get("id", ""))
        spec = _merge_model(
            provider,
            vendored.get(model_id),
            entry,
            route_api=route_api,
            route_base_url=route_base_url,
            default_input=default_input,
        )
        if spec.id in resolved:
            raise LlmError(
                f'pi-ai provider "{provider}" lists model "{spec.id}" more than once',
                "INVALID_CONFIG",
            )
        resolved[spec.id] = spec
    return resolved


def _overridden_models(
    provider: str,
    vendored: Mapping[str, ModelSpec],
    overrides: Mapping[str, Any],
    *,
    route_api: str | None,
    route_base_url: str | None,
    default_input: Sequence[ModelModality],
) -> dict[str, ModelSpec]:
    """Resolve a vendored catalog with per-model customizations applied."""
    resolved = dict(vendored)
    for model_id, override in overrides.items():
        if model_id not in resolved:
            raise LlmError(
                f'pi-ai provider "{provider}" names model "{model_id}" in'
                " modelOverrides, but the catalog does not describe it",
                "INVALID_CONFIG",
            )
        if not isinstance(override, Mapping):
            raise LlmError(
                f'pi-ai provider "{provider}" has a malformed modelOverride for'
                f' "{model_id}"',
                "INVALID_CONFIG",
            )
        resolved[model_id] = _merge_model(
            provider,
            resolved[model_id],
            {"id": model_id, **override},
            route_api=route_api,
            route_base_url=route_base_url,
            default_input=default_input,
        )
    return resolved


def resolve_route_models(
    provider: str,
    profile: Mapping[str, Any],
) -> dict[str, ModelSpec]:
    """Resolve the effective model catalog for one configured route."""
    vendored = lane_catalog(provider)
    raw_api = profile.get("api")
    route_api = str(raw_api) if raw_api else None
    raw_base_url = profile.get("baseURL") or profile.get("baseUrl")
    route_base_url = str(raw_base_url) if raw_base_url else None
    default_input = cast(
        Sequence[ModelModality], tuple(profile.get("defaultInput") or DEFAULT_INPUT)
    )
    configured = profile.get("models")
    overrides = profile.get("modelOverrides") or {}

    resolved: dict[str, ModelSpec] = {}
    if isinstance(configured, list) and configured:
        resolved = _configured_models(
            provider,
            configured,
            vendored,
            route_api=route_api,
            route_base_url=route_base_url,
            default_input=default_input,
        )
    elif vendored:
        if isinstance(overrides, Mapping) and overrides:
            resolved = _overridden_models(
                provider,
                vendored,
                overrides,
                route_api=route_api,
                route_base_url=route_base_url,
                default_input=default_input,
            )
        else:
            resolved = dict(vendored)
    elif isinstance(overrides, Mapping) and overrides:
        raise LlmError(
            f'pi-ai provider "{provider}" sets modelOverrides, but the catalog does'
            " not describe this route; declare its models instead",
            "INVALID_CONFIG",
        )

    if route_api or route_base_url:
        for model_id, spec in list(resolved.items()):
            resolved[model_id] = _merge_model(
                provider,
                spec,
                {"id": model_id},
                route_api=route_api,
                route_base_url=route_base_url,
                default_input=default_input,
            )

    if not resolved:
        raise LlmError(
            f'pi-ai provider "{provider}" resolves no models; declare them in the'
            " route's models list",
            "INVALID_CONFIG",
        )
    return resolved
