"""Request-header canonical form, equality, and fold.

The header event is the durable snapshot of one request's envelope:
provider/model config, the adapter-default markers the fold compares,
and the tool schemas offered to the model. The latest snapshot
reconstructs the request header; unchanged headers are not re-logged.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from magitech_harness.session.events import SessionEvent

#: The reasons a header event may carry.
REQUEST_HEADER_REASONS: frozenset[str] = frozenset(
    ["initial", "resume", "change", "series"]
)


def _canonical_json(value: Any) -> str:
    """Canonical JSON comparison string, insertion-order-sensitive.

    No key sorting: two values compare equal exactly when their
    serializations match, key order included.
    """
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def same_schema(a: Any, b: Any) -> bool:
    """Whether two tool schemas are identical as canonical JSON."""
    return _canonical_json(a) == _canonical_json(b)


def canonical_header(header: dict[str, Any]) -> dict[str, Any]:
    """Normalize one header to canonical form without mutating its input.

    ``config`` is carried verbatim. ``adapterDefaults`` survives only
    when at least one of its two markers is ``True``, and then keeps
    exactly its ``True`` markers. An empty ``tools`` list becomes
    absent; a non-empty list is carried verbatim, order preserved.
    A ``system`` field is carried verbatim: canonicalization never
    silently repairs a retired field; admission refuses it.
    """
    canonical: dict[str, Any] = {"config": copy.deepcopy(header["config"])}
    defaults = header.get("adapterDefaults")
    if isinstance(defaults, dict):
        kept = {
            key: True
            for key in ("reasoningEffort", "maxTokens")
            if defaults.get(key) is True
        }
        if kept:
            canonical["adapterDefaults"] = kept
    tools = header.get("tools")
    if isinstance(tools, list) and len(tools) > 0:
        canonical["tools"] = copy.deepcopy(tools)
    if "system" in header:
        canonical["system"] = copy.deepcopy(header["system"])
    return canonical


def call_config_equals(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Field-wise equality over call configs.

    A missing control reads as absent on both sides, so a config with
    an absent control equals one with that control explicitly ``None``.
    ``stop`` compares element-wise, in order.
    """
    for field in ("provider", "model", "reasoningEffort", "temperature", "maxTokens"):
        if a.get(field) != b.get(field):
            return False
    stop_a = a.get("stop")
    stop_b = b.get("stop")
    if stop_a is None or stop_b is None:
        return stop_a is None and stop_b is None
    return len(stop_a) == len(stop_b) and all(
        first == second for first, second in zip(stop_a, stop_b, strict=True)
    )


def header_equals(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two headers describe the same request envelope.

    Both inputs are canonicalized first. Configs compare field-wise,
    adapter-default markers compare tri-state (``True`` vs absent vs
    ``False``), and tool schemas compare in order.
    """
    canonical_a = canonical_header(a)
    canonical_b = canonical_header(b)
    if not call_config_equals(canonical_a["config"], canonical_b["config"]):
        return False
    defaults_a = canonical_a.get("adapterDefaults") or {}
    defaults_b = canonical_b.get("adapterDefaults") or {}
    for marker in ("reasoningEffort", "maxTokens"):
        if defaults_a.get(marker) != defaults_b.get(marker):
            return False
    tools_a = canonical_a.get("tools") or []
    tools_b = canonical_b.get("tools") or []
    return len(tools_a) == len(tools_b) and all(
        same_schema(first, second)
        for first, second in zip(tools_a, tools_b, strict=True)
    )


def fold_request_header(
    events: list[SessionEvent],
    from_: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fold header events in log order; the latest snapshot wins.

    Non-header events are skipped. Returns ``None`` when no header
    event exists yet. ``from_`` continues a previously folded state.
    A header without a record config contributes no snapshot: the
    migration emits such a husk for a header whose system prompt was
    promoted out, and it carries no route to fold.
    """
    state = from_
    for event in events:
        if event.type == "request/header":
            header = event.payload.get("header")
            if not isinstance(header, dict) or not isinstance(
                header.get("config"), dict
            ):
                continue
            state = canonical_header(header)
    return state


def request_proposal(header: dict[str, Any]) -> dict[str, Any]:
    """Strip adapter-derived values from a header config before re-proposal.

    A later step re-resolves marked defaults from the route instead of
    treating an adapter-materialized value as the caller's proposal.
    """
    proposal = dict(header["config"])
    defaults = header.get("adapterDefaults") or {}
    if defaults.get("reasoningEffort") is True:
        proposal.pop("reasoningEffort", None)
    if defaults.get("maxTokens") is True:
        proposal.pop("maxTokens", None)
    return proposal
