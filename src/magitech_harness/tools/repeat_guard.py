"""Repeat-tool reminder guard: advisory nudge out of identical tool-call loops.

A Tier 1 profile-bundle plugin. Tracks one consecutive-repeats chain per
agent (a different tracked call resets the count to 1) and, at configured
counts, prepends a sourced user message to a post-execute decision's
``additionalContexts``. Advisory only: never blocks or rewrites a result.
"""

from __future__ import annotations

import json
from typing import Any

from anima import Context, EventMode

from magitech_harness.llm.message import create_user_message

DEFAULT_THRESHOLDS = [3, 5, 8]


def _canonical(arguments: Any) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


class RepeatChain:
    """One consecutive-call chain for one agent."""

    def __init__(self, thresholds: list[int], preview_chars: int) -> None:
        if not thresholds:
            raise ValueError("repeat guard thresholds must not be empty")
        if any(t < 2 for t in thresholds):
            raise ValueError("repeat guard thresholds must be >= 2")
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("repeat guard thresholds must be unique")
        self.thresholds = sorted(thresholds)
        self.preview_chars = preview_chars
        # agent -> {key, count}; a single chain per agent (not per signature)
        self._chains: dict[Any, dict[str, Any]] = {}

    def observe(
        self, agent: Any, tool_name: str, arguments: Any
    ) -> tuple[str, int] | None:
        """Record a call; return (reminder text, count) if a threshold fired."""
        key = json.dumps([tool_name, _canonical(arguments)])
        chain = self._chains.get(agent)
        if chain is None or chain["key"] != key:
            self._chains[agent] = {"key": key, "count": 1}
            count = 1
        else:
            chain["count"] += 1
            count = chain["count"]
        if count not in self.thresholds:
            return None
        return self._message(count, tool_name, arguments), count

    def reset(self, agent: Any) -> None:
        self._chains.pop(agent, None)

    def _message(self, count: int, tool_name: str, arguments: Any) -> str:
        if count == self.thresholds[0]:
            return (
                "You are repeating the exact same tool call with identical "
                "arguments. Carefully analyze the previous result before "
                "calling again: if the task is not complete, try a different "
                "approach or different arguments instead of repeating the call."
            )
        preview = _canonical(arguments)
        if len(preview) > self.preview_chars:
            preview = preview[: self.preview_chars] + (
                f" … (+{len(preview) - self.preview_chars} more chars)"
            )
        return (
            "Repeated tool call detected:\n"
            f"- tool: {tool_name}\n"
            f"- consecutive_calls: {count}\n"
            f"- arguments: {preview}\n"
            "The repeated calls are not making progress. Do not call this tool "
            "with these exact arguments again. Inspect the latest result and "
            "choose a different action, different arguments, or finish the "
            "task if enough evidence has been gathered."
        )


def apply(
    ctx: Context,
    *,
    thresholds: list[int] | None = None,
    arguments_preview_chars: int = 500,
    **kwargs: Any,
) -> None:
    """Mount the repeat-tool reminder on tools/post-execute."""
    chain = RepeatChain(thresholds or list(DEFAULT_THRESHOLDS), arguments_preview_chars)

    async def _post_execute(exec_view: Any, result: Any, next_fn: Any) -> Any:
        # The pipeline passes a dict exec_view; the legacy Phase 2 path passes
        # (raw_string_output, call_data). Only the dict shape carries an agent.
        is_view = isinstance(exec_view, dict)
        agent = exec_view.get("agent") if is_view else None
        tool_name = exec_view.get("name", "") if is_view else ""
        observed = (
            chain.observe(agent, tool_name, exec_view.get("arguments", {}))
            if agent is not None and tool_name
            else None
        )
        decision = await next_fn(exec_view, result)
        if observed is None:
            return decision
        message, count = observed
        contexts = list(decision.get("additionalContexts") or [])
        contexts.append(
            create_user_message(
                content=[{"type": "text", "text": message}],
                source={
                    "kind": "repeat-tool-reminder",
                    "form": "notice",
                    "summary": f"{tool_name} x {count}",
                },
            )
        )
        decision["additionalContexts"] = contexts
        return decision

    async def _pre_step(decision: dict[str, Any], next_fn: Any) -> Any:
        # Reset only on a new USER prompt, not every step.
        messages = decision.get("messages") or []
        if any(
            isinstance(m, dict)
            and isinstance(m.get("source"), dict)
            and m["source"].get("kind") == "user"
            for m in messages
        ):
            agent = decision.get("agent")
            if agent is not None:  # pragma: no branch - pre-step carries agent
                chain.reset(agent)
        return await next_fn(decision)

    ctx.on("tools/post-execute", _post_execute, mode=EventMode.WATERFALL)
    ctx.on("agent/pre-step", _pre_step, mode=EventMode.WATERFALL)


apply.inject = ["tools"]  # type: ignore[attr-defined]

__all__ = ["DEFAULT_THRESHOLDS", "RepeatChain", "apply"]
