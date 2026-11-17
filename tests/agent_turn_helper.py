"""Test helper: run one turn through the unified (Phase 3) agent driver.

Replaces the retired legacy ``ctx.agent_loop.run_turn`` / ``stream_turn``
calls in tests. Uses the same driver as ``run``/``repl``/``headless`` so
tests exercise the one live code path.
"""

from __future__ import annotations

from typing import Any, ClassVar

from magitech_harness.cli import summarize_headless_outcome
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.service import MockLlmAdapter, MockLLMProvider


class MockProviderComponent:
    """Register the offline mock provider AND adapter for the unified driver."""

    inject: ClassVar[list[str]] = ["llm"]

    def __init__(self, ctx: Any) -> None:
        ctx.llm.register_provider("mock", MockLLMProvider())
        ctx.llm.register_adapter(["mock"], MockLlmAdapter())


async def run_agent_turn(
    ctx: Any,
    prompt: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
) -> Any:
    """Run one turn and return a simple result with ``text`` and ``session``."""
    provider, model_name = _resolve_route(ctx, model)
    options: dict[str, Any] = {
        "agentOptions": {"provider": provider, "model": model_name},
    }
    if session_id is not None:
        options["sessionId"] = session_id
    handle = await ctx.agents.create(options)
    agent = handle["agent"]
    dispose = handle["dispose"]
    session = agent.session
    first_seq = len(session.events)
    agent.followup(
        create_user_message(
            content=[{"type": "text", "text": prompt}],
            source={"kind": "user"},
        )
    )
    await agent.when_idle()
    outcome = summarize_headless_outcome(session, first_seq)
    return _TurnResult(
        text=outcome.text,
        reason=outcome.reason,
        session=session,
        agent=agent,
        dispose=dispose,
    )


async def stream_agent_turn(
    ctx: Any,
    prompt: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Run one turn, collecting stream frames from the unified driver's bus."""
    provider, model_name = _resolve_route(ctx, model)
    options: dict[str, Any] = {
        "agentOptions": {"provider": provider, "model": model_name},
    }
    if session_id is not None:
        options["sessionId"] = session_id
    frames: list[dict[str, Any]] = []

    def _on_stream(payload: dict[str, Any]) -> None:
        frames.append(payload.get("frame") or {})

    stop = ctx.on("agent/assistant-stream", _on_stream)
    try:
        handle = await ctx.agents.create(options)
        agent = handle["agent"]
        agent.followup(
            create_user_message(
                content=[{"type": "text", "text": prompt}],
                source={"kind": "user"},
            )
        )
        await agent.when_idle()
    finally:
        stop()
    return frames, agent.session


def _resolve_route(ctx: Any, model: str | None) -> tuple[str, str]:
    if model and "/" in model:
        provider, name = model.split("/", 1)
        return provider, name
    registered = ctx.llm.list_providers()
    provider = registered[0]["id"] if registered else "mock"
    return provider, model or "mock"


class _TurnResult:
    """Minimal run_turn-shaped result for rewritten tests."""

    def __init__(
        self,
        text: str,
        reason: dict[str, Any] | None,
        session: Any,
        agent: Any,
        dispose: Any,
    ) -> None:
        self.text = text
        self.reason = reason
        self.session = session
        self.agent = agent
        self.dispose = dispose

    async def close(self) -> None:
        """Release the agent's ownership of the session."""
        result = self.dispose()
        if result is not None and hasattr(result, "__await__"):
            await result

    def __str__(self) -> str:
        return self.text
