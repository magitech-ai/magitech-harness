"""In-process subagent providers (dsh spawn/fork rows).

Both providers create a child harness context through the same bundle the
parent booted, so a child carries the full base substrate rather than a
hand-built subset. ``fork`` additionally seeds the child session from the
parent's completed event prefix for cache reuse.
"""

from __future__ import annotations

from typing import Any

from magitech_harness import create_harness
from magitech_harness.subagent.service import SubagentProvider
from magitech_harness.subagent.types import SubagentSpec


async def _start(
    ctx: Any, spec: SubagentSpec, *, fork: bool, session: Any = None
) -> dict[str, Any]:
    child = create_harness()
    setattr(child, "workspace_dir", getattr(ctx, "workspace_dir", None))  # noqa: B010
    await child.flush()
    child_session = child.sessions.create()
    if fork:
        parent_session = session or getattr(ctx, "session", None)
        if parent_session is not None:
            inherited = list(parent_session.events)
            # The in-flight turn may contain an assistant tool call with no
            # result yet. Replaying it would produce an unbalanced child log,
            # so seed only through the last completed turn (dsh slice rule).
            balanced_end = 0
            for index, event in enumerate(inherited):
                if event.type == "turn/end":
                    balanced_end = index + 1
            for event in inherited[:balanced_end]:
                child_session.append(
                    event.type,
                    dict(event.payload),
                    surface_op=event.surface_op,
                    source_event_seqs=event.source_event_seqs,
                )
    return {
        "sessionId": child_session.session_id,
        "mode": "fork" if fork else "spawn",
    }


async def start_spawn(
    spec: SubagentSpec, ctx: Any, *, session: Any = None
) -> dict[str, Any]:
    """Start a fresh child context with no inherited conversation."""
    return await _start(ctx, spec, fork=False, session=session)


async def start_fork(
    spec: SubagentSpec, ctx: Any, *, session: Any = None
) -> dict[str, Any]:
    """Start a child seeded with the parent's completed turns."""
    return await _start(ctx, spec, fork=True, session=session)


def spawn_provider() -> SubagentProvider:
    """The ``spawn`` provider descriptor."""
    return SubagentProvider(
        name="spawn",
        start=start_spawn,
        capabilities=frozenset({"agentOptions", "depthLimit"}),
    )


def fork_provider() -> SubagentProvider:
    """The ``fork`` provider descriptor (inherits parent history)."""
    return SubagentProvider(
        name="fork",
        start=start_fork,
        capabilities=frozenset({"depthLimit"}),
    )


def apply_spawn(ctx: Any, config: dict[str, Any] | None = None) -> None:
    """Mount the in-process spawn provider."""
    del config
    ctx.subagents.register_provider(spawn_provider())


def apply_fork(ctx: Any, config: dict[str, Any] | None = None) -> None:
    """Mount the in-process fork provider."""
    del config
    ctx.subagents.register_provider(fork_provider())


apply_spawn.inject = ["subagents"]  # type: ignore[attr-defined]
apply_fork.inject = ["subagents"]  # type: ignore[attr-defined]

__all__ = [
    "apply_fork",
    "apply_spawn",
    "fork_provider",
    "spawn_provider",
    "start_fork",
    "start_spawn",
]
