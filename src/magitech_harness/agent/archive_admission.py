"""Archive-admission seam connecting running Agent turns to Workspace checks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from anima import Context


def install_turn_archive_admission(
    ctx: Context, lookup: Callable[[str], Any]
) -> Callable[[], None]:
    """Answer Workspace session activity and stop requests for live running agents."""

    async def _on_activity(
        request: dict[str, Any], next_fn: Callable[[], Awaitable[list[Any]]]
    ) -> list[Any]:
        agent = lookup(request["sessionId"])
        active = agent is not None and agent.status == "running"
        downstream = await next_fn()
        if active:
            return [{"kind": "turn"}, *downstream]
        return downstream

    def _on_stop(request: dict[str, Any]) -> None:
        agent = lookup(request["sessionId"])
        if agent is not None and agent.status == "running":
            agent.cancel({"kind": "user"})

    dispose_activity = ctx.on("workspace/session-activity", _on_activity, prepend=True)
    dispose_stop = ctx.on("workspace/session-stop", _on_stop)

    def _dispose() -> None:
        dispose_activity()
        dispose_stop()

    return _dispose
