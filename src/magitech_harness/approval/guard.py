"""Approval guard plugin: consult the approval service before mutating tools.

A Tier 1 profile-bundle plugin that registers a ``tools/pre-execute``
waterfall listener. Tools in the read-only allowlist run unguarded; every
other tool emits an ``ask`` decision, which the pipeline resolves through
the mounted approval service (session-audited, fail-closed).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from animas import Context, EventMode

# Read-only tools: these do not mutate workspace state and need no grant.
READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "grep",
        "lsp_definition",
        "lsp_references",
    }
)


def apply(
    ctx: Context, *, read_only_tools: list[str] | None = None, **kwargs: Any
) -> None:
    """Mount the mutating-tool approval guard."""
    allowlist = (
        frozenset(read_only_tools) if read_only_tools is not None else READ_ONLY_TOOLS
    )

    async def _pre_execute(decision: Any, next_fn: Any) -> Any:
        # Both call-data shapes exist: the pipeline passes a dict exec_view,
        # while the legacy Phase 2 execute path passes a ToolCallData model.
        if isinstance(decision, Mapping):
            name = decision.get("name", "")
            requires = bool(decision.get("requiresApproval", False))
        else:
            name = getattr(decision, "name", "")
            requires = False  # legacy ToolCallData carries no approval flag
        if name in allowlist or not requires:
            return await next_fn(decision)
        return {
            "kind": "ask",
            "reason": f'tool "{name}" requires approval',
        }

    ctx.on("tools/pre-execute", _pre_execute, mode=EventMode.WATERFALL)


apply.inject = ["approval", "tools"]  # type: ignore[attr-defined]

__all__ = ["READ_ONLY_TOOLS", "apply"]
