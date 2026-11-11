"""ApprovalService (ctx.approvals): Centralized approval broker with serialization mutex
and canonical sha256 argument digest hashing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from anima import Context, Service


@dataclass(frozen=True)
class ApprovalRequest:
    """Approval request payload bound to a tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    digest: str
    description: str | None = None


def calculate_argument_digest(arguments: dict[str, Any]) -> str:
    """Calculates canonical sha256:<hex> digest over sorted JSON representation of arguments."""
    canonical_json = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


class ApprovalService(Service):
    """Centralized approval broker managing mutex-serialized authorization requests."""

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "approvals")
        self._lock = asyncio.Lock()
        self._default_verdict = bool((config or {}).get("default_verdict", False))
        self._handler: Callable[[ApprovalRequest], Awaitable[bool]] | None = None
        self._cache: dict[str, bool] = {}

    @property
    def default_verdict(self) -> bool:
        return self._default_verdict

    @default_verdict.setter
    def default_verdict(self, val: bool) -> None:
        self._default_verdict = val

    def set_handler(
        self, handler: Callable[[ApprovalRequest], Awaitable[bool]] | None
    ) -> None:
        """Register a custom approval handler (e.g. interactive terminal prompt)."""
        self._handler = handler

    def remember(self, digest: str, verdict: bool) -> None:
        """Cache an approval verdict by canonical argument digest."""
        self._cache[digest] = verdict

    def get_cached(self, digest: str) -> bool | None:
        """Retrieve a cached verdict if present."""
        return self._cache.get(digest)

    def clear_cache(self) -> None:
        """Clears all cached approval verdicts."""
        self._cache.clear()

    async def request_approval(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        description: str | None = None,
    ) -> bool:
        """Requests authorization for a sensitive tool call under serialization mutex."""
        digest = calculate_argument_digest(arguments)

        if digest in self._cache:
            return self._cache[digest]

        req = ApprovalRequest(
            tool_name=tool_name,
            arguments=arguments,
            digest=digest,
            description=description,
        )

        async with self._lock:
            if digest in self._cache:
                return self._cache[digest]

            async def _default_resolver(request: ApprovalRequest) -> bool:
                if self._handler is not None:
                    return await self._handler(request)
                return self._default_verdict

            verdict = await self.ctx.waterfall(
                "approval/request",
                req,
                default_handler=_default_resolver,
            )
            return bool(verdict)
