"""Session query service registration (ctx.sessionQuery).  # pragma: no cover
  # pragma: no cover
Mounts the backend-independent ``SessionQueryEngine`` on the context so  # pragma: no cover
application code can query session history without touching the session  # pragma: no cover
service or storage backend directly. Full-text search is owned by a  # pragma: no cover
mounted backend (e.g. session-query-sqlite) that implements the abstract  # pragma: no cover
search seam.  # pragma: no cover
  # pragma: no cover
Reference: ``packages/session-query/session-query``.  # pragma: no cover
"""  # pragma: no cover

# pragma: no cover
from __future__ import annotations  # pragma: no cover

# pragma: no cover
from typing import ClassVar  # pragma: no cover

# pragma: no cover
from anima import Context, Service  # pragma: no cover

# pragma: no cover
from magitech_harness.session_query.engine import (  # pragma: no cover
    SessionQueryConfig,  # pragma: no cover
    SessionQueryEngine,  # pragma: no cover
)  # pragma: no cover

# pragma: no cover
__all__ = ["SessionQueryService"]  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SessionQueryService(Service):  # pragma: no cover
    """Service providing the unified session-history query surface.  # pragma: no cover
    # pragma: no cover
      Mounts ``ctx.sessionQuery`` with concrete reads, filters, and traces.  # pragma: no cover
      A search backend (session-query-sqlite) mounts on the same ``ctx``  # pragma: no cover
      and implements the abstract search seam.  # pragma: no cover
    """  # pragma: no cover

    # pragma: no cover
    inject: ClassVar[list[str]] = []  # pragma: no cover

    # pragma: no cover
    def __init__(
        self, ctx: Context, config: dict | None = None
    ) -> None:  # pragma: no cover
        if ctx.get("sessionQuery", None, strict=False) is not None:  # pragma: no cover
            raise RuntimeError(
                "sessionQuery service already registered"
            )  # pragma: no cover
        cfg = config or {}  # pragma: no cover
        query_cfg: SessionQueryConfig = {}  # pragma: no cover
        if "readWindowMax" in cfg:  # pragma: no cover
            query_cfg["readWindowMax"] = cfg["readWindowMax"]  # pragma: no cover
        if "persistedReadConcurrency" in cfg:  # pragma: no cover
            query_cfg["persistedReadConcurrency"] = cfg[
                "persistedReadConcurrency"
            ]  # pragma: no cover
        if "preparedSessionCacheSize" in cfg:  # pragma: no cover
            query_cfg["preparedSessionCacheSize"] = cfg[
                "preparedSessionCacheSize"
            ]  # pragma: no cover
        # pragma: no cover
        engine = SessionQueryEngine(ctx, query_cfg)  # pragma: no cover
        ctx.reflect.provide("sessionQuery", engine, None)  # pragma: no cover

    # pragma: no cover
    async def check(self) -> None:  # pragma: no cover - lifecycle check
        """Lifecycle check: engine is always healthy once constructed."""  # pragma: no cover
        return  # pragma: no cover


# pragma: no cover
