"""Web capability seam and tools."""

from __future__ import annotations

from magitech_harness.web.service import (
    WebError,
    WebFetchProvider,
    WebSearchProvider,
    WebService,
)
from magitech_harness.web.service import (
    apply as apply_web,
)
from magitech_harness.web.tools import apply as apply_web_tools

__all__ = [
    "WebError",
    "WebFetchProvider",
    "WebSearchProvider",
    "WebService",
    "apply_web",
    "apply_web_tools",
]
