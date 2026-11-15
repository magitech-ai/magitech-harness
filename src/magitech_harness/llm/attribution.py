"""Public application identity and User-Agent attribution headers."""

from __future__ import annotations

from importlib import metadata
from typing import TypedDict


class AppIdentity(TypedDict):
    """Non-secret application metadata sent on provider HTTP requests."""

    product: str
    version: str
    url: str


APP_IDENTITY: AppIdentity = {
    "product": "magitech-harness",
    "version": metadata.version("magitech-harness"),
    "url": "https://github.com/magitech-ai/magitech-harness",
}


def user_agent(identity: AppIdentity = APP_IDENTITY) -> str:
    """Format RFC 9110 product/version (+url) User-Agent string."""
    return f"{identity['product']}/{identity['version']} (+{identity['url']})"


def attribution_headers(identity: AppIdentity = APP_IDENTITY) -> dict[str, str]:
    """Return lowercase HTTP headers identifying the harness."""
    return {"user-agent": user_agent(identity)}


userAgent = user_agent
attributionHeaders = attribution_headers
