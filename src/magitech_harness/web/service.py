"""Web capability seam (``ctx.web``) for model-facing search and fetch.

Ports dsh's ``@deepseek-ai/dsh-web`` at ``packages/web/web/src``: a
provider-neutral seam with ``searchReady``/``fetchReady`` admission and
``search``/``fetch`` operations. The base bundle mounts the local HTTP fetch
provider; search providers are opt-in, so an unconfigured search call fails
with ``WEB_SEARCH_UNAVAILABLE`` rather than silently returning nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx
from animas import Context, Service

from magitech_harness.llm.error import HarnessError

DEFAULT_FETCH_TIMEOUT_SECONDS = 30.0
DEFAULT_SEARCH_TIMEOUT_MS = 60_000
DEFAULT_MAX_OUTPUT_CHARS = 50_000
DEFAULT_MAX_RESULTS = 5
PRIVATE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class WebError(HarnessError):
    """Stable failure taxonomy for the web seam."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


def is_public_http_url(url: str) -> bool:
    """Only public http(s) URLs are accepted by the anonymous fetcher."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    return not (not parsed.hostname or parsed.hostname in PRIVATE_HOSTS)


FetchProvider = Callable[[str, float], Awaitable[dict[str, Any]]]
SearchProvider = Callable[[list[str], int, float], Awaitable[dict[str, Any]]]


def _html_to_text(body: str) -> str:
    """Strip script/style blocks and tags from an HTML response."""
    import html
    import re

    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
    text = re.sub(r"(?s)<[^>]+>", " ", body)
    return html.unescape(re.sub(r"[ \t]+", " ", text)).strip()


class WebService(Service):
    """``ctx.web``: search and fetch with per-route readiness."""

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, "web")
        raw = dict(config or {})
        self.fetch_timeout = float(
            raw.get("fetch_timeout_seconds", DEFAULT_FETCH_TIMEOUT_SECONDS)
        )
        self.search_timeout_ms = int(
            raw.get("search_timeout_ms", DEFAULT_SEARCH_TIMEOUT_MS)
        )
        self.max_output_chars = int(
            raw.get("max_output_chars", DEFAULT_MAX_OUTPUT_CHARS)
        )
        self.max_results = int(raw.get("max_results", DEFAULT_MAX_RESULTS))
        self._fetch_providers: dict[str, FetchProvider] = {}
        self._search_providers: dict[str, SearchProvider] = {}

    def register_fetch(self, name: str, provider: FetchProvider) -> Callable[[], None]:
        self._fetch_providers[name] = provider

        def dispose() -> None:
            if self._fetch_providers.get(name) is provider:
                self._fetch_providers.pop(name)

        return dispose

    def register_search(
        self, name: str, provider: SearchProvider
    ) -> Callable[[], None]:
        self._search_providers[name] = provider

        def dispose() -> None:
            if self._search_providers.get(name) is provider:
                self._search_providers.pop(name)

        return dispose

    async def fetch_ready(self, provider: str | None = None) -> bool:
        """At least one fetch provider must be registered."""
        del provider
        return bool(self._fetch_providers)

    async def search_ready(self, provider: str | None = None) -> bool:
        """A search provider must be registered for the requested route."""
        if provider is not None:
            return provider in self._search_providers
        return bool(self._search_providers)

    async def fetch(
        self, request: Mapping[str, Any], provider: str | None = None
    ) -> dict[str, Any]:
        """Fetch one public http(s) URL and return bounded decoded text."""
        url = request.get("url")
        if not isinstance(url, str) or not is_public_http_url(url):
            raise WebError(
                "web_fetch requires a public http(s) URL", "WEB_FETCH_INVALID_URL"
            )
        if not self._fetch_providers:
            raise WebError(
                "no web fetch provider is registered", "WEB_FETCH_UNAVAILABLE"
            )
        backend = (
            self._fetch_providers.get(provider)
            if provider is not None
            else next(iter(self._fetch_providers.values()))
        )
        if backend is None:
            raise WebError(
                f'no web fetch provider named "{provider}"', "WEB_FETCH_UNAVAILABLE"
            )
        result = await backend(url, self.fetch_timeout)
        body = result.get("body")
        if not isinstance(body, dict) or body.get("kind") not in ("html", "text"):
            raise WebError(
                "fetch provider returned an invalid body", "WEB_FETCH_INVALID_RESULT"
            )
        content = str(body.get("content", ""))
        truncated = len(content) > self.max_output_chars
        return {
            "url": str(result.get("url", url)),
            "statusCode": int(result.get("statusCode", 200)),
            "body": {
                "kind": body["kind"],
                "content": content[: self.max_output_chars],
            },
            "truncated": bool(truncated),
        }

    async def search(
        self, queries: list[str], provider: str | None = None
    ) -> dict[str, Any]:
        """Run one or more queries through a registered search provider."""
        if not queries or not all(isinstance(q, str) and q.strip() for q in queries):
            raise WebError(
                "queries must be non-empty strings", "WEB_SEARCH_INVALID_QUERY"
            )
        selected = provider or next(iter(self._search_providers), None)
        backend = self._search_providers.get(selected) if selected else None
        if backend is None:
            raise WebError(
                "no web search provider is registered", "WEB_SEARCH_UNAVAILABLE"
            )
        result = await backend(queries, self.max_results, self.search_timeout_ms / 1000)
        sources = result.get("sources")
        if not isinstance(sources, list):
            raise WebError(
                "search provider returned invalid sources", "WEB_SEARCH_INVALID_RESULT"
            )
        return {
            "content": result.get("content"),
            "sources": sources,
            "truncated": bool(result.get("truncated", False)),
        }


async def http_fetch(url: str, timeout: float) -> dict[str, Any]:
    """Anonymous HTTP(S) fetcher with a bounded timeout."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
    kind = "html" if "html" in response.headers.get("content-type", "") else "text"
    body = _html_to_text(response.text) if kind == "html" else response.text
    return {
        "url": url,
        "statusCode": response.status_code,
        "body": {"kind": kind, "content": body},
        "truncated": False,
    }


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> WebService:
    """Mount ``ctx.web`` and register the anonymous HTTP fetch provider."""
    service = WebService(ctx, config)
    service.register_fetch("http", http_fetch)
    return service


apply.inject = []  # type: ignore[attr-defined]

__all__ = [
    "DEFAULT_FETCH_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_SEARCH_TIMEOUT_MS",
    "PRIVATE_HOSTS",
    "WebError",
    "WebService",
    "apply",
    "http_fetch",
    "is_public_http_url",
]
