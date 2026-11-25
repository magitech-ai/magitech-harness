"""Web capability seam (``ctx.web``) for model-facing search and fetch.

Ports dsh's ``@deepseek-ai/dsh-web`` at ``packages/web/web/src``: a
provider-neutral seam with ``registerSearchProvider``/``registerFetchProvider``,
execution-time route selection, ``searchReady``/``fetchReady`` admission,
``listProviders``, ``web/providers-changed`` events and a stable ``WEB_*``
error taxonomy. The base bundle mounts the anonymous HTTP fetch provider;
search providers are opt-in, so an unconfigured search call fails with
``WEB_PROVIDER_UNAVAILABLE`` rather than silently returning nothing.
"""

from __future__ import annotations

import codecs
import os
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urljoin, urlparse

import httpx
from animas import Context, Service

from magitech_harness.llm.error import HarnessError

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

MAX_URL_LENGTH = 2048
DEFAULT_MAX_RESPONSE_BYTES = 5_000_000
DEFAULT_MAX_BODY_CHARS = 100_000
DEFAULT_TIMEOUT_MS = 30_000
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_USER_AGENT = "magitech-harness/0.0.1"
DEEPSEEK_MODEL_ROUTES = frozenset({"deepseek-official", "deepseek-account"})
DEEPSEEK_ROUTE_SEARCH_PROVIDER = "deepseek-official"
OTHER_ROUTE_SEARCH_PROVIDER = "deepseek-search-api"
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class WebError(HarnessError):
    """Stable, machine-routable failure taxonomy for the web seam."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message, code)


class WebSearchProvider(Protocol):
    """A search backend registered into ``ctx.web``."""

    id: str
    title: str | None
    description: str | None

    def available(self) -> bool: ...

    async def ready(self, llm_provider: str | None) -> bool: ...

    async def search(
        self, request: dict[str, Any], signal: Any = None
    ) -> dict[str, Any]: ...


class WebFetchProvider(Protocol):
    """A fetch backend registered into ``ctx.web``."""

    id: str
    title: str | None
    description: str | None

    def available(self) -> bool: ...

    async def ready(self, llm_provider: str | None) -> bool: ...

    async def fetch(
        self, request: dict[str, Any], signal: Any = None
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Anonymous HTTP(S) fetch provider (ports dsh-web-fetch-http)
# ---------------------------------------------------------------------------


def validate_fetch_url(input_url: Any) -> str:
    """Validate one URL against the provider's pre-network policy."""
    if not isinstance(input_url, str):
        raise WebError("invalid URL", "WEB_INVALID_URL")
    if len(input_url) > MAX_URL_LENGTH:
        raise WebError(
            f"URL exceeds the maximum length of {MAX_URL_LENGTH}", "WEB_INVALID_URL"
        )
    parsed = urlparse(input_url)
    if parsed.scheme not in ("http", "https"):
        raise WebError(
            f'unsupported URL scheme "{parsed.scheme}" (only http and https are allowed)',
            "WEB_INVALID_URL",
        )
    if parsed.username or parsed.password:
        raise WebError("credentials in URLs are not allowed", "WEB_BLOCKED_URL")
    if not parsed.hostname:
        raise WebError(f"invalid URL: {input_url}", "WEB_INVALID_URL")
    return input_url


def is_same_origin(left: str, right: str) -> bool:
    """Whether two URLs share scheme, hostname and port."""
    a, b = urlparse(left), urlparse(right)
    return (
        a.scheme == b.scheme
        and a.hostname == b.hostname
        and (a.port or (443 if a.scheme == "https" else 80))
        == (b.port or (443 if b.scheme == "https" else 80))
    )


def classify_content_type(content_type: str | None) -> str | None:
    """Classify a response Content-Type into a decodable body kind."""
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime in ("text/html", "application/xhtml+xml"):
        return "html"
    if (
        mime.startswith("text/")
        or mime in ("application/json", "application/xml")
        or mime.endswith(("+json", "+xml"))
    ):
        return "text"
    return None


def parse_charset(content_type: str | None) -> str | None:
    """Extract the charset parameter from a Content-Type header."""
    import re

    match = re.search(
        r';\s*charset\s*=\s*"?([^";]+)"?', content_type or "", re.IGNORECASE
    )
    return match.group(1).strip().lower() if match else None


def _decoder_for(charset: str | None) -> codecs.IncrementalDecoder:
    declared = charset or "utf-8"
    try:
        return codecs.getincrementaldecoder(declared)("replace")
    except LookupError as error:
        if any(
            label.strip().lower() in ("utf-8", "utf8") for label in declared.split(",")
        ):
            return codecs.getincrementaldecoder("utf-8")("replace")
        raise WebError(
            f'unsupported charset "{declared}"', "WEB_UNSUPPORTED_CONTENT_TYPE"
        ) from error


@dataclass
class HttpFetchLimits:
    """Resolved transport and response limits for the HTTP fetcher."""

    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    user_agent: str = DEFAULT_USER_AGENT


class HttpFetchProvider:
    """Anonymous HTTP(S) fetcher: bounded, same-origin redirects, text decoding."""

    id = "http"
    title: str | None = "HTTP"
    description: str | None = (
        "Free, reaches intranet/local addresses; no JavaScript or sign-in."
    )

    def __init__(self, limits: HttpFetchLimits | None = None) -> None:
        self.limits = limits or HttpFetchLimits()

    def available(self) -> bool:
        return True

    async def ready(self, llm_provider: str | None) -> bool:
        del llm_provider
        return True

    async def fetch(
        self, request: dict[str, Any], signal: Any = None
    ) -> dict[str, Any]:
        if signal is not None:
            signal.throw_if_aborted()
        current = validate_fetch_url(request.get("url"))
        redirected = 0
        timeout = self.limits.timeout_ms / 1000
        headers = {
            "user-agent": self.limits.user_agent,
            "accept": "text/html,application/xhtml+xml,text/*;q=0.9,application/json;q=0.8",
        }
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, headers=headers
        ) as client:
            while True:
                if signal is not None:
                    signal.throw_if_aborted()
                response = await client.get(current)
                if response.status_code in REDIRECT_STATUSES:
                    if redirected >= self.limits.max_redirects:
                        raise WebError(
                            f"exceeded the maximum of {self.limits.max_redirects} redirects",
                            "WEB_REDIRECT_BLOCKED",
                        )
                    location = response.headers.get("location")
                    if not location:
                        raise WebError(
                            f"redirect response (HTTP {response.status_code}) without a Location header",
                            "WEB_PROVIDER_ERROR",
                        )
                    target = validate_fetch_url(urljoin(current, location))
                    if not is_same_origin(target, current):
                        raise WebError(
                            f"cross-origin redirect to {urlparse(target).netloc} is not followed automatically",
                            "WEB_REDIRECT_BLOCKED",
                        )
                    current = target
                    redirected += 1
                    continue
                return self._read_body(response, current)

    def _read_body(self, response: httpx.Response, final_url: str) -> dict[str, Any]:
        content_type = response.headers.get("content-type")
        kind = classify_content_type(content_type)
        if kind is None:
            raise WebError(
                f'unsupported content type "{content_type or "unknown"}"',
                "WEB_UNSUPPORTED_CONTENT_TYPE",
            )
        decoder = _decoder_for(parse_charset(content_type))
        raw = response.content[: self.limits.max_response_bytes]
        truncated = len(response.content) > self.limits.max_response_bytes
        text = decoder.decode(raw, final=True)
        if len(text) > self.limits.max_body_chars:
            text = text[: self.limits.max_body_chars]
            truncated = True
        return {
            "url": final_url,
            "statusCode": response.status_code,
            "body": {"kind": kind, "content": text},
            "truncated": truncated,
        }


# ---------------------------------------------------------------------------
# Seam
# ---------------------------------------------------------------------------


@dataclass
class _ProviderInfo:
    id: str
    title: str | None
    description: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            **({} if self.title is None else {"title": self.title}),
            **({} if self.description is None else {"description": self.description}),
        }


class WebService(Service):
    """``ctx.web``: provider-neutral search/fetch with route-based selection."""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(ctx, "web")
        cfg = dict(config or {})
        self.search_provider = cfg.get("searchProvider") or os.environ.get(
            "DSH_WEB_SEARCH_PROVIDER"
        )
        self.fetch_provider = cfg.get("fetchProvider") or os.environ.get(
            "DSH_WEB_FETCH_PROVIDER"
        )
        self.max_results = int(cfg.get("maxResults", 5))
        self._search_providers: dict[str, WebSearchProvider] = {}
        self._fetch_providers: dict[str, WebFetchProvider] = {}

    def _emit_changed(self) -> None:
        try:
            self.ctx.emit("web/providers-changed", {})
        except Exception:  # noqa: BLE001, S110 -- observer failures cannot veto a registry mutation (dsh semantics)
            pass

    def register_search_provider(self, provider: WebSearchProvider) -> Any:
        """Register a search provider; reject duplicate ids."""
        if provider.id in self._search_providers:
            raise WebError(
                f'a web provider with id "{provider.id}" is already registered',
                "WEB_DUPLICATE_PROVIDER",
            )
        self._search_providers[provider.id] = provider
        self._emit_changed()

        def dispose() -> None:
            if self._search_providers.get(provider.id) is provider:
                self._search_providers.pop(provider.id)
                self._emit_changed()

        return dispose

    def register_fetch_provider(self, provider: WebFetchProvider) -> Any:
        """Register a fetch provider; reject duplicate ids."""
        if provider.id in self._fetch_providers:
            raise WebError(
                f'a web provider with id "{provider.id}" is already registered',
                "WEB_DUPLICATE_PROVIDER",
            )
        self._fetch_providers[provider.id] = provider
        self._emit_changed()

        def dispose() -> None:
            if self._fetch_providers.get(provider.id) is provider:
                self._fetch_providers.pop(provider.id)
                self._emit_changed()

        return dispose

    def list_providers(self) -> dict[str, list[dict[str, Any]]]:
        """List every registered provider (availability does not filter)."""
        return {
            "search": [
                _ProviderInfo(
                    p.id, getattr(p, "title", None), getattr(p, "description", None)
                ).as_dict()
                for p in self._search_providers.values()
            ],
            "fetch": [
                _ProviderInfo(
                    p.id, getattr(p, "title", None), getattr(p, "description", None)
                ).as_dict()
                for p in self._fetch_providers.values()
            ],
        }

    def _select(
        self,
        providers: dict[str, Any],
        configured: str | None,
        preferred: str | None,
    ) -> Any:
        if configured is not None:
            provider = providers.get(configured)
            if provider is None:
                raise WebError(
                    f'configured web provider "{configured}" is not registered',
                    "WEB_PROVIDER_CONFIGURED_MISSING",
                )
            if not provider.available():
                raise WebError(
                    f'configured web provider "{configured}" is registered but unavailable',
                    "WEB_PROVIDER_CONFIGURED_UNAVAILABLE",
                )
            return provider
        usable = [p for p in providers.values() if p.available()]
        chosen = next((p for p in usable if p.id == preferred), None)
        if chosen is not None:
            return chosen
        if not usable:
            raise WebError(
                "no usable web provider is registered", "WEB_PROVIDER_UNAVAILABLE"
            )
        if len(usable) > 1:
            ids = ", ".join(p.id for p in usable)
            raise WebError(
                f"multiple usable web providers are registered ({ids}); configure one explicitly",
                "WEB_PROVIDER_AMBIGUOUS",
            )
        return usable[0]

    def select_search_provider(self, llm_provider: str | None) -> Any:
        """Resolve the search provider for one model route."""
        preferred = (
            DEEPSEEK_ROUTE_SEARCH_PROVIDER
            if llm_provider in DEEPSEEK_MODEL_ROUTES
            else OTHER_ROUTE_SEARCH_PROVIDER
        )
        return self._select(self._search_providers, self.search_provider, preferred)

    def select_fetch_provider(self) -> Any:
        """Resolve the fetch provider."""
        return self._select(self._fetch_providers, self.fetch_provider, None)

    async def _ready(self, select: Any, llm_provider: str | None) -> bool:
        try:
            provider = select()
        except WebError:
            return True
        ready = getattr(provider, "ready", None)
        if ready is None:
            return True
        return bool(await ready(llm_provider))

    async def search_ready(self, llm_provider: str | None = None) -> bool:
        """Whether the selected search provider can authenticate for this route."""
        return await self._ready(
            lambda: self.select_search_provider(llm_provider), llm_provider
        )

    async def fetch_ready(self, llm_provider: str | None = None) -> bool:
        """Whether the selected fetch provider can authenticate for this route."""
        return await self._ready(self.select_fetch_provider, llm_provider)

    async def search(
        self, request: dict[str, Any], signal: Any = None
    ) -> dict[str, Any]:
        """Run one search and enforce the seam's result cap."""
        provider = self.select_search_provider(request.get("llmProvider"))
        result = await provider.search(request, signal)
        sources = result.get("sources")
        if not isinstance(sources, list):
            raise WebError(
                "search provider returned invalid sources", "WEB_SEARCH_INVALID_RESULT"
            )
        limit = request.get("maxResults")
        if limit is None:
            limit = self.max_results
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise WebError(
                "maxResults must be a positive integer", "WEB_INVALID_ARGUMENT"
            )
        truncated = bool(result.get("truncated", False))
        if len(sources) > limit:
            sources = sources[:limit]
            truncated = True
        return {
            **result,
            "sources": sources,
            "truncated": truncated,
        }

    async def fetch(
        self, request: dict[str, Any], signal: Any = None
    ) -> dict[str, Any]:
        """Retrieve one URL through the selected fetch provider."""
        provider = self.select_fetch_provider()
        return cast(dict[str, Any], await provider.fetch(request, signal))


def apply(ctx: Context, config: dict[str, Any] | None = None) -> WebService:
    """Mount ``ctx.web`` and register the anonymous HTTP fetch provider."""
    service = WebService(ctx, config)
    service.register_fetch_provider(HttpFetchProvider())
    return service


apply.inject = []  # type: ignore[attr-defined]

__all__ = [
    "DEEPSEEK_MODEL_ROUTES",
    "DEEPSEEK_ROUTE_SEARCH_PROVIDER",
    "DEFAULT_MAX_BODY_CHARS",
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_MS",
    "DEFAULT_USER_AGENT",
    "MAX_URL_LENGTH",
    "OTHER_ROUTE_SEARCH_PROVIDER",
    "HttpFetchLimits",
    "HttpFetchProvider",
    "WebError",
    "WebFetchProvider",
    "WebSearchProvider",
    "WebService",
    "apply",
    "classify_content_type",
    "is_same_origin",
    "parse_charset",
    "validate_fetch_url",
]
