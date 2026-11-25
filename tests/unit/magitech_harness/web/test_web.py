"""Tests for the web seam and web tools (dsh-web parity)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.session.events import ToolCallData
from magitech_harness.web.service import (
    DEEPSEEK_MODEL_ROUTES,
    DEFAULT_MAX_REDIRECTS,
    HttpFetchLimits,
    HttpFetchProvider,
    WebError,
    WebService,
    classify_content_type,
    is_same_origin,
    parse_charset,
    validate_fetch_url,
)
from magitech_harness.web.tools import (
    FETCH_SECTION,
    SEARCH_SECTION,
    _merge_results,
    _render_fetch,
    _render_search,
    _route_of,
)


class FakeSearch:
    def __init__(
        self,
        identifier: str = "s",
        available: bool = True,
        ready: bool = True,
        result: dict[str, Any] | None = None,
        has_ready: bool = True,
    ) -> None:
        self.id = identifier
        self.title = "Search"
        self.description = "fake"
        self._available = available
        self._ready = ready
        self._result = result if result is not None else {"sources": []}
        self.calls: list[dict[str, Any]] = []
        if not has_ready:
            self.ready = None  # type: ignore[assignment]

    def available(self) -> bool:
        return self._available

    async def ready(self, llm_provider: str | None) -> bool:
        del llm_provider
        return self._ready

    async def search(
        self, request: dict[str, Any], signal: Any = None
    ) -> dict[str, Any]:
        del signal
        self.calls.append(request)
        return self._result


class FakeFetch:
    def __init__(
        self,
        identifier: str = "f",
        available: bool = True,
        result: dict[str, Any] | None = None,
    ) -> None:
        self.id = identifier
        self.title = None
        self.description = None
        self._available = available
        self._result = result or {
            "url": "https://example.com",
            "statusCode": 200,
            "body": {"kind": "text", "content": "ok"},
            "truncated": False,
        }
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return self._available

    async def ready(self, llm_provider: str | None) -> bool:
        del llm_provider
        return True

    async def fetch(
        self, request: dict[str, Any], signal: Any = None
    ) -> dict[str, Any]:
        del signal
        self.calls.append(request)
        return self._result


def _response(
    status: int, content_type: str | None, body: bytes, location: str | None = None
) -> SimpleNamespace:
    headers = {"content-type": content_type} if content_type is not None else {}
    if location is not None:
        headers["location"] = location
    return SimpleNamespace(status_code=status, headers=headers, content=body)


def _client(responses: list[SimpleNamespace]) -> AsyncMock:
    client = AsyncMock()
    client.get.side_effect = responses
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


# -- pure helpers ----------------------------------------------------------


def test_validate_fetch_url() -> None:
    assert validate_fetch_url("https://example.com/a") == "https://example.com/a"
    assert validate_fetch_url("http://example.com") == "http://example.com"
    with pytest.raises(WebError, match="invalid URL"):
        validate_fetch_url(123)  # type: ignore[arg-type]
    with pytest.raises(WebError, match="maximum length"):
        validate_fetch_url("https://example.com/" + "a" * MAX_URL_LENGTH)
    with pytest.raises(WebError, match="unsupported URL scheme"):
        validate_fetch_url("ftp://example.com")
    with pytest.raises(WebError, match="credentials in URLs"):
        validate_fetch_url("https://user:pass@example.com")
    with pytest.raises(WebError, match="invalid URL"):
        validate_fetch_url("https:///missing")


MAX_URL_LENGTH = 2048


def test_is_same_origin() -> None:
    assert is_same_origin("https://a.com/x", "https://a.com/y")
    assert is_same_origin("http://a.com:80/x", "http://a.com/y")
    assert not is_same_origin("https://a.com", "http://a.com")
    assert not is_same_origin("https://a.com", "https://b.com")
    assert not is_same_origin("https://a.com:443", "https://a.com:8443")


def test_classify_content_type() -> None:
    assert classify_content_type("text/html; charset=utf-8") == "html"
    assert classify_content_type("application/xhtml+xml") == "html"
    assert classify_content_type("text/plain") == "text"
    assert classify_content_type("application/json") == "text"
    assert classify_content_type("application/xml") == "text"
    assert classify_content_type("application/ld+json") == "text"
    assert classify_content_type("application/atom+xml") == "text"
    assert classify_content_type("application/octet-stream") is None
    assert classify_content_type(None) is None


def test_parse_charset() -> None:
    assert parse_charset("text/html; charset=UTF-8") == "utf-8"
    assert parse_charset('text/html; charset="iso-8859-1"') == "iso-8859-1"
    assert parse_charset("text/html") is None


# -- HTTP fetch provider ---------------------------------------------------


@pytest.mark.asyncio
async def test_http_fetch_html() -> None:
    provider = HttpFetchProvider()
    client = _client([_response(200, "text/html; charset=utf-8", b"<p>Hello</p>")])
    with patch("magitech_harness.web.service.httpx.AsyncClient", return_value=client):
        result = await provider.fetch({"url": "https://example.com"})
    assert result["body"] == {"kind": "html", "content": "<p>Hello</p>"}
    assert result["statusCode"] == 200
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_http_fetch_byte_and_char_caps() -> None:
    provider = HttpFetchProvider(
        HttpFetchLimits(max_response_bytes=4, max_body_chars=3)
    )
    client = _client([_response(200, "text/plain", b"0123456789")])
    with patch("magitech_harness.web.service.httpx.AsyncClient", return_value=client):
        result = await provider.fetch({"url": "https://example.com"})
    assert result["body"]["content"] == "012"
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_http_fetch_unsupported_content_type() -> None:
    provider = HttpFetchProvider()
    client = _client([_response(200, None, b"binary")])
    with (
        patch("magitech_harness.web.service.httpx.AsyncClient", return_value=client),
        pytest.raises(WebError, match="unsupported content type"),
    ):
        await provider.fetch({"url": "https://example.com"})


@pytest.mark.asyncio
async def test_http_fetch_redirects() -> None:
    provider = HttpFetchProvider()
    client = _client(
        [
            _response(302, None, b"", "/next"),
            _response(200, "text/plain", b"done"),
        ]
    )
    with patch("magitech_harness.web.service.httpx.AsyncClient", return_value=client):
        result = await provider.fetch({"url": "https://example.com/a"})
    assert result["url"] == "https://example.com/next"
    assert result["body"]["content"] == "done"


@pytest.mark.asyncio
async def test_http_fetch_redirect_errors() -> None:
    provider = HttpFetchProvider(HttpFetchLimits(max_redirects=0))
    with (
        patch(
            "magitech_harness.web.service.httpx.AsyncClient",
            return_value=_client([_response(302, None, b"", "/next")]),
        ),
        pytest.raises(WebError, match="maximum of 0 redirects"),
    ):
        await provider.fetch({"url": "https://example.com"})
    normal = HttpFetchProvider()
    with (
        patch(
            "magitech_harness.web.service.httpx.AsyncClient",
            return_value=_client([_response(302, None, b"")]),
        ),
        pytest.raises(WebError, match="without a Location"),
    ):
        await normal.fetch({"url": "https://example.com"})
    with (
        patch(
            "magitech_harness.web.service.httpx.AsyncClient",
            return_value=_client([_response(302, None, b"", "https://other.com/x")]),
        ),
        pytest.raises(WebError, match="cross-origin redirect"),
    ):
        await normal.fetch({"url": "https://example.com"})


@pytest.mark.asyncio
async def test_http_fetch_abort() -> None:
    provider = HttpFetchProvider()
    signal = AbortSignal()
    signal.abort()
    with pytest.raises(AbortError):
        await provider.fetch({"url": "https://example.com"}, signal)


@pytest.mark.asyncio
async def test_http_fetch_abort_during_redirect() -> None:
    provider = HttpFetchProvider()
    signal = AbortSignal()
    client = _client([_response(302, None, b"", "/next")])

    async def abort_then_respond(_url: str) -> SimpleNamespace:
        signal.abort()
        return _response(302, None, b"", "/next")

    client.get.side_effect = abort_then_respond
    with (
        patch("magitech_harness.web.service.httpx.AsyncClient", return_value=client),
        pytest.raises(AbortError),
    ):
        await provider.fetch({"url": "https://example.com"}, signal)


@pytest.mark.asyncio
async def test_http_fetch_invalid_decoder() -> None:
    from magitech_harness.web.service import _decoder_for

    decoder = _decoder_for("bogus-charset, utf-8")
    assert decoder.decode(b"hi", final=True) == "hi"
    with pytest.raises(WebError, match="unsupported charset"):
        _decoder_for("bogus-charset")


# -- seam ------------------------------------------------------------------


def test_registry_and_events() -> None:
    ctx = Context()
    service = WebService(ctx)
    events: list[Any] = []
    ctx.on("web/providers-changed", lambda *a: events.append(True), global_=True)

    search = FakeSearch()
    dispose_search = service.register_search_provider(search)
    fetch = FakeFetch()
    dispose_fetch = service.register_fetch_provider(fetch)
    assert len(service.list_providers()["search"]) == 1
    assert service.list_providers()["fetch"][0]["id"] == "f"
    assert len(events) == 2

    with pytest.raises(WebError, match="already registered"):
        service.register_search_provider(FakeSearch())
    with pytest.raises(WebError, match="already registered"):
        service.register_fetch_provider(FakeFetch())

    dispose_search()
    dispose_search()
    dispose_fetch()
    dispose_fetch()
    assert service.list_providers()["search"] == []
    assert len(events) == 4

    # A listener that throws must not veto the mutation.
    def bad_listener(*args: Any) -> None:
        raise RuntimeError("boom")

    ctx.on("web/providers-changed", bad_listener, global_=True)
    service.register_search_provider(FakeSearch())


def test_selection_rules() -> None:
    ctx = Context()
    service = WebService(ctx)
    service.register_search_provider(FakeSearch("s1", available=False))

    with pytest.raises(WebError, match="no usable web provider"):
        service.select_search_provider(None)

    service.register_search_provider(FakeSearch("deepseek-official"))
    assert service.select_search_provider("deepseek-official").id == "deepseek-official"
    assert service.select_search_provider("kilo").id == "deepseek-official"

    other = FakeSearch("deepseek-search-api")
    service.register_search_provider(other)
    assert service.select_search_provider("kilo").id == "deepseek-search-api"
    service.search_provider = "missing"
    with pytest.raises(WebError, match="configured .* is not registered"):
        service.select_search_provider(None)
    service.search_provider = "s1"
    with pytest.raises(WebError, match="configured .* unavailable"):
        service.select_search_provider(None)
    service.search_provider = None

    ambiguous = WebService(ctx)
    ambiguous.register_search_provider(FakeSearch("a"))
    ambiguous.register_search_provider(FakeSearch("b"))
    with pytest.raises(WebError, match="multiple usable web providers"):
        ambiguous.select_search_provider(None)


@pytest.mark.asyncio
async def test_readiness() -> None:
    ctx = Context()
    service = WebService(ctx)
    from magitech_harness.web.service import DEEPSEEK_ROUTE_SEARCH_PROVIDER

    service.register_search_provider(
        FakeSearch(DEEPSEEK_ROUTE_SEARCH_PROVIDER, ready=False, has_ready=True)
    )
    assert await service.search_ready(None) is False
    service.register_search_provider(FakeSearch("no-method", has_ready=False))
    assert await service.search_ready("kilo") is True
    assert await service.search_ready("deepseek-official") is False
    assert await service.fetch_ready(None) is True

    fetch = FakeFetch()
    service.register_fetch_provider(fetch)
    assert await service.fetch_ready(None) is True

    unavailable = WebService(ctx, {"searchProvider": "missing"})
    assert await unavailable.search_ready(None) is True

    no_ready_ctx = Context()
    no_ready = WebService(no_ready_ctx)
    no_ready.register_search_provider(FakeSearch(has_ready=False))
    assert await no_ready.search_ready(None) is True


@pytest.mark.asyncio
async def test_search_cap_and_errors() -> None:
    ctx = Context()
    service = WebService(ctx)
    provider = FakeSearch(
        result={"sources": [{"url": f"https://a/{i}"} for i in range(10)]}
    )
    service.register_search_provider(provider)
    result = await service.search({"query": "x", "maxResults": 3})
    assert len(result["sources"]) == 3
    assert result["truncated"] is True
    assert provider.calls[0]["query"] == "x"

    result = await service.search({"query": "y"})
    assert len(result["sources"]) == service.max_results

    bad = WebService(ctx)
    bad.register_search_provider(FakeSearch(result={"sources": "nope"}))
    with pytest.raises(WebError, match="invalid sources"):
        await bad.search({"query": "x"})
    with pytest.raises(WebError, match="maxResults must be"):
        await service.search({"query": "x", "maxResults": 0})


@pytest.mark.asyncio
async def test_fetch_delegates() -> None:
    ctx = Context()
    service = WebService(ctx)
    provider = FakeFetch()
    service.register_fetch_provider(provider)
    result = await service.fetch({"url": "https://example.com"})
    assert result["statusCode"] == 200
    assert provider.calls[0]["url"] == "https://example.com"


def test_env_selection() -> None:
    with patch.dict("os.environ", {"DSH_WEB_SEARCH_PROVIDER": "env"}):
        service = WebService(Context())
        assert service.search_provider == "env"
    assert DEEPSEEK_MODEL_ROUTES == {"deepseek-official", "deepseek-account"}
    assert DEFAULT_MAX_REDIRECTS == 5


# -- tools -----------------------------------------------------------------


def test_route_of() -> None:
    session = SimpleNamespace(request_context=lambda: {"provider": "kilo"})
    agent = SimpleNamespace(session=session, options={})
    assert _route_of({"agent": agent}) == "kilo"
    agent = SimpleNamespace(
        session=SimpleNamespace(request_context=lambda: None), options={"provider": "x"}
    )
    assert _route_of({"agent": agent}) == "x"
    assert _route_of({}) is None
    assert _route_of(None) is None


def test_render_helpers() -> None:
    rendered = _render_search(
        {},
        {
            "content": "summary",
            "sources": [
                {"url": "https://a", "title": "A", "snippet": "snip"},
                {"url": "https://b"},
            ],
            "truncated": True,
        },
    )
    assert "[A](https://a): snip" in rendered[0]["text"]
    assert "(results truncated)" in rendered[0]["text"]
    assert (
        _render_search({}, {"sources": [{"url": "https://c"}]})[0]["text"]
        == "- [https://c](https://c)"
    )

    rendered = _render_fetch(
        {},
        {
            "url": "https://example.com",
            "statusCode": 200,
            "body": {"kind": "text", "content": "body"},
            "truncated": True,
        },
    )
    assert "https://example.com [200]" in rendered[0]["text"]
    assert "(output truncated)" in rendered[0]["text"]
    assert _render_fetch({}, {})[0]["text"] == " [0]\n"


def test_merge_results() -> None:
    merged = _merge_results(
        [
            {"content": "a", "sources": [{"url": "https://a"}], "truncated": False},
            {
                "content": "b",
                "sources": [{"url": "https://a"}, {"url": "https://b"}],
                "truncated": True,
            },
        ]
    )
    assert merged["content"] == "a\nb"
    assert len(merged["sources"]) == 2
    assert merged["truncated"] is True
    assert _merge_results([{"sources": [], "truncated": False}]) == {
        "sources": [],
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_apply_without_system_prompt() -> None:
    from magitech_harness.web.tools import apply as apply_tools

    ctx = Context()
    registered: list[Any] = []
    ctx.provide(
        "tools",
        type(
            "T",
            (),
            {
                "register": lambda self, definition, options=None: registered.append(
                    definition
                )
            },
        )(),
    )
    ctx.provide("web", type("W", (), {"max_results": 5})())
    apply_tools(ctx, {"fetch": True})
    assert len(registered) == 2


@pytest.mark.asyncio
async def test_apply_fetch_disabled() -> None:
    from magitech_harness.web.tools import apply as apply_tools

    registered: list[Any] = []
    sections: list[Any] = []
    ctx = Context()
    ctx.provide(
        "tools",
        type(
            "T",
            (),
            {
                "register": lambda self, definition, options=None: registered.append(
                    definition
                )
            },
        )(),
    )
    ctx.provide("web", type("W", (), {"max_results": 5})())
    ctx.provide(
        "systemPrompt",
        type(
            "SP",
            (),
            {
                "section": lambda self, section: sections.append(section),
                "get_section_order": lambda self, name: 1000,
            },
        )(),
    )
    apply_tools(ctx, {"fetch": False})
    assert [definition["name"] for definition in registered] == ["web_search"]
    assert [section["name"] for section in sections] == ["tool:web_search"]


@pytest.mark.asyncio
async def test_tool_admission_and_execution(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "web-tools")
    await ctx.flush()
    session = ctx.sessions.create("web-tool-session")
    agent = SimpleNamespace(
        session=session,
        options={},
        id="root",
    )

    # No search provider: admission predicate reports True (selection failure).
    resolved = await ctx.tools.resolve(
        {"agent": agent, "modelSelection": {}, "signal": AbortSignal()}
    )
    names = {
        definition["name"]
        for definition in ctx.tools.definitions(agent, tool_set=resolved)
    }
    assert "web_fetch" in names

    result = await ctx.tools.execute(
        ToolCallData(call_id="wf", name="web_fetch", arguments={"url": "ftp://x"}),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is True

    search = FakeSearch(
        result={
            "content": None,
            "sources": [{"url": "https://example.com", "title": "Result 0"}],
        }
    )
    ctx.web.register_search_provider(search)
    result = await ctx.tools.execute(
        ToolCallData(call_id="ws", name="web_search", arguments={"queries": ["x"]}),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["sources"][0]["title"] == "Result 0"

    fetch = FakeFetch()
    ctx.web.register_fetch_provider(fetch)
    ctx.web.fetch_provider = "f"
    result = await ctx.tools.execute(
        ToolCallData(
            call_id="wf2", name="web_fetch", arguments={"url": "https://example.com"}
        ),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["body"]["content"] == "ok"
    assert FETCH_SECTION and SEARCH_SECTION


def test_mount_shape(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "web-mount")
    assert isinstance(ctx.web, WebService)
    providers = ctx.web.list_providers()
    assert providers["fetch"][0]["id"] == "http"
    assert providers["search"] == []
    names = {definition["name"] for definition in ctx.tools.definitions()}
    assert {"web_search", "web_fetch"} <= names
