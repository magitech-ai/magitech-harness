"""Tests for the web seam and web tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.session.events import ToolCallData
from magitech_harness.web.service import (
    WebError,
    WebService,
    http_fetch,
    is_public_http_url,
)
from magitech_harness.web.tools import FETCH_SECTION, SEARCH_SECTION


def test_public_url_checks() -> None:
    assert is_public_http_url("https://example.com/a")
    assert is_public_http_url("http://example.com")
    assert not is_public_http_url("ftp://example.com")
    assert not is_public_http_url("http://localhost:8000")
    assert not is_public_http_url("https://127.0.0.1/x")
    assert not is_public_http_url("not a url")


def test_html_to_text() -> None:
    from magitech_harness.web.service import _html_to_text

    text = _html_to_text(
        "<html><script>bad()</script><style>x{}</style>"
        "<p>Hello &amp; goodbye</p></html>"
    )
    assert "bad()" not in text
    assert "Hello & goodbye" in text


@pytest.mark.asyncio
async def test_http_fetch_uses_httpx() -> None:
    response = type(
        "R",
        (),
        {
            "headers": {"content-type": "text/html"},
            "text": "<p>Hi</p>",
            "status_code": 200,
        },
    )()
    client = AsyncMock()
    client.get.return_value = response
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    with patch("magitech_harness.web.service.httpx.AsyncClient", return_value=client):
        result = await http_fetch("https://example.com", 5.0)
    assert result["body"] == {"kind": "html", "content": "Hi"}
    assert result["statusCode"] == 200


@pytest.mark.asyncio
async def test_web_service_ready_and_disposal() -> None:
    ctx = Context()
    service = WebService(ctx)

    async def fetch(_url: str, _timeout: float) -> dict[str, Any]:
        return {
            "url": "x",
            "statusCode": 200,
            "body": {"kind": "text", "content": "ok"},
        }

    dispose = service.register_fetch("http", fetch)
    assert await service.fetch_ready() is True
    assert await service.search_ready() is False
    dispose()
    assert await service.fetch_ready() is False
    dispose()

    async def search(_queries: list[str], _max: int, _timeout: float) -> dict[str, Any]:
        return {"sources": [{"url": "https://example.com"}]}

    dispose_search = service.register_search("s", search)
    assert await service.search_ready() is True
    assert await service.search_ready("s") is True
    assert await service.search_ready("other") is False
    dispose_search()
    assert await service.search_ready() is False


@pytest.mark.asyncio
async def test_web_service_fetch_and_search() -> None:
    ctx = Context()
    service = WebService(ctx, {"max_output_chars": 4})

    async def fetch(url: str, _timeout: float) -> dict[str, Any]:
        return {
            "url": url,
            "statusCode": 201,
            "body": {"kind": "text", "content": "0123456789"},
        }

    async def search(
        queries: list[str], max_results: int, _timeout: float
    ) -> dict[str, Any]:
        return {
            "content": " ".join(queries),
            "sources": [{"url": "https://example.com"} for _ in range(max_results)],
            "truncated": False,
        }

    service.register_fetch("http", fetch)
    service.register_search("s", search)
    result = await service.fetch({"url": "https://example.com"})
    assert result["body"]["content"] == "0123"
    assert result["truncated"] is True
    result = await service.search(["a", "b"], "s")
    assert result["content"] == "a b"
    assert len(result["sources"]) == service.max_results


@pytest.mark.asyncio
async def test_web_service_errors() -> None:
    ctx = Context()
    service = WebService(ctx)
    with pytest.raises(WebError, match="public http"):
        await service.fetch({"url": "http://localhost"})
    with pytest.raises(WebError, match="no web fetch provider"):
        await service.fetch({"url": "https://example.com"})
    with pytest.raises(WebError, match="non-empty strings"):
        await service.search([], None)
    with pytest.raises(WebError, match="no web search provider"):
        await service.search(["a"], None)

    async def bad_fetch(_url: str, _timeout: float) -> dict[str, Any]:
        return {"body": {"kind": "bad", "content": "x"}}

    async def bad_search(
        _queries: list[str], _max: int, _timeout: float
    ) -> dict[str, Any]:
        return {"sources": "nope"}

    service.register_fetch("bad", bad_fetch)
    service.register_search("bad", bad_search)
    with pytest.raises(WebError, match="invalid body"):
        await service.fetch({"url": "https://example.com"})
    with pytest.raises(WebError, match="invalid sources"):
        await service.search(["a"], "bad")


@pytest.mark.asyncio
async def test_web_tool_paths(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "web-tools")
    await ctx.flush()
    session = ctx.sessions.create("web-tool-session")

    result = await ctx.tools.execute(
        ToolCallData(
            call_id="wf", name="web_fetch", arguments={"url": "http://localhost"}
        ),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is True

    result = await ctx.tools.execute(
        ToolCallData(call_id="ws", name="web_search", arguments={"queries": ["x"]}),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is True

    async def search(
        queries: list[str], max_results: int, _timeout: float
    ) -> dict[str, Any]:
        return {
            "content": None,
            "sources": [
                {
                    "url": f"https://example.com/{index}",
                    "title": f"Result {index}",
                    "snippet": "snippet",
                }
                for index in range(max_results)
            ],
            "truncated": False,
        }

    ctx.web.register_search("test", search)
    result = await ctx.tools.execute(
        ToolCallData(
            call_id="ws2",
            name="web_search",
            arguments={"queries": ["x"]},
        ),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["sources"]
    assert "Result 0" in str(result.content)
    assert FETCH_SECTION and SEARCH_SECTION


@pytest.mark.asyncio
async def test_web_fetch_success_via_tool(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "web-fetch-tool")
    await ctx.flush()
    session = ctx.sessions.create("web-fetch-session")

    async def fetch(url: str, _timeout: float) -> dict[str, Any]:
        return {
            "url": url,
            "statusCode": 200,
            "body": {"kind": "text", "content": "body"},
            "truncated": False,
        }

    ctx.web.register_fetch("http", fetch)
    result = await ctx.tools.execute(
        ToolCallData(
            call_id="wf2",
            name="web_fetch",
            arguments={"url": "https://example.com"},
        ),
        session=session,
        signal=AbortSignal(),
    )
    assert result.is_error is False
    assert result.content["body"]["content"] == "body"


def test_mount_shape(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "web-mount")
    assert isinstance(ctx.web, WebService)
    names = {definition["name"] for definition in ctx.tools.definitions()}
    assert {"web_search", "web_fetch"} <= names


@pytest.mark.asyncio
async def test_search_disposer_is_idempotent() -> None:
    ctx = Context()
    service = WebService(ctx)

    async def search(_queries: list[str], _max: int, _timeout: float) -> dict[str, Any]:
        return {"sources": []}

    dispose = service.register_search("s", search)
    dispose()
    dispose()  # second call: provider already gone (covers the no-op branch)
    assert await service.search_ready() is False


@pytest.mark.asyncio
async def test_fetch_missing_named_provider() -> None:
    ctx = Context()
    service = WebService(ctx)

    async def fetch(_url: str, _timeout: float) -> dict[str, Any]:
        return {"body": {"kind": "text", "content": "ok"}}

    service.register_fetch("http", fetch)
    with pytest.raises(WebError, match="no web fetch provider named"):
        await service.fetch({"url": "https://example.com"}, "missing")


def test_render_helpers() -> None:
    from magitech_harness.web.tools import _render_fetch, _render_search

    # Search with content, snippets, and truncation.
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
    text = rendered[0]["text"]
    assert "summary" in text
    assert "[A](https://a): snip" in text
    assert "[https://b](https://b)" in text
    assert "(results truncated)" in text

    # Search without content/snippet/truncation.
    rendered = _render_search({}, {"sources": [{"url": "https://c"}]})
    assert rendered[0]["text"] == "- [https://c](https://c)"

    # Fetch with and without truncation.
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
    assert "body\n(output truncated)" in rendered[0]["text"]
    rendered = _render_fetch({}, {})
    assert rendered[0]["text"] == " [0]\n"


def test_apply_without_system_prompt() -> None:
    from magitech_harness.web.tools import apply as apply_tools

    ctx = Context()
    registered: list[Any] = []
    ctx.provide("tools", type("T", (), {"register": registered.append})())
    ctx.provide("web", type("W", (), {"search_timeout_ms": 60000})())
    apply_tools(ctx, {"fetch": True})
    assert len(registered) == 2


def test_apply_fetch_disabled() -> None:
    from magitech_harness.web.tools import apply as apply_tools

    registered: list[Any] = []
    sections: list[Any] = []
    ctx = Context()
    ctx.provide("tools", type("T", (), {"register": registered.append})())
    ctx.provide("web", type("W", (), {"search_timeout_ms": 60000})())
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
