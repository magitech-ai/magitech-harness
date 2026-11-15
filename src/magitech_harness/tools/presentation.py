"""Structured render-intent types for tool call and result presentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, NotRequired, TypedDict

from magitech_harness.llm.types import ContentBlock

ToolCallKind = Literal[
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "fetch",
    "other",
]


class FileLocation(TypedDict):
    """Exact source coordinate associated with a tool inspection or match."""

    path: str
    line: NotRequired[int]


class FileDiff(TypedDict):
    """Single-file change derived from tool arguments or applied result."""

    path: str
    oldText: str | None
    newText: str


class GenericCallView(TypedDict):
    """Default card for pending tool calls with optional kind, input, and locations."""

    card: Literal["generic"]
    title: str
    kind: NotRequired[ToolCallKind]
    rawInput: NotRequired[Any]
    content: NotRequired[Sequence[ContentBlock]]
    locations: NotRequired[Sequence[FileLocation]]


class TerminalCallView(TypedDict):
    """Terminal command invocation call view."""

    card: Literal["terminal"]
    title: str
    description: NotRequired[str]
    cwd: NotRequired[str]


class DiffCallView(TypedDict):
    """File diff invocation call view."""

    card: Literal["diff"]
    title: str
    diffs: Sequence[FileDiff]
    locations: NotRequired[Sequence[FileLocation]]


ToolCallView = GenericCallView | TerminalCallView | DiffCallView


class ReadFileLine(TypedDict):
    """One 1-based numbered file line inside a structured read result."""

    number: int
    text: str


class GenericResultView(TypedDict):
    """Completed generic card with optional replacement title and formatted blocks."""

    card: Literal["generic"]
    title: NotRequired[str]
    content: NotRequired[Sequence[ContentBlock]]


class TerminalResultView(TypedDict):
    """Completed terminal command result presentation."""

    card: Literal["terminal"]
    title: NotRequired[str]
    output: NotRequired[str]
    exitCode: NotRequired[int]
    signal: NotRequired[str]


class DiffResultView(TypedDict):
    """Applied file edits presentation view."""

    card: Literal["diff"]
    title: NotRequired[str]
    diffs: Sequence[FileDiff]


class SearchLineMatch(TypedDict):
    """One matched line inside a grouped file match entry."""

    lineNumber: int
    line: str


class SearchFileMatches(TypedDict):
    """Grouped content matches for one file inside a search matches result."""

    path: str
    matches: Sequence[SearchLineMatch]


class SearchMatchesResultView(TypedDict):
    """Line-match search presentation view."""

    card: Literal["search"]
    shape: Literal["matches"]
    title: NotRequired[str]
    files: Sequence[SearchFileMatches]
    truncated: bool
    total: int


class SearchPathsResultView(TypedDict):
    """File-path listing search presentation view."""

    card: Literal["search"]
    shape: Literal["paths"]
    title: NotRequired[str]
    paths: Sequence[str]
    truncated: bool
    total: int


SearchResultView = SearchMatchesResultView | SearchPathsResultView


class ReadResultView(TypedDict):
    """Structured file read window presentation view."""

    card: Literal["read"]
    title: NotRequired[str]
    path: str
    offset: int
    lines: Sequence[ReadFileLine]
    totalLines: int
    lang: NotRequired[str]
    content: NotRequired[Sequence[ContentBlock]]


class WebSource(TypedDict):
    """One citeable web search source projection."""

    url: str
    title: NotRequired[str]
    snippet: NotRequired[str]
    publishedAt: NotRequired[str]


class WebSearchResultView(TypedDict):
    """Web search presentation view."""

    card: Literal["web"]
    kind: Literal["search"]
    title: NotRequired[str]
    sources: Sequence[WebSource]
    answer: NotRequired[str]
    truncated: bool


class WebFetchResultView(TypedDict):
    """Fetched web page presentation view."""

    card: Literal["web"]
    kind: Literal["fetch"]
    title: NotRequired[str]
    url: str
    statusCode: int
    truncated: bool


WebResultView = WebSearchResultView | WebFetchResultView

ToolResultView = (
    GenericResultView
    | TerminalResultView
    | DiffResultView
    | SearchResultView
    | ReadResultView
    | WebResultView
)

__all__ = [
    "DiffCallView",
    "DiffResultView",
    "FileDiff",
    "FileLocation",
    "GenericCallView",
    "GenericResultView",
    "ReadFileLine",
    "ReadResultView",
    "SearchFileMatches",
    "SearchLineMatch",
    "SearchMatchesResultView",
    "SearchPathsResultView",
    "SearchResultView",
    "TerminalCallView",
    "TerminalResultView",
    "ToolCallKind",
    "ToolCallView",
    "ToolResultView",
    "WebFetchResultView",
    "WebResultView",
    "WebSearchResultView",
    "WebSource",
]
